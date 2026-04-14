/**
 * Generic WGSL model runner — executes any model from its graph JSON + weights.
 * Automatically allocates buffers, wires dispatches, handles ping-pong.
 *
 * Usage:
 *   const runner = new ModelRunner(device, pipelines, weightBufs);
 *   const outputs = await runner.run(graphJson, inputBuffer);
 */

export class ModelRunner {
  constructor(device, pipelines, weightBufs, allWeightsBuf) {
    this.device = device;
    this.P = pipelines;     // { conv2d, maxpool, resize, gemm, global_avg_pool, add, fused_block }
    this.W = weightBufs;    // weight name -> GPUBuffer
    this.allWeightsBuf = allWeightsBuf; // single GPU buffer with ALL weights (for fused shader)
    this.dummy = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
    // Uniform buffer pool: reuse instead of creating per dispatch
    this._ubPool = [];
    this._ubIdx = 0;
  }

  _getUniformBuf(byteLength) {
    // Round up to 16-byte alignment
    const size = Math.ceil(byteLength / 16) * 16;
    if (this._ubIdx < this._ubPool.length && this._ubPool[this._ubIdx].size >= size) {
      return this._ubPool[this._ubIdx++];
    }
    const buf = this.device.createBuffer({ size: Math.max(size, 64), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    if (this._ubIdx < this._ubPool.length) {
      this._ubPool[this._ubIdx].destroy();
      this._ubPool[this._ubIdx] = buf;
    } else {
      this._ubPool.push(buf);
    }
    this._ubIdx++;
    return buf;
  }

  /**
   * Run a model end-to-end.
   * @param {Object} graph - parsed graph JSON
   * @param {GPUBuffer} inputBuf - input in NHWC format
   * @returns {Object} output name -> Float32Array
   */
  async run(graph, inputBuf, allWeights, debug = false) {
    const g = graph.graph;
    const w = graph.weights;
    const device = this.device;
    const BF = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    // Propagate shapes to know buffer sizes
    const shapes = {};
    shapes[graph.input.name] = graph.input.shape;
    for (const [name, info] of Object.entries(w)) shapes[name] = info.shape;

    // Tensor name -> GPUBuffer
    const tensors = {};
    tensors[graph.input.name] = inputBuf;

    const getOrAlloc = (name, shape) => {
      if (tensors[name]) return tensors[name];
      let floats = 1;
      for (const d of shape) floats *= d;
      const buf = device.createBuffer({ size: Math.max(floats * 4, 4), usage: BF });
      tensors[name] = buf;
      return buf;
    };

    this._ubIdx = 0; // reset uniform buffer pool for this run

    // --- Fused block pattern detection ---
    // Pattern: Conv(DW) -> Conv(1x1) -> [Pad ->] Add -> Relu/PRelu/Clip
    // Mark fuseable sequences so the main loop can dispatch fused_block instead.
    const fusedBlocks = new Map(); // start index -> { dwIdx, pwIdx, padIdx?, addIdx, actIdx, actType }
    const skipNodes = new Set();
    if (this.P.fused_block && this.allWeightsBuf) {
      for (let i = 0; i < g.length; i++) {
        if (skipNodes.has(i)) continue;
        const dw = g[i];
        if (dw.op !== 'Conv') continue;
        const dwW = w[dw.inputs[1]];
        if (!dwW) continue;
        // DW conv: group == input channels, kernel > 1
        const dwGroup = dw.attrs?.group || 1;
        if (dwGroup <= 1 || dwW.shape[2] <= 1) continue; // not depthwise

        // Next: 1x1 Conv consuming DW output, or Clip/Relu then 1x1 Conv
        let j = i + 1;
        let dwActIdx = -1, dwActType = 0;
        let dwOutName = dw.outputs[0];
        if (j < g.length && (g[j].op === 'Clip' || g[j].op === 'Relu') && g[j].inputs[0] === dwOutName) {
          dwActIdx = j;
          dwActType = g[j].op === 'Clip' ? 2 : 3; // 2=ReLU6, 3=ReLU
          dwOutName = g[j].outputs[0];
          j++;
        }
        if (j >= g.length || g[j].op !== 'Conv') continue;
        if (g[j].inputs[0] !== dwOutName) continue;
        const pwW = w[g[j].inputs[1]];
        if (!pwW || pwW.shape[2] !== 1 || pwW.shape[3] !== 1) continue; // not 1x1

        // Skip fusion when 1x1 narrows channels significantly -- the fused shader
        // recomputes DW for every output channel, so narrow 1x1 causes massive
        // redundant DW compute (e.g. 96ch DW -> 16ch 1x1 = 6x redundant work).
        // Only fuse when output channels >= input channels (widening or same-size).
        const pwOutCh = pwW.shape[0];
        const dwInCh = dwW.shape[0];
        if (pwOutCh < dwInCh) continue;

        // Now look for Add and optional Pad, then activation
        let k = j + 1;
        let padIdx = -1, addIdx = -1, actIdx = -1, actType = 0;

        // Optional Pad (on the residual path, not after 1x1)
        // The Add's inputs are: (padded_residual, conv1x1_output) or (residual, conv1x1_output)
        // We need to find the Add that consumes the 1x1 output
        if (k < g.length) {
          // Scan ahead a few nodes for the Add that uses the 1x1 output
          for (let look = k; look < Math.min(k + 3, g.length); look++) {
            if (g[look].op === 'Add' && (g[look].inputs[0] === g[j].outputs[0] || g[look].inputs[1] === g[j].outputs[0])) {
              addIdx = look;
              break;
            }
          }
        }
        if (addIdx === -1) continue; // no residual Add found

        // Check for Pad between 1x1 and Add (on the OTHER input to Add)
        const addOtherInput = g[addIdx].inputs[0] === g[j].outputs[0] ? g[addIdx].inputs[1] : g[addIdx].inputs[0];

        // Skip if residual path goes through MaxPool (can't fuse MaxPool into the block)
        const residualProducer = g.find(n => n.outputs[0] === addOtherInput);
        if (residualProducer && residualProducer.op === 'MaxPool') continue;

        for (let look = j + 1; look < addIdx; look++) {
          if (g[look].op === 'Pad' && g[look].outputs[0] === addOtherInput) {
            padIdx = look;
            break;
          }
        }

        // Check for activation after Add
        const nextAfterAdd = addIdx + 1;
        if (nextAfterAdd < g.length) {
          const act = g[nextAfterAdd];
          if (act.inputs[0] === g[addIdx].outputs[0]) {
            if (act.op === 'Relu') { actIdx = nextAfterAdd; actType = 3; }
            else if (act.op === 'Clip') { actIdx = nextAfterAdd; actType = 2; }
            else if (act.op === 'PRelu') { actIdx = nextAfterAdd; actType = 1; }
          }
        }

        // We have a fuseable block!
        fusedBlocks.set(i, { dwIdx: i, dwActIdx, dwActType, pwIdx: j, padIdx, addIdx, actIdx, actType });
        skipNodes.add(i);   // DW conv
        if (dwActIdx >= 0) skipNodes.add(dwActIdx); // DW activation
        skipNodes.add(j);   // 1x1 conv
        if (padIdx >= 0) skipNodes.add(padIdx);
        skipNodes.add(addIdx);
        if (actIdx >= 0) skipNodes.add(actIdx);
      }
      if (fusedBlocks.size > 0 && !this._fusionLogged) {
        console.log(`[fusion] ${fusedBlocks.size} blocks fused, ${skipNodes.size} nodes skipped`);
        this._fusionLogged = true;
      }
    }

    let enc = device.createCommandEncoder();

    for (let i = 0; i < g.length; i++) {
      if (skipNodes.has(i) && !fusedBlocks.has(i)) continue; // skip nodes consumed by fusion
      const node = g[i];
      const op = node.op;
      const attrs = node.attrs || {};
      const inp = node.inputs;
      const out = node.outputs[0];

      // --- Fused residual block dispatch ---
      if (fusedBlocks.has(i)) {
        const fb = fusedBlocks.get(i);
        const dwNode = g[fb.dwIdx];
        const pwNode = g[fb.pwIdx];
        const addNode = g[fb.addIdx];
        const dwAttrs = dwNode.attrs || {};
        const dwWName = dwNode.inputs[1];
        const dwBName = dwNode.inputs[2];
        const pwWName = pwNode.inputs[1];
        const pwBName = pwNode.inputs[2];
        const dwWShape = w[dwWName].shape;
        const pwWShape = w[pwWName].shape;
        const inShape = shapes[dwNode.inputs[0]];
        const stride = dwAttrs.strides?.[0] || 1;
        const [, iC, iH, iW] = inShape;
        const kSize = dwWShape[2];
        const pads = dwAttrs.pads || [0, 0, 0, 0]; // [top, left, bottom, right]
        const padT = pads[0], padL = pads[1], padB = pads[2], padR = pads[3];
        const oC = pwWShape[0];
        const oH = Math.floor((iH + padT + padB - kSize) / stride) + 1;
        const oW = Math.floor((iW + padL + padR - kSize) / stride) + 1;

        // Figure out final output name and shape
        const finalOut = fb.actIdx >= 0 ? g[fb.actIdx].outputs[0] : addNode.outputs[0];
        const outShape = [1, oC, oH, oW];
        shapes[finalOut] = outShape;
        // Also set shapes for intermediate nodes so downstream can reference them
        shapes[dwNode.outputs[0]] = [1, iC, oH, oW];
        if (fb.dwActIdx >= 0) shapes[g[fb.dwActIdx].outputs[0]] = [1, iC, oH, oW];
        shapes[pwNode.outputs[0]] = [1, oC, oH, oW];
        shapes[addNode.outputs[0]] = outShape;

        const outBuf = getOrAlloc(finalOut, outShape);
        // Set all intermediate tensor names to point to the same output
        tensors[dwNode.outputs[0]] = outBuf;
        if (fb.dwActIdx >= 0) tensors[g[fb.dwActIdx].outputs[0]] = outBuf;
        tensors[pwNode.outputs[0]] = outBuf;
        tensors[addNode.outputs[0]] = outBuf;

        // Residual input: the other side of the Add (not the 1x1 output)
        let residualInput;
        let hasResidual = 1;
        let resCh = oC;
        if (fb.padIdx >= 0) {
          // Padded residual: the Pad's input is the actual residual source
          const padNode = g[fb.padIdx];
          residualInput = tensors[padNode.inputs[0]];
          const padInShape = shapes[padNode.inputs[0]];
          resCh = padInShape ? padInShape[1] : oC;
          hasResidual = 2;
          shapes[padNode.outputs[0]] = outShape;
          tensors[padNode.outputs[0]] = outBuf;
        } else {
          const addOther = addNode.inputs[0] === pwNode.outputs[0] ? addNode.inputs[1] : addNode.inputs[0];
          residualInput = tensors[addOther];
        }

        // PReLU slope offset
        let actOff = 0;
        if (fb.actType === 1 && fb.actIdx >= 0) {
          const slopeName = g[fb.actIdx].inputs[1];
          actOff = w[slopeName].offset;
        }

        // Build descriptor (21 u32s = 84 bytes)
        const desc = new Uint32Array([
          iC, kSize, stride, padT, padL, padB, padR,
          fb.dwActType || 0, // DW activation (0=none, 2=ReLU6, 3=ReLU)
          w[dwWName].offset, w[dwBName].offset,
          oC, w[pwWName].offset, w[pwBName].offset,
          iH, iW, oH, oW,
          hasResidual, resCh,
          fb.actType, actOff,
        ]);
        const pb = this._getUniformBuf(desc.byteLength);
        device.queue.writeBuffer(pb, 0, desc);

        const pass = enc.beginComputePass();
        pass.setPipeline(this.P.fused_block);
        pass.setBindGroup(0, device.createBindGroup({
          layout: this.P.fused_block.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: pb } },
            { binding: 1, resource: { buffer: tensors[dwNode.inputs[0]] } },
            { binding: 2, resource: { buffer: this.allWeightsBuf } },
            { binding: 3, resource: { buffer: residualInput || this.dummy } },
            { binding: 4, resource: { buffer: outBuf } },
          ],
        }));
        pass.dispatchWorkgroups(Math.ceil(oW / 8), Math.ceil(oH / 8), oC);
        pass.end();
        continue;
      }

      if (op === 'Transpose') {
        // NHWC -> NCHW: handle on CPU before calling run(), or use a transpose shader
        // For now: the caller provides NCHW input directly
        const inShape = shapes[inp[0]];
        const perm = attrs.perm || [0, 3, 1, 2];
        shapes[out] = perm.map(p => inShape[p]);
        tensors[out] = tensors[inp[0]]; // same buffer if already NCHW
        continue;
      }

      if (op === 'Conv') {
        const inName = inp[0];
        const wName = inp[1];
        const bName = inp.length > 2 ? inp[2] : null;
        const inShape = shapes[inName];
        const wShape = w[wName].shape;
        const stride = attrs.strides || [1, 1];
        const pads = attrs.pads || [0, 0, 0, 0];
        const group = attrs.group || 1;

        const [, iC, iH, iW] = inShape;
        const oC = wShape[0];
        const kH = wShape[2], kW = wShape[3];
        const oH = Math.floor((iH + pads[0] + pads[2] - kH) / stride[0]) + 1;
        const oW = Math.floor((iW + pads[1] + pads[3] - kW) / stride[1]) + 1;
        const outShape = [1, oC, oH, oW];
        shapes[out] = outShape;

        // Check for fused Clip (ReLU6) or PRelu following this conv
        let activation = 0; // 0=none, 1=prelu, 2=relu6
        let preluName = null;
        if (i + 1 < g.length && g[i + 1].op === 'Clip' && g[i + 1].inputs[0] === out) {
          activation = 2;
          shapes[g[i + 1].outputs[0]] = outShape;
          tensors[g[i + 1].outputs[0]] = null;
          i++;
        } else if (i + 1 < g.length && g[i + 1].op === 'PRelu' && g[i + 1].inputs[0] === out) {
          activation = 1;
          preluName = g[i + 1].inputs[1];
          shapes[g[i + 1].outputs[0]] = outShape;
          i++;
        } else if (i + 1 < g.length && g[i + 1].op === 'Relu' && g[i + 1].inputs[0] === out) {
          activation = 3;
          shapes[g[i + 1].outputs[0]] = outShape;
          tensors[g[i + 1].outputs[0]] = null;
          i++;
        }

        // Check for fused Add before activation (pattern: Conv -> Add -> Clip/PRelu)
        // Actually in hand landmark: Conv(project) -> Add is the pattern
        // Let's check if NEXT node after possible activation is Add, and the Add's second input is this conv's output
        // For now: don't fuse Add here, handle separately

        const outBuf = getOrAlloc(out, outShape);
        // If we skipped an activation, make sure that output name also points here
        if (activation > 0 && g[i]) tensors[g[i].outputs[0]] = outBuf;

        const params = new Uint32Array([
          1, iC, iH, iW, oC, oH, oW, kH, kW,
          stride[0], stride[1], pads[0], pads[1], group,
          activation, 0, // has_residual = 0 (handled by separate Add)
        ]);
        const pb = this._getUniformBuf(params.byteLength);
        device.queue.writeBuffer(pb, 0, params);

        if (!tensors[inName]) throw new Error(`Conv node ${i}: missing input buffer '${inName}'`);
        if (!this.W[wName]) throw new Error(`Conv node ${i}: missing weight '${wName}'`);

        const pass = enc.beginComputePass();
        pass.setPipeline(this.P.conv2d);
        pass.setBindGroup(0, device.createBindGroup({
          layout: this.P.conv2d.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: pb } },
            { binding: 1, resource: { buffer: tensors[inName] } },
            { binding: 2, resource: { buffer: this.W[wName] } },
            { binding: 3, resource: { buffer: bName ? this.W[bName] : this.dummy } },
            { binding: 4, resource: { buffer: preluName ? this.W[preluName] : this.dummy } },
            { binding: 5, resource: { buffer: this.dummy } }, // no residual
            { binding: 6, resource: { buffer: outBuf } },
          ],
        }));
        pass.dispatchWorkgroups(Math.ceil(oW / 8), Math.ceil(oH / 8), oC);
        pass.end();
        continue;
      }

      if (op === 'Add') {
        const inA = inp[0], inB = inp[1];
        const shape = shapes[inA] || shapes[inB];
        shapes[out] = shape;
        let floats = 1;
        for (const d of shape) floats *= d;
        const outBuf = getOrAlloc(out, shape);

        const params = new Uint32Array([floats, 0]); // mode 0 = plain add
        const pb = this._getUniformBuf(params.byteLength);
        device.queue.writeBuffer(pb, 0, params);
        const pass = enc.beginComputePass();
        pass.setPipeline(this.P.add);
        pass.setBindGroup(0, device.createBindGroup({
          layout: this.P.add.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: pb } },
            { binding: 1, resource: { buffer: tensors[inA] } },
            { binding: 2, resource: { buffer: tensors[inB] } },
            { binding: 3, resource: { buffer: outBuf } },
          ],
        }));
        pass.dispatchWorkgroups(Math.ceil(floats / 256));
        pass.end();
        continue;
      }

      if (op === 'Clip') {
        // Should be fused into conv above. If standalone, just pass through.
        shapes[out] = shapes[inp[0]];
        tensors[out] = tensors[inp[0]];
        continue;
      }

      if (op === 'Relu') {
        // Standalone Relu: dispatch via add shader in mode 1 (relu only).
        const inShape = shapes[inp[0]];
        shapes[out] = inShape;
        let floats = 1;
        if (inShape && Array.isArray(inShape)) for (const d of inShape) floats *= d;
        else floats = tensors[inp[0]].size / 4;
        const outBuf = getOrAlloc(out, inShape || [floats]);

        const params = new Uint32Array([floats, 1]); // mode 1 = relu
        const pb = this._getUniformBuf(params.byteLength);
        device.queue.writeBuffer(pb, 0, params);
        const pass = enc.beginComputePass();
        pass.setPipeline(this.P.add);
        pass.setBindGroup(0, device.createBindGroup({
          layout: this.P.add.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: pb } },
            { binding: 1, resource: { buffer: tensors[inp[0]] } },
            { binding: 2, resource: { buffer: this.dummy } }, // b unused in mode 1
            { binding: 3, resource: { buffer: outBuf } },
          ],
        }));
        pass.dispatchWorkgroups(Math.ceil(floats / 256));
        pass.end();
        continue;
      }

      if (op === 'PRelu') {
        // Standalone PReLU on GPU via add.wgsl mode 3.
        // Slope buffer (per-channel) is in weight buffers; data is NCHW.
        const inShape = shapes[inp[0]];
        shapes[out] = inShape;
        let floats = 1;
        if (inShape && Array.isArray(inShape)) for (const d of inShape) floats *= d;
        else floats = tensors[inp[0]].size / 4;

        const slopeName = inp[1];
        const slopeInfo = w[slopeName];
        const C = slopeInfo.length; // number of channels
        const spatial = floats / C;

        const outBuf = getOrAlloc(out, inShape || [floats]);
        const params = new Uint32Array([floats, 3, C, spatial]); // mode 3 = prelu
        const pb = this._getUniformBuf(params.byteLength);
        device.queue.writeBuffer(pb, 0, params);
        const pass = enc.beginComputePass();
        pass.setPipeline(this.P.add);
        pass.setBindGroup(0, device.createBindGroup({
          layout: this.P.add.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: pb } },
            { binding: 1, resource: { buffer: tensors[inp[0]] } },
            { binding: 2, resource: { buffer: this.W[slopeName] } },
            { binding: 3, resource: { buffer: outBuf } },
          ],
        }));
        pass.dispatchWorkgroups(Math.ceil(floats / 256));
        pass.end();
        continue;
      }

      if (op === 'Sigmoid') {
        // Standalone sigmoid via Gemm shader as identity matmul with sigmoid flag.
        // Tiny (1-63 floats) but keeping on GPU avoids breaking the encoder chain.
        const inShape = shapes[inp[0]];
        shapes[out] = inShape;
        let floats = 1;
        for (const d of inShape) floats *= d;
        const outBuf = getOrAlloc(out, inShape);

        // Use Gemm as identity: M=1, K=floats, N=floats, no weight/bias, sigmoid=1
        // Actually Gemm needs a weight matrix. Simpler: keep on CPU for now (only 1-63 floats).
        device.queue.submit([enc.finish()]);
        const data = await this._readBuffer(tensors[inp[0]], floats);
        for (let j = 0; j < data.length; j++) data[j] = 1 / (1 + Math.exp(-data[j]));
        device.queue.writeBuffer(outBuf, 0, data);
        enc = device.createCommandEncoder();
        continue;
      }

      if (op === 'MaxPool') {
        const inShape = shapes[inp[0]];
        const stride = attrs.strides || [2, 2];
        const [, ch, iH, iW] = inShape;
        const oH = Math.floor(iH / stride[0]);
        const oW = Math.floor(iW / stride[1]);
        const outShape = [1, ch, oH, oW];
        shapes[out] = outShape;
        const outBuf = getOrAlloc(out, outShape);

        const params = new Uint32Array([ch, iH, iW, oH, oW, ch]); // no channel padding
        const pb = this._getUniformBuf(params.byteLength);
        device.queue.writeBuffer(pb, 0, params);
        const pass = enc.beginComputePass();
        pass.setPipeline(this.P.maxpool);
        pass.setBindGroup(0, device.createBindGroup({
          layout: this.P.maxpool.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: pb } },
            { binding: 1, resource: { buffer: tensors[inp[0]] } },
            { binding: 2, resource: { buffer: outBuf } },
          ],
        }));
        pass.dispatchWorkgroups(Math.ceil(oW / 8), Math.ceil(oH / 8), ch);
        pass.end();
        continue;
      }

      if (op === 'GlobalAveragePool') {
        const inShape = shapes[inp[0]];
        const [, ch, iH, iW] = inShape;
        shapes[out] = [1, ch, 1, 1];
        const outBuf = getOrAlloc(out, [ch]);

        const params = new Uint32Array([ch, iH, iW]);
        const pb = this._getUniformBuf(params.byteLength);
        device.queue.writeBuffer(pb, 0, params);
        const pass = enc.beginComputePass();
        pass.setPipeline(this.P.global_avg_pool);
        pass.setBindGroup(0, device.createBindGroup({
          layout: this.P.global_avg_pool.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: pb } },
            { binding: 1, resource: { buffer: tensors[inp[0]] } },
            { binding: 2, resource: { buffer: outBuf } },
          ],
        }));
        pass.dispatchWorkgroups(Math.ceil(ch / 64));
        pass.end();
        continue;
      }

      if (op === 'Squeeze') {
        // Just reinterpret shape, no data movement
        const inShape = shapes[inp[0]];
        shapes[out] = [inShape[1]]; // remove batch + spatial dims
        tensors[out] = tensors[inp[0]];
        continue;
      }

      if (op === 'Gemm') {
        const wName = inp[1];
        const bName = inp.length > 2 ? inp[2] : null;
        const wShape = w[wName].shape;
        const K = wShape[0], N = wShape[1];

        // Check for fused Sigmoid — only fuse if the Sigmoid's input is THIS Gemm's output
        let hasSigmoid = 0;
        if (i + 1 < g.length && g[i + 1].op === 'Sigmoid' && g[i + 1].inputs[0] === out) {
          hasSigmoid = 1;
          shapes[g[i + 1].outputs[0]] = [1, N];
          i++; // skip Sigmoid
        }

        shapes[out] = [1, N];
        const outBuf = getOrAlloc(out, [1, N]);
        if (hasSigmoid && g[i]) tensors[g[i].outputs[0]] = outBuf;

        const params = new Uint32Array([1, K, N, bName ? 1 : 0, hasSigmoid]);
        const pb = this._getUniformBuf(params.byteLength);
        device.queue.writeBuffer(pb, 0, params);
        const pass = enc.beginComputePass();
        pass.setPipeline(this.P.gemm);
        pass.setBindGroup(0, device.createBindGroup({
          layout: this.P.gemm.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: pb } },
            { binding: 1, resource: { buffer: tensors[inp[0]] } },
            { binding: 2, resource: { buffer: this.W[wName] } },
            { binding: 3, resource: { buffer: bName ? this.W[bName] : this.dummy } },
            { binding: 4, resource: { buffer: outBuf } },
          ],
        }));
        pass.dispatchWorkgroups(Math.ceil(N / 64));
        pass.end();
        continue;
      }

      if (op === 'Pad') {
        // Channel padding: append zero channels. Pad constant tensor has 8 ints:
        // [N_before, C_before, H_before, W_before, N_after, C_after, H_after, W_after]
        const inShape = shapes[inp[0]];
        const padName = inp[1];
        // Read pad values from weights
        const padInfo = w[padName];
        const padVals = new Float32Array(allWeights.subarray(padInfo.offset, padInfo.offset + padInfo.length));
        const cAfter = Math.round(padVals[5]); // channels to add after
        const outShape = [inShape[0], inShape[1] + cAfter, inShape[2], inShape[3]];
        shapes[out] = outShape;
        const outBuf = getOrAlloc(out, outShape);

        const params = new Uint32Array([inShape[1], outShape[1], inShape[2], inShape[3]]);
        const pb = this._getUniformBuf(params.byteLength);
        device.queue.writeBuffer(pb, 0, params);
        const pass = enc.beginComputePass();
        pass.setPipeline(this.P.pad_channels);
        pass.setBindGroup(0, device.createBindGroup({
          layout: this.P.pad_channels.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: pb } },
            { binding: 1, resource: { buffer: tensors[inp[0]] } },
            { binding: 2, resource: { buffer: outBuf } },
          ],
        }));
        pass.dispatchWorkgroups(Math.ceil(inShape[3] / 8), Math.ceil(inShape[2] / 8), outShape[1]);
        pass.end();
        continue;
      }

      if (op === 'Reshape') {
        // Reshape is reinterpretation — same data, different shape.
        // Read target shape from weight tensor if available.
        const inShape = shapes[inp[0]];
        if (inp[1] && w[inp[1]]) {
          const shapeInfo = w[inp[1]];
          const shapeVals = Array.from(allWeights.subarray(shapeInfo.offset, shapeInfo.offset + shapeInfo.length)).map(Math.round);
          // Resolve -1
          let totalIn = 1;
          if (inShape && Array.isArray(inShape)) for (const d of inShape) totalIn *= d;
          const known = shapeVals.filter(v => v > 0).reduce((a, b) => a * b, 1);
          const resolved = shapeVals.map(v => v === -1 ? totalIn / known : v);
          shapes[out] = resolved;
        } else {
          shapes[out] = inShape;
        }
        tensors[out] = tensors[inp[0]];
        continue;
      }

      if (op === 'Concat') {
        // GPU transpose + concat: no CPU readback.
        // Each input traces back through Reshape -> Transpose to the NCHW Conv output.
        // We transpose NCHW->NHWC on GPU, then copyBufferToBuffer to assemble.

        // First, figure out each input's NCHW shape by tracing the graph
        const pieces = [];
        let totalFloats = 0;
        for (const inName of inp) {
          if (!tensors[inName]) continue;
          const buf = tensors[inName];
          const nFloats = buf.size / 4;

          // Trace back: inName was produced by Reshape, whose input was Transpose,
          // whose input was the Conv. The Conv's output shape is the NCHW shape.
          // Find the Reshape node that outputs inName
          let nchwShape = null;
          for (const n of g) {
            if (n.op === 'Reshape' && n.outputs[0] === inName) {
              // Reshape's input is the Transpose output
              for (const n2 of g) {
                if (n2.op === 'Transpose' && n2.outputs[0] === n.inputs[0]) {
                  // Transpose's input is the Conv output -- shapes has its NCHW shape
                  nchwShape = shapes[n2.inputs[0]];
                  break;
                }
              }
              break;
            }
          }

          if (nchwShape && nchwShape.length === 4) {
            const [, C, H, Wd] = nchwShape;
            // Transpose NCHW->NHWC on GPU
            const nhwcBuf = device.createBuffer({ size: nFloats * 4, usage: BF });
            const tParams = new Uint32Array([C, H, Wd, 0]);
            const tpb = this._getUniformBuf(tParams.byteLength);
            device.queue.writeBuffer(tpb, 0, tParams);
            const pass = enc.beginComputePass();
            pass.setPipeline(this.P.transpose_nhwc);
            pass.setBindGroup(0, device.createBindGroup({
              layout: this.P.transpose_nhwc.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: tpb } },
                { binding: 1, resource: { buffer: buf } },
                { binding: 2, resource: { buffer: nhwcBuf } },
              ],
            }));
            pass.dispatchWorkgroups(Math.ceil(nFloats / 256));
            pass.end();
            pieces.push({ buf: nhwcBuf, floats: nFloats });
          } else {
            pieces.push({ buf, floats: nFloats });
          }
          totalFloats += nFloats;
        }

        // Allocate output and copy pieces into it
        const outBuf = getOrAlloc(out, [1, totalFloats]);
        let dstOffset = 0;
        for (const piece of pieces) {
          enc.copyBufferToBuffer(piece.buf, 0, outBuf, dstOffset * 4, piece.floats * 4);
          dstOffset += piece.floats;
        }
        shapes[out] = [1, totalFloats];
        continue;
      }

      if (op === 'Resize') {
        const inShape = shapes[inp[0]];
        const [, ch, iH, iW] = inShape;
        // Assume 2x upsample
        const outShape = [1, ch, iH * 2, iW * 2];
        shapes[out] = outShape;
        const outBuf = getOrAlloc(out, outShape);

        const params = new Uint32Array([ch, iH, iW, iH * 2, iW * 2]);
        const pb = this._getUniformBuf(params.byteLength);
        device.queue.writeBuffer(pb, 0, params);
        const pass = enc.beginComputePass();
        pass.setPipeline(this.P.resize);
        pass.setBindGroup(0, device.createBindGroup({
          layout: this.P.resize.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: pb } },
            { binding: 1, resource: { buffer: tensors[inp[0]] } },
            { binding: 2, resource: { buffer: outBuf } },
          ],
        }));
        pass.dispatchWorkgroups(Math.ceil(iW * 2 / 8), Math.ceil(iH * 2 / 8), ch);
        pass.end();
        continue;
      }

      if (op === 'Neg' || op === 'Mul') {
        // Part of decomposed PReLU (face landmark). Pass through for now.
        // TODO: fuse Relu+Neg+Mul+Add back into PReLU during graph preprocessing.
        shapes[out] = shapes[inp[0]];
        tensors[out] = tensors[inp[0]];
        console.warn(`Passthrough op: ${op} at node ${i} (decomposed PReLU?)`);
        continue;
      }

      console.warn(`Unknown op: ${op} at node ${i}`);
    }

    // Submit final commands
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Debug: dump stats for key tensors
    if (debug) {
      const debugNodes = [2, 5, 6, 9, 10, 11, 58, 59, 84, 88, 91];
      for (const idx of debugNodes) {
        if (idx >= g.length) continue;
        const name = g[idx].outputs[0];
        const buf = tensors[name];
        if (!buf) { console.log(`  node ${idx} (${g[idx].op}): no buffer for '${name}'`); continue; }
        const n = buf.size / 4;
        const data = await this._readBuffer(buf, n);
        let mn = Infinity, mx = -Infinity;
        for (let j = 0; j < data.length; j++) { mn = Math.min(mn, data[j]); mx = Math.max(mx, data[j]); }
        console.log(`  node ${idx} ${g[idx].op}: '${name.substring(0,25)}' [${n}] min=${mn.toFixed(4)} max=${mx.toFixed(4)} first=[${data.slice(0,3).map(v=>v.toFixed(4)).join(',')}]`);
      }
    }

    // Read back outputs
    const outputs = {};
    for (const outDef of graph.outputs) {
      const name = outDef.name;
      let floats = 1;
      for (const d of outDef.shape) floats *= d;
      if (tensors[name]) {
        outputs[name] = await this._readBuffer(tensors[name], floats);
      }
    }
    return outputs;
  }

  async _readBuffer(gpuBuf, floats) {
    const rb = this.device.createBuffer({ size: floats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(gpuBuf, 0, rb, 0, floats * 4);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(rb.getMappedRange()).slice();
    rb.unmap();
    rb.destroy();
    return data;
  }
}
