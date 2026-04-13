/**
 * Generic WGSL model runner — executes any model from its graph JSON + weights.
 * Automatically allocates buffers, wires dispatches, handles ping-pong.
 *
 * Usage:
 *   const runner = new ModelRunner(device, pipelines, weightBufs);
 *   const outputs = await runner.run(graphJson, inputBuffer);
 */

export class ModelRunner {
  constructor(device, pipelines, weightBufs) {
    this.device = device;
    this.P = pipelines;     // { conv2d, maxpool, resize, gemm, global_avg_pool, add }
    this.W = weightBufs;    // weight name -> GPUBuffer
    this.dummy = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
  }

  /**
   * Run a model end-to-end.
   * @param {Object} graph - parsed graph JSON
   * @param {GPUBuffer} inputBuf - input in NHWC format
   * @returns {Object} output name -> Float32Array
   */
  async run(graph, inputBuf) {
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

    let enc = device.createCommandEncoder();

    for (let i = 0; i < g.length; i++) {
      const node = g[i];
      const op = node.op;
      const attrs = node.attrs || {};
      const inp = node.inputs;
      const out = node.outputs[0];

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
        if (i + 1 < g.length && g[i + 1].op === 'Clip') {
          activation = 2;
          shapes[g[i + 1].outputs[0]] = outShape;
          tensors[g[i + 1].outputs[0]] = null; // will point to same buf
          i++; // skip Clip
        } else if (i + 1 < g.length && g[i + 1].op === 'PRelu') {
          activation = 1;
          preluName = g[i + 1].inputs[1];
          shapes[g[i + 1].outputs[0]] = outShape;
          i++; // skip PRelu
        }

        // Check for fused Add before activation (pattern: Conv -> Add -> Clip/PRelu)
        // Actually in hand landmark: Conv(project) -> Add is the pattern
        // Let's check if NEXT node after possible activation is Add, and the Add's second input is this conv's output
        // For now: don't fuse Add here, handle separately

        const outBuf = getOrAlloc(out, outShape);
        // If we skipped an activation, make sure that output name also points here
        if (activation === 2 && g[i]) tensors[g[i].outputs[0]] = outBuf;
        if (activation === 1 && g[i]) tensors[g[i].outputs[0]] = outBuf;

        const params = new Uint32Array([
          1, iC, iH, iW, oC, oH, oW, kH, kW,
          stride[0], stride[1], pads[0], pads[1], group,
          activation, 0, // has_residual = 0 (handled by separate Add)
        ]);
        const pb = device.createBuffer({ size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(pb, 0, params);

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

        const params = new Uint32Array([floats]);
        const pb = device.createBuffer({ size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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

      if (op === 'PRelu') {
        // Should be fused. Pass through.
        shapes[out] = shapes[inp[0]];
        tensors[out] = tensors[inp[0]];
        continue;
      }

      if (op === 'Sigmoid') {
        // Standalone sigmoid: dispatch Gemm as identity with sigmoid.
        // M=1, K=N, N=N, no weights needed — but Gemm requires weight buffer.
        // Simpler: just read, apply on CPU, write back. Only 1-63 floats.
        const inShape = shapes[inp[0]];
        shapes[out] = inShape;
        let floats = 1;
        for (const d of inShape) floats *= d;

        // Submit pending work, apply sigmoid on CPU (tiny: 1-63 floats)
        device.queue.submit([enc.finish()]);
        const data = await this._readBuffer(tensors[inp[0]], floats);
        for (let j = 0; j < data.length; j++) data[j] = 1 / (1 + Math.exp(-data[j]));
        const outBuf = getOrAlloc(out, inShape);
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
        const pb = device.createBuffer({ size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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
        const pb = device.createBuffer({ size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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
        const pb = device.createBuffer({ size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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

      if (op === 'Reshape' || op === 'Concat' || op === 'Resize') {
        // TODO: handle these for models that need them (palm detector)
        // For hand landmark, these don't appear
        console.warn(`Unhandled op: ${op} at node ${i}`);
        shapes[out] = shapes[inp[0]];
        tensors[out] = tensors[inp[0]];
        continue;
      }

      console.warn(`Unknown op: ${op} at node ${i}`);
    }

    // Submit final commands
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

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
