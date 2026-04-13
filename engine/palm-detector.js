/**
 * Palm detection model runner — pure WebGPU, zero ONNX Runtime.
 *
 * Executes the BlazePalm model graph using fused compute shaders.
 * Input: [1, 192, 192, 3] float32 (NHWC, values in [0,1])
 * Output: { regressors: Float32Array[2016*18], classifiers: Float32Array[2016*1] }
 *
 * Usage:
 *   const detector = new PalmDetector();
 *   await detector.init(device);
 *   const result = await detector.detect(inputBuffer);
 */

export class PalmDetector {
  constructor() {
    this.device = null;
    this.graph = null;
    this.weightViews = {};  // weight name -> GPUBuffer
    this.activations = {};  // activation name -> { buffer, shape }
    this.pipelines = {};
  }

  async init(device, basePath = '../models') {
    this.device = device;

    // Load graph definition
    const graphResp = await fetch(`${basePath}/palm_detection_lite.json`);
    this.graph = await graphResp.json();

    // Load flat weight buffer
    const binResp = await fetch(`${basePath}/palm_detection_lite.bin`);
    const weightData = new Float32Array(await binResp.arrayBuffer());

    // Pre-slice weights into individual GPU buffers.
    // Each weight tensor gets its own buffer so we can bind it directly.
    for (const [name, info] of Object.entries(this.graph.weights)) {
      if (info.length === 0) continue;
      const slice = weightData.subarray(info.offset, info.offset + info.length);
      const buf = device.createBuffer({
        size: Math.max(slice.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: name,
      });
      device.queue.writeBuffer(buf, 0, slice);
      this.weightViews[name] = buf;
    }

    // Load shaders
    const convCode = await (await fetch(new URL('./conv2d.wgsl', import.meta.url))).text();
    const poolCode = await (await fetch(new URL('./maxpool.wgsl', import.meta.url))).text();
    const resizeCode = await (await fetch(new URL('./resize.wgsl', import.meta.url))).text();

    this.shaders = {
      conv2d: device.createShaderModule({ code: convCode, label: 'conv2d' }),
      maxpool: device.createShaderModule({ code: poolCode, label: 'maxpool' }),
      resize: device.createShaderModule({ code: resizeCode, label: 'resize' }),
    };

    // Pre-build pipelines
    this.pipelines.conv2d = device.createComputePipeline({
      layout: 'auto', compute: { module: this.shaders.conv2d, entryPoint: 'main' },
    });
    this.pipelines.maxpool = device.createComputePipeline({
      layout: 'auto', compute: { module: this.shaders.maxpool, entryPoint: 'main' },
    });
    this.pipelines.resize = device.createComputePipeline({
      layout: 'auto', compute: { module: this.shaders.resize, entryPoint: 'main' },
    });

    // Pre-allocate activation buffers for all intermediate tensor shapes.
    // We double-buffer (A/B) so conv can read from one and write to another.
    this._allocateActivationBuffers();

    console.log(`[palm-detector] ready: ${this.graph.graph.length} nodes, ${Object.keys(this.weightViews).length} weight tensors`);
  }

  _allocateActivationBuffers() {
    // All unique activation sizes needed by the model.
    // We over-allocate two buffers at each size for ping-pong.
    const sizes = [
      { name: '3_192', shape: [1, 3, 192, 192], floats: 110592 },
      { name: '32_96', shape: [1, 32, 96, 96], floats: 294912 },
      { name: '32_48', shape: [1, 32, 48, 48], floats: 73728 },
      { name: '64_48', shape: [1, 64, 48, 48], floats: 147456 },
      { name: '64_24', shape: [1, 64, 24, 24], floats: 36864 },
      { name: '128_24', shape: [1, 128, 24, 24], floats: 73728 },
      { name: '128_12', shape: [1, 128, 12, 12], floats: 18432 },
      { name: '256_12', shape: [1, 256, 12, 12], floats: 36864 },
      { name: '256_6', shape: [1, 256, 6, 6], floats: 9216 },
      { name: '256_24', shape: [1, 256, 24, 24], floats: 147456 },
      // Output head buffers
      { name: '108_12', shape: [1, 108, 12, 12], floats: 15552 },
      { name: '6_12', shape: [1, 6, 12, 12], floats: 864 },
      { name: '36_24', shape: [1, 36, 24, 24], floats: 20736 },
      { name: '2_24', shape: [1, 2, 24, 24], floats: 1152 },
      // Final concat outputs
      { name: 'regressors', shape: [1, 2016, 18], floats: 2016 * 18 },
      { name: 'classifiers', shape: [1, 2016, 1], floats: 2016 },
    ];

    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.buffers = {};
    for (const s of sizes) {
      // Allocate A and B for ping-pong
      this.buffers[s.name + '_A'] = this.device.createBuffer({
        size: s.floats * 4, usage, label: s.name + '_A',
      });
      this.buffers[s.name + '_B'] = this.device.createBuffer({
        size: s.floats * 4, usage, label: s.name + '_B',
      });
    }
    // Input buffer (NHWC, will be transposed to NCHW)
    this.buffers.input = this.device.createBuffer({
      size: 192 * 192 * 3 * 4, usage, label: 'input_nhwc',
    });
  }

  /**
   * Create a dummy buffer for unused bindings (residual, prelu when not needed).
   */
  _getDummy() {
    if (!this._dummy) {
      this._dummy = this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE,
        label: 'dummy',
      });
    }
    return this._dummy;
  }

  /**
   * Dispatch a single conv2d operation.
   */
  _dispatchConv(encoder, {
    input, output, residual,
    weightName, biasName, preluName,
    inC, inH, inW, outC, outH, outW,
    kernH, kernW, strideH, strideW,
    padTop, padLeft, group,
  }) {
    const hasPReLU = !!preluName;
    const hasResidual = !!residual;

    const params = new Uint32Array([
      1, inC, inH, inW, outC, outH, outW,
      kernH, kernW, strideH, strideW,
      padTop, padLeft, group,
      hasPReLU ? 1 : 0, hasResidual ? 1 : 0,
    ]);
    const paramBuf = this.device.createBuffer({
      size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(paramBuf, 0, params);

    const pipeline = this.pipelines.conv2d;
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: this.weightViews[weightName] } },
        { binding: 3, resource: { buffer: this.weightViews[biasName] } },
        { binding: 4, resource: { buffer: hasPReLU ? this.weightViews[preluName] : this._getDummy() } },
        { binding: 5, resource: { buffer: hasResidual ? residual : this._getDummy() } },
        { binding: 6, resource: { buffer: output } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outW / 8), Math.ceil(outH / 8), outC);
    pass.end();
  }

  /**
   * Dispatch maxpool+pad.
   */
  _dispatchMaxPool(encoder, { input, output, channels, inH, inW, outChannels }) {
    const outH = inH / 2, outW = inW / 2;
    const params = new Uint32Array([channels, inH, inW, outH, outW, outChannels]);
    const paramBuf = this.device.createBuffer({
      size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(paramBuf, 0, params);

    const pipeline = this.pipelines.maxpool;
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: output } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outW / 8), Math.ceil(outH / 8), outChannels);
    pass.end();
  }

  /**
   * Dispatch bilinear 2x resize.
   */
  _dispatchResize(encoder, { input, output, channels, inH, inW }) {
    const outH = inH * 2, outW = inW * 2;
    const params = new Uint32Array([channels, inH, inW, outH, outW]);
    const paramBuf = this.device.createBuffer({
      size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(paramBuf, 0, params);

    const pipeline = this.pipelines.resize;
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: output } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outW / 8), Math.ceil(outH / 8), channels);
    pass.end();
  }

  /**
   * Run inference on a 192x192x3 input image.
   * @param {Float32Array} inputData - NHWC [1,192,192,3] normalized to [0,1]
   * @returns {{ regressors: Float32Array, classifiers: Float32Array }}
   */
  async detect(inputData) {
    // Upload input
    this.device.queue.writeBuffer(this.buffers.input, 0, inputData);

    const encoder = this.device.createCommandEncoder();

    // TODO: Execute the full graph here by walking through nodes
    // and dispatching the appropriate shader for each operation.
    // For now this is a skeleton — the wiring of each specific node
    // to its weight tensors and activation buffers is the next step.

    this.device.queue.submit([encoder.finish()]);

    // Read back results
    // (In the future this stays on GPU — Phase 4 endgame)
    const regressorReadBuf = this.device.createBuffer({
      size: 2016 * 18 * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const classifierReadBuf = this.device.createBuffer({
      size: 2016 * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const readEncoder = this.device.createCommandEncoder();
    readEncoder.copyBufferToBuffer(this.buffers.regressors_A, 0, regressorReadBuf, 0, 2016 * 18 * 4);
    readEncoder.copyBufferToBuffer(this.buffers.classifiers_A, 0, classifierReadBuf, 0, 2016 * 4);
    this.device.queue.submit([readEncoder.finish()]);

    await regressorReadBuf.mapAsync(GPUMapMode.READ);
    await classifierReadBuf.mapAsync(GPUMapMode.READ);

    const regressors = new Float32Array(regressorReadBuf.getMappedRange()).slice();
    const classifiers = new Float32Array(classifierReadBuf.getMappedRange()).slice();

    regressorReadBuf.unmap();
    classifierReadBuf.unmap();
    regressorReadBuf.destroy();
    classifierReadBuf.destroy();

    return { regressors, classifiers };
  }
}
