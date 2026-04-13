/**
 * WebGPU Vision inference engine.
 * Runs neural network models using pure WebGPU compute shaders.
 * Zero WASM. Zero ONNX Runtime. Zero SharedArrayBuffer.
 *
 * Usage:
 *   const engine = new InferenceEngine();
 *   await engine.init();
 *   await engine.loadModel('palm_detection_lite');
 *   const output = await engine.run(inputFloat32Array);
 */

export class InferenceEngine {
  constructor() {
    this.device = null;
    this.pipelines = {};  // cached compute pipelines by shader name
    this.shaderCode = {}; // loaded WGSL source
  }

  async init() {
    if (!navigator.gpu) throw new Error('WebGPU not available');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    this.device = await adapter.requestDevice();
    console.log('[engine] WebGPU device ready');

    // Load shader sources
    const shaderNames = ['conv2d', 'maxpool', 'resize'];
    for (const name of shaderNames) {
      const url = new URL(`./${name}.wgsl`, import.meta.url);
      const resp = await fetch(url);
      this.shaderCode[name] = await resp.text();
    }
    console.log(`[engine] ${shaderNames.length} shaders loaded`);
  }

  /**
   * Load a model's graph definition and weights.
   * @param {string} name - Model name (matches files in models/ directory)
   * @param {string} basePath - Path to models directory
   */
  async loadModel(name, basePath = '../models') {
    const graphResp = await fetch(`${basePath}/${name}.json`);
    this.graph = await graphResp.json();

    // Load the flat weight buffer
    const binResp = await fetch(`${basePath}/${name}.bin`);
    const weightData = new Float32Array(await binResp.arrayBuffer());

    // Upload all weights to one big GPU storage buffer
    this.weightBuffer = this.device.createBuffer({
      size: weightData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: `${name}_weights`,
    });
    this.device.queue.writeBuffer(this.weightBuffer, 0, weightData);

    console.log(`[engine] Model "${name}" loaded: ${this.graph.graph.length} nodes, ${(weightData.byteLength / 1024 / 1024).toFixed(2)} MB weights`);
  }

  /**
   * Create a GPU buffer for intermediate activations.
   */
  createBuffer(floats, label = '') {
    const buf = this.device.createBuffer({
      size: floats * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label,
    });
    return buf;
  }

  /**
   * Create a uniform buffer from a struct.
   */
  createUniformBuffer(data, label = '') {
    const buf = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label,
    });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  /**
   * Get or create a compute pipeline for a shader.
   */
  getPipeline(shaderName) {
    if (!this.pipelines[shaderName]) {
      const module = this.device.createShaderModule({
        code: this.shaderCode[shaderName],
        label: shaderName,
      });
      this.pipelines[shaderName] = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
        label: shaderName,
      });
    }
    return this.pipelines[shaderName];
  }

  /**
   * Dispatch a conv2d+prelu shader.
   */
  dispatchConv(encoder, {
    input, output, residual,
    weightOffset, weightLength,
    biasOffset, biasLength,
    preluOffset, preluLength,
    inC, inH, inW, outC, outH, outW,
    kernH, kernW, strideH, strideW,
    padTop, padLeft, group,
    hasPReLU, hasResidual,
  }) {
    const pipeline = this.getPipeline('conv2d');

    // Build the params uniform
    const params = new Uint32Array([
      1, inC, inH, inW, outC, outH, outW,
      kernH, kernW, strideH, strideW,
      padTop, padLeft, group,
      hasPReLU ? 1 : 0, hasResidual ? 1 : 0,
    ]);
    const paramBuf = this.createUniformBuffer(params, 'conv_params');

    // Create views into the weight buffer for this layer's weights
    const weightView = this.createWeightView(weightOffset, weightLength);
    const biasView = this.createWeightView(biasOffset, biasLength);
    const preluView = hasPReLU
      ? this.createWeightView(preluOffset, preluLength)
      : this.createWeightView(0, 1); // dummy
    const residualBuf = hasResidual ? residual : this.createBuffer(1, 'dummy_residual');

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: weightView } },
        { binding: 3, resource: { buffer: biasView } },
        { binding: 4, resource: { buffer: preluView } },
        { binding: 5, resource: { buffer: residualBuf } },
        { binding: 6, resource: { buffer: output } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(outW / 8),
      Math.ceil(outH / 8),
      outC
    );
    pass.end();
  }

  /**
   * Create a buffer that is a copy of a slice of the weight buffer.
   * (WebGPU doesn't support buffer views with arbitrary offsets on storage bindings,
   * so we copy the slice we need. This is a one-time cost per layer at model load time.)
   */
  createWeightView(floatOffset, floatLength) {
    // Cache these so we don't re-create every frame
    const key = `${floatOffset}_${floatLength}`;
    if (!this._weightViews) this._weightViews = {};
    if (this._weightViews[key]) return this._weightViews[key];

    const buf = this.device.createBuffer({
      size: Math.max(floatLength * 4, 4), // minimum 4 bytes
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: `weight_${floatOffset}`,
    });

    // Copy from the main weight buffer
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      this.weightBuffer, floatOffset * 4,
      buf, 0,
      floatLength * 4
    );
    this.device.queue.submit([encoder.finish()]);

    this._weightViews[key] = buf;
    return buf;
  }
}
