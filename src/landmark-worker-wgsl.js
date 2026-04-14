// Hand landmark worker: WGSL engine (replaces ORT).
// Receives ImageBitmap + rotated rect, does GPU affine warp + WGSL inference.

import { ModelRunner } from '../engine/model-runner.js';

const S = 224; // landmark model input size
const MODEL_JSON_URL = '../models/hand_landmark_full.json';
const MODEL_BIN_URL = '../models/hand_landmark_full.bin';

let runner = null;
let device = null;
let inputBuf = null; // also the warp output buffer

// GPU warp state
let warpPipeline = null;
let uniformBuffer = null;
let gpuSampler = null;

// Output field mapping (discovered during init from model outputs)
let outputNames = {};

const WGSL_WARP = `
struct Uniforms {
  a: f32, b: f32, c: f32, _pad0: f32,
  d: f32, e: f32, f_: f32, _pad1: f32,
  src_w: f32, src_h: f32,
}

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> u: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x;
  let y = gid.y;
  if (x >= ${S}u || y >= ${S}u) { return; }

  let fx = f32(x) + 0.5;
  let fy = f32(y) + 0.5;

  let sx = u.a * fx + u.b * fy + u.c;
  let sy = u.d * fx + u.e * fy + u.f_;

  let uv = vec2f(sx / u.src_w, sy / u.src_h);
  let pixel = textureSampleLevel(srcTexture, srcSampler, uv, 0.0);

  // Output NCHW: channel-first
  let spatial = ${S}u * ${S}u;
  let idx = y * ${S}u + x;
  output[idx]               = pixel.r;
  output[spatial + idx]     = pixel.g;
  output[2u * spatial + idx] = pixel.b;
}
`;

function rotatedRectPoints(cx, cy, w, h, rotation) {
  const b = Math.cos(rotation) * 0.5;
  const a = Math.sin(rotation) * 0.5;
  const p0x = cx - a * h - b * w, p0y = cy + b * h - a * w;
  const p1x = cx + a * h - b * w, p1y = cy - b * h - a * w;
  return [[p0x, p0y], [p1x, p1y], [2 * cx - p0x, 2 * cy - p0y], [2 * cx - p1x, 2 * cy - p1y]];
}

function computeAffineParams(rect) {
  const pts = rotatedRectPoints(rect.cx, rect.cy, rect.w, rect.h, rect.angle);
  const [p1x, p1y] = pts[1];
  const [p2x, p2y] = pts[2];
  const [p3x, p3y] = pts[3];

  const dx1 = p2x - p1x, dy1 = p2y - p1y;
  const dx2 = p3x - p1x, dy2 = p3y - p1y;
  const det = dx1 * dy2 - dx2 * dy1;
  if (Math.abs(det) < 1e-6) return null;

  const invDet = 1 / det;
  const fa = S * (dy2 - dy1) * invDet;
  const fb = S * (dx1 - dx2) * invDet;
  const fd = S * (-dy1) * invDet;
  const fe = S * dx1 * invDet;
  const fc = -fa * p1x - fb * p1y;
  const ff = -fd * p1x - fe * p1y;

  const detM = fa * fe - fb * fd;
  return {
    a: fe / detM, b: -fb / detM,
    c: -(fe * fc - fb * ff) / detM,
    d: -fd / detM, e: fa / detM,
    f: -(-fd * fc + fa * ff) / detM,
  };
}

async function initGPU() {
  const adapter = await navigator.gpu.requestAdapter();
  device = await adapter.requestDevice();

  // Warp pipeline
  warpPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: WGSL_WARP }), entryPoint: 'main' },
  });

  const BF = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  inputBuf = device.createBuffer({ size: S * S * 3 * 4, usage: BF });
  uniformBuffer = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  gpuSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  // Load model
  const graph = await (await fetch(MODEL_JSON_URL)).json();
  const allWeights = new Float32Array(await (await fetch(MODEL_BIN_URL)).arrayBuffer());

  const W = {};
  for (const [name, info] of Object.entries(graph.weights)) {
    if (info.length === 0) continue;
    const buf = device.createBuffer({ size: Math.max(info.length * 4, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, allWeights.subarray(info.offset, info.offset + info.length));
    W[name] = buf;
  }

  const allWeightsBuf = device.createBuffer({ size: allWeights.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(allWeightsBuf, 0, allWeights);

  const mkP = async (name) => {
    const code = await (await fetch(`../engine/${name}.wgsl`)).text();
    return device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  };
  const P = {
    conv2d: await mkP('conv2d'), maxpool: await mkP('maxpool'), resize: await mkP('resize'),
    gemm: await mkP('gemm'), global_avg_pool: await mkP('global_avg_pool'),
    add: await mkP('add'), fused_block: await mkP('fused_block'),
    transpose_nhwc: await mkP('transpose_nhwc'),
  };

  runner = new ModelRunner(device, P, W, allWeightsBuf);
  const outputs = await runner.compile(graph, inputBuf, allWeights);

  // Discover output names by size
  for (const [name, data] of Object.entries(outputs)) {
    if (data.length === 63 && !outputNames.landmarks) outputNames.landmarks = name;
    else if (data.length === 63) outputNames.worldLandmarks = name;
    else if (data.length === 1 && !outputNames.handFlag) outputNames.handFlag = name;
    else if (data.length === 1) outputNames.handedness = name;
  }
}

function dispatchWarp(bitmap, inv) {
  const srcTexture = device.createTexture({
    size: [bitmap.width, bitmap.height], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: srcTexture }, [bitmap.width, bitmap.height]);
  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
    inv.a, inv.b, inv.c, 0,
    inv.d, inv.e, inv.f, 0,
    bitmap.width, bitmap.height, 0, 0,
  ]));

  const bindGroup = device.createBindGroup({
    layout: warpPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: srcTexture.createView() },
      { binding: 1, resource: gpuSampler },
      { binding: 2, resource: { buffer: inputBuf } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(warpPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(S / 16), Math.ceil(S / 16));
  pass.end();
  device.queue.submit([enc.finish()]);
  srcTexture.destroy();
}

self.onmessage = async (e) => {
  const { type } = e.data;

  if (type === 'init') {
    try {
      await initGPU();
      console.log('[landmark-worker-wgsl] ready (compiled WGSL engine)');
      self.postMessage({ type: 'ready', gpuDirect: true });
    } catch (err) {
      console.error('[landmark-worker-wgsl] init error:', err);
      self.postMessage({ type: 'error', message: err.message });
    }
  }

  if (type === 'infer') {
    try {
      const { bitmap, rect, vw, vh } = e.data;

      const inv = computeAffineParams(rect);
      if (!inv) {
        bitmap.close();
        self.postMessage({ type: 'result', handFlag: 0, landmarks: null });
        return;
      }

      // GPU affine warp writes NCHW directly into the compiled input buffer
      dispatchWarp(bitmap, inv);
      bitmap.close();

      // Run compiled model
      const outputs = await runner.runCompiled();

      const rawLandmarks = outputNames.landmarks ? outputs[outputNames.landmarks] : null;
      const handFlag = outputNames.handFlag ? outputs[outputNames.handFlag][0] : 0;
      const handednessRaw = outputNames.handedness ? outputs[outputNames.handedness][0] : 0.5;
      const handedness = handednessRaw > 0.5 ? 'Right' : 'Left';

      // Project landmarks back through inverse affine
      let projectedLandmarks = null;
      if (rawLandmarks && rawLandmarks.length === 63) {
        projectedLandmarks = new Float32Array(63);
        for (let i = 0; i < 21; i++) {
          const ox = rawLandmarks[i * 3];
          const oy = rawLandmarks[i * 3 + 1];
          const oz = rawLandmarks[i * 3 + 2];
          projectedLandmarks[i * 3]     = (inv.a * ox + inv.b * oy + inv.c) / vw;
          projectedLandmarks[i * 3 + 1] = (inv.d * ox + inv.e * oy + inv.f) / vh;
          projectedLandmarks[i * 3 + 2] = oz / S;
        }
      }

      self.postMessage({
        type: 'result',
        handFlag,
        handedness,
        landmarks: projectedLandmarks ? projectedLandmarks.buffer : null,
      }, projectedLandmarks ? [projectedLandmarks.buffer] : []);
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }
};
