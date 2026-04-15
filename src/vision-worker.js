// Unified vision worker: ONE GPU device, ALL models compiled.
// Replaces 4 separate workers (palm, 2x landmark, face det, face lm).

import { ModelRunner } from '../engine/model-runner.js';
import { generateAnchors, decodeDetections } from './anchors.js';
import { weightedNMS } from './nms.js';
import { generateFaceAnchors, decodeFaceDetections } from './face-anchors.js';
import { weightedNMS as faceNMS } from './face-nms.js';

const PALM_SIZE = 192;
const HAND_SIZE = 224;
const FACE_DET_SIZE = 128;
const FACE_LM_SIZE = 256;
const NUM_FACE_LM = 478;

let device = null;
let palmRunner = null, handRunners = [null, null], faceDetRunner = null, faceLmRunner = null;
let palmAnchors = null, faceAnchors = null;
let handOutputNames = {}, faceLmOutputNames = {};

// Shared GPU resources
let sampler = null;

// Per-model letterbox/warp pipelines and buffers
const models = {};

function makeWarpShader(size, nchw = true) {
  const outCode = nchw
    ? `let spatial = ${size}u * ${size}u;
  let idx = y * ${size}u + x;
  output[idx]               = pixel.r;
  output[spatial + idx]     = pixel.g;
  output[2u * spatial + idx] = pixel.b;`
    : `let idx = (y * ${size}u + x) * 3u;
  output[idx]      = pixel.r;
  output[idx + 1u] = pixel.g;
  output[idx + 2u] = pixel.b;`;

  return `
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
  if (x >= ${size}u || y >= ${size}u) { return; }

  let fx = f32(x) + 0.5;
  let fy = f32(y) + 0.5;
  let sx = u.a * fx + u.b * fy + u.c;
  let sy = u.d * fx + u.e * fy + u.f_;

  var pixel = vec4f(0.0);
  if (sx >= 0.0 && sx < u.src_w && sy >= 0.0 && sy < u.src_h) {
    let uv = vec2f(sx / u.src_w, sy / u.src_h);
    pixel = textureSampleLevel(srcTexture, srcSampler, uv, 0.0);
  }

  ${outCode}
}
`;
}

function rotatedRectPoints(cx, cy, w, h, rotation) {
  const b = Math.cos(rotation) * 0.5;
  const a = Math.sin(rotation) * 0.5;
  const p0x = cx - a * h - b * w, p0y = cy + b * h - a * w;
  const p1x = cx + a * h - b * w, p1y = cy - b * h - a * w;
  return [[p0x, p0y], [p1x, p1y], [2 * cx - p0x, 2 * cy - p0y], [2 * cx - p1x, 2 * cy - p1y]];
}

function computeAffineParams(rect, S) {
  const pts = rotatedRectPoints(rect.cx, rect.cy, rect.w, rect.h, rect.angle);
  const [p1x, p1y] = pts[1];
  const [p2x, p2y] = pts[2];
  const [p3x, p3y] = pts[3];
  const dx1 = p2x - p1x, dy1 = p2y - p1y;
  const dx2 = p3x - p1x, dy2 = p3y - p1y;
  const det = dx1 * dy2 - dx2 * dy1;
  if (Math.abs(det) < 1e-6) return null;
  const invDet = 1 / det;
  const fa = S * (dy2 - dy1) * invDet, fb = S * (dx1 - dx2) * invDet;
  const fd = S * (-dy1) * invDet, fe = S * dx1 * invDet;
  const fc = -fa * p1x - fb * p1y, ff = -fd * p1x - fe * p1y;
  const detM = fa * fe - fb * fd;
  return {
    a: fe / detM, b: -fb / detM, c: -(fe * fc - fb * ff) / detM,
    d: -fd / detM, e: fa / detM, f: -(-fd * fc + fa * ff) / detM,
  };
}

function setupModel(name, size) {
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: makeWarpShader(size) }), entryPoint: 'main' },
  });
  const BF = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const inputBuf = device.createBuffer({ size: size * size * 3 * 4, usage: BF });
  const uniformBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  models[name] = { pipeline, inputBuf, uniformBuf, size };
}

// Encode a warp dispatch into an external encoder using a shared texture
function encodeWarp(enc, name, srcTexture, srcW, srcH, affine) {
  const m = models[name];
  device.queue.writeBuffer(m.uniformBuf, 0, new Float32Array([
    affine.a, affine.b, affine.c, 0,
    affine.d, affine.e, affine.f, 0,
    srcW, srcH, 0, 0,
  ]));
  const bindGroup = device.createBindGroup({
    layout: m.pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: srcTexture.createView() },
      { binding: 1, resource: sampler },
      { binding: 2, resource: { buffer: m.inputBuf } },
      { binding: 3, resource: { buffer: m.uniformBuf } },
    ],
  });
  const pass = enc.beginComputePass();
  pass.setPipeline(m.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(m.size / 16), Math.ceil(m.size / 16));
  pass.end();
}

// Legacy: standalone warp with own texture + submit (for non-batched calls)
function dispatchWarp(name, bitmap, affine) {
  const srcTexture = device.createTexture({
    size: [bitmap.width, bitmap.height], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: srcTexture }, [bitmap.width, bitmap.height]);
  const enc = device.createCommandEncoder();
  encodeWarp(enc, name, srcTexture, bitmap.width, bitmap.height, affine);
  device.queue.submit([enc.finish()]);
  srcTexture.destroy();
}

function letterboxAffine(size, srcW, srcH) {
  const scale = size / Math.max(srcW, srcH);
  const dstW = Math.round(srcW * scale), dstH = Math.round(srcH * scale);
  const offsetX = (size - dstW) / 2, offsetY = (size - dstH) / 2;
  return {
    affine: { a: 1/scale, b: 0, c: -offsetX/scale, d: 0, e: 1/scale, f: -offsetY/scale },
    letterbox: { scaleX: dstW/size, scaleY: dstH/size, offsetX: offsetX/size, offsetY: offsetY/size },
  };
}

async function loadModel(jsonUrl, binUrl) {
  const graph = await (await fetch(jsonUrl)).json();
  const allWeights = new Float32Array(await (await fetch(binUrl)).arrayBuffer());
  const W = {};
  for (const [name, info] of Object.entries(graph.weights)) {
    if (info.length === 0) continue;
    const buf = device.createBuffer({ size: Math.max(info.length * 4, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, allWeights.subarray(info.offset, info.offset + info.length));
    W[name] = buf;
  }
  const allWeightsBuf = device.createBuffer({ size: allWeights.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(allWeightsBuf, 0, allWeights);
  return { graph, allWeights, W, allWeightsBuf };
}

async function init() {
  const adapter = await navigator.gpu.requestAdapter();
  device = await adapter.requestDevice();
  sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  // Create shared pipelines
  const mkP = async (name) => {
    const code = await (await fetch(`../engine/${name}.wgsl`)).text();
    return device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  };
  const P = {
    conv2d: await mkP('conv2d'), maxpool: await mkP('maxpool'), resize: await mkP('resize'),
    gemm: await mkP('gemm'), global_avg_pool: await mkP('global_avg_pool'),
    add: await mkP('add'), pad_channels: await mkP('pad_channels'),
    fused_block: await mkP('fused_block'), transpose_nhwc: await mkP('transpose_nhwc'),
  };

  // Setup warp shaders
  setupModel('palm', PALM_SIZE);
  setupModel('hand', HAND_SIZE);
  setupModel('faceDet', FACE_DET_SIZE);
  setupModel('faceLm', FACE_LM_SIZE);

  // Load and compile all 4 models
  const [palm, hand, faceDet, faceLm] = await Promise.all([
    loadModel('../models/palm_detection_lite.json', '../models/palm_detection_lite.bin'),
    loadModel('../models/hand_landmark_full.json', '../models/hand_landmark_full.bin'),
    loadModel('../models/face_detector.json', '../models/face_detector.bin'),
    loadModel('../models/face_landmarks_detector.json', '../models/face_landmarks_detector.bin'),
  ]);

  palmRunner = new ModelRunner(device, P, palm.W, palm.allWeightsBuf);
  await palmRunner.compile(palm.graph, models.palm.inputBuf, palm.allWeights);

  // Two hand runners with separate buffers for true parallel two-hand inference
  setupModel('hand0', HAND_SIZE);
  setupModel('hand1', HAND_SIZE);
  // Each runner needs its own weight buffers (they share the same weights data but separate GPU buffers)
  const handW0 = {}, handW1 = {};
  for (const [name, info] of Object.entries(hand.graph.weights)) {
    if (info.length === 0) continue;
    const BF = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const buf0 = device.createBuffer({ size: Math.max(info.length * 4, 4), usage: BF });
    device.queue.writeBuffer(buf0, 0, hand.allWeights.subarray(info.offset, info.offset + info.length));
    handW0[name] = buf0;
    const buf1 = device.createBuffer({ size: Math.max(info.length * 4, 4), usage: BF });
    device.queue.writeBuffer(buf1, 0, hand.allWeights.subarray(info.offset, info.offset + info.length));
    handW1[name] = buf1;
  }
  const handWBuf0 = device.createBuffer({ size: hand.allWeights.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(handWBuf0, 0, hand.allWeights);
  const handWBuf1 = device.createBuffer({ size: hand.allWeights.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(handWBuf1, 0, hand.allWeights);

  handRunners[0] = new ModelRunner(device, P, handW0, handWBuf0);
  await handRunners[0].compile(hand.graph, models.hand0.inputBuf, hand.allWeights);
  handRunners[1] = new ModelRunner(device, P, handW1, handWBuf1);
  await handRunners[1].compile(hand.graph, models.hand1.inputBuf, hand.allWeights);

  faceDetRunner = new ModelRunner(device, P, faceDet.W, faceDet.allWeightsBuf);
  await faceDetRunner.compile(faceDet.graph, models.faceDet.inputBuf, faceDet.allWeights);

  faceLmRunner = new ModelRunner(device, P, faceLm.W, faceLm.allWeightsBuf);
  await faceLmRunner.compile(faceLm.graph, models.faceLm.inputBuf, faceLm.allWeights);

  // Discover hand output names
  const handOut = await handRunners[0].runCompiled();
  for (const [name, data] of Object.entries(handOut)) {
    if (data.length === 63 && !handOutputNames.landmarks) handOutputNames.landmarks = name;
    else if (data.length === 63) handOutputNames.worldLandmarks = name;
    else if (data.length === 1 && !handOutputNames.handFlag) handOutputNames.handFlag = name;
    else if (data.length === 1) handOutputNames.handedness = name;
  }

  // Discover face landmark output names
  const faceOut = await faceLmRunner.runCompiled();
  for (const [name, data] of Object.entries(faceOut)) {
    if (data.length === NUM_FACE_LM * 3) faceLmOutputNames.landmarks = name;
    else if (data.length === 1 && !faceLmOutputNames.faceFlag) faceLmOutputNames.faceFlag = name;
  }

  palmAnchors = generateAnchors();
  faceAnchors = generateFaceAnchors();
}

// ── Batched command encoding ──
// Queue incoming requests, encode ALL into one command encoder, submit once,
// then read back all outputs and respond. This mimics how ORT batches
// multiple model executions on one device.

let _queue = [];
let _flushScheduled = false;

function enqueue(entry) {
  _queue.push(entry);
  if (!_flushScheduled) {
    _flushScheduled = true;
    // Flush on next microtask -- all sync messages in the current event loop
    // tick get batched into one GPU submit
    Promise.resolve().then(flushQueue);
  }
}

async function flushQueue() {
  _flushScheduled = false;
  const batch = _queue;
  _queue = [];
  if (batch.length === 0) return;

  // ONE encoder for everything: warps + inference + readback copies
  const enc = device.createCommandEncoder();

  // Phase 1: Upload bitmaps as shared textures, encode warps
  // Group by bitmap to share texture uploads where possible
  const texturesToDestroy = [];
  for (const entry of batch) {
    if (entry.bitmap && entry.warpName && entry.affine) {
      const bmp = entry.bitmap;
      const srcTexture = device.createTexture({
        size: [bmp.width, bmp.height], format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      device.queue.copyExternalImageToTexture({ source: bmp }, { texture: srcTexture }, [bmp.width, bmp.height]);
      encodeWarp(enc, entry.warpName, srcTexture, bmp.width, bmp.height, entry.affine);
      bmp.close();
      texturesToDestroy.push(srcTexture);
    }
  }

  // Phase 2: Encode ALL model inferences into the SAME encoder
  for (const entry of batch) {
    entry.runner.encodeInto(enc);
  }

  // ONE submit for everything
  device.queue.submit([enc.finish()]);

  // Clean up textures
  for (const t of texturesToDestroy) t.destroy();

  // Phase 3: Read ALL outputs in parallel, then post results
  await Promise.all(batch.map(async (entry) => {
    try {
      const outputs = await entry.runner.readOutputs();
      entry.respond(outputs);
    } catch (err) {
      self.postMessage({ type: 'error', reqId: entry.reqId, message: err.message });
    }
  }));
}

self.onmessage = async (e) => {
  const { type, reqId } = e.data;

  if (type === 'init') {
    try {
      await init();
      console.log('[vision-worker] ready: 4 models compiled on 1 GPU device');
      self.postMessage({ type: 'ready' });
    } catch (err) {
      console.error('[vision-worker] init error:', err);
      self.postMessage({ type: 'error', message: err.message });
    }
    return;
  }

  if (type === 'palmDetect') {
    const { bitmap } = e.data;
    const { affine, letterbox } = letterboxAffine(PALM_SIZE, bitmap.width, bitmap.height);
    enqueue({
      reqId,
      runner: palmRunner,
      bitmap, warpName: 'palm', affine,
      respond: (outputs) => {
        let regressors, scores;
        for (const [, data] of Object.entries(outputs)) {
          if (data.length > 2016) regressors = data;
          else if (data.length === 2016) scores = data;
        }
        let detections = decodeDetections(regressors, scores, palmAnchors);
        detections = weightedNMS(detections);
        self.postMessage({ type: 'palmDetections', reqId, detections, letterbox });
      },
    });
    return;
  }

  if (type === 'handLandmark') {
    const { bitmap, rect, vw, vh, slot = 0 } = e.data;
    const inv = computeAffineParams(rect, HAND_SIZE);
    if (!inv) { bitmap.close(); self.postMessage({ type: 'handResult', reqId, handFlag: 0 }); return; }
    const warpName = slot === 0 ? 'hand0' : 'hand1';
    enqueue({
      reqId,
      runner: handRunners[slot],
      bitmap, warpName, affine: inv,
      respond: (outputs) => {
        const rawLM = handOutputNames.landmarks ? outputs[handOutputNames.landmarks] : null;
        const handFlag = handOutputNames.handFlag ? outputs[handOutputNames.handFlag][0] : 0;
        const handednessRaw = handOutputNames.handedness ? outputs[handOutputNames.handedness][0] : 0.5;
        let projected = null;
        if (rawLM && rawLM.length === 63) {
          projected = new Float32Array(63);
          for (let i = 0; i < 21; i++) {
            projected[i*3]   = (inv.a * rawLM[i*3] + inv.b * rawLM[i*3+1] + inv.c) / vw;
            projected[i*3+1] = (inv.d * rawLM[i*3] + inv.e * rawLM[i*3+1] + inv.f) / vh;
            projected[i*3+2] = rawLM[i*3+2] / HAND_SIZE;
          }
        }
        self.postMessage({
          type: 'handResult', reqId, handFlag,
          handedness: handednessRaw > 0.5 ? 'Right' : 'Left',
          landmarks: projected?.buffer || null,
        }, projected ? [projected.buffer] : []);
      },
    });
    return;
  }

  if (type === 'faceDetect') {
    const { bitmap } = e.data;
    const { affine, letterbox } = letterboxAffine(FACE_DET_SIZE, bitmap.width, bitmap.height);
    enqueue({
      reqId,
      runner: faceDetRunner,
      bitmap, warpName: 'faceDet', affine,
      respond: (outputs) => {
        let regressors, scores;
        for (const [, data] of Object.entries(outputs)) {
          if (data.length > 896) regressors = data;
          else if (data.length === 896) scores = data;
        }
        let detections = decodeFaceDetections(regressors, scores, faceAnchors);
        detections = faceNMS(detections);
        self.postMessage({ type: 'faceDetections', reqId, detections, letterbox });
      },
    });
    return;
  }

  if (type === 'faceLandmark') {
    const { bitmap, rect, vw, vh } = e.data;
    const inv = computeAffineParams(rect, FACE_LM_SIZE);
    if (!inv) { bitmap.close(); self.postMessage({ type: 'faceResult', reqId, faceFlag: 0 }); return; }
    enqueue({
      reqId,
      runner: faceLmRunner,
      bitmap, warpName: 'faceLm', affine: inv,
      respond: (outputs) => {
        const rawLM = faceLmOutputNames.landmarks ? outputs[faceLmOutputNames.landmarks] : null;
        const faceFlag = faceLmOutputNames.faceFlag ? outputs[faceLmOutputNames.faceFlag][0] : 0;
        let projected = null;
        if (rawLM && rawLM.length === NUM_FACE_LM * 3) {
          projected = new Float32Array(NUM_FACE_LM * 3);
          for (let i = 0; i < NUM_FACE_LM; i++) {
            projected[i*3]   = (inv.a * rawLM[i*3] + inv.b * rawLM[i*3+1] + inv.c) / vw;
            projected[i*3+1] = (inv.d * rawLM[i*3] + inv.e * rawLM[i*3+1] + inv.f) / vh;
            projected[i*3+2] = rawLM[i*3+2] / FACE_LM_SIZE;
          }
        }
        const rawCopy = rawLM ? new Float32Array(rawLM).buffer : null;
        const transfers = [];
        if (projected) transfers.push(projected.buffer);
        if (rawCopy) transfers.push(rawCopy);
        self.postMessage({
          type: 'faceResult', reqId, faceFlag,
          landmarks: projected?.buffer || null,
          rawLandmarks: rawCopy, modelSize: FACE_LM_SIZE,
        }, transfers);
      },
    });
    return;
  }
};
