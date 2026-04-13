// Face landmark inference worker: FULL GPU pipeline -- zero CPU readback for preprocessing.
// Compute shader warp -> GPU buffer -> ONNX tensor (same device) -> inference -> 478 landmarks out.

import * as ort from '../vendor/onnxruntime-web/ort.webgpu.min.mjs';

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = new URL('../vendor/onnxruntime-web/', import.meta.url).href;

const S = 256; // face landmark model input size
const NUM_LANDMARKS = 478;
const LANDMARK_FLOATS = NUM_LANDMARKS * 3; // 1434

let session = null;
let outputMap = null;

// WebGPU state
let gpuDevice = null;
let warpPipeline = null;
let warpOutputBuffer = null; // compute shader writes here
let warpReadBuffer = null;   // readback buffer for WASM backend
let uniformBuffer = null;
let gpuSampler = null;
let useGPUDirect = false; // true = zero-copy GPU path (WebGPU backend)
let useGPUWarp = false;   // true = GPU warp + readback (WASM backend)

const WGSL_SHADER = `
struct Uniforms {
  a: f32, b: f32, c: f32, _pad0: f32,
  d: f32, e: f32, f: f32, _pad1: f32,
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
  let sy = u.d * fx + u.e * fy + u.f;

  let uv = vec2f(sx / u.src_w, sy / u.src_h);
  let pixel = textureSampleLevel(srcTexture, srcSampler, uv, 0.0);

  let idx = (y * ${S}u + x) * 3u;
  output[idx]      = pixel.r;
  output[idx + 1u] = pixel.g;
  output[idx + 2u] = pixel.b;
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

async function initGPU(device) {
  gpuDevice = device;

  const shaderModule = gpuDevice.createShaderModule({ code: WGSL_SHADER });

  warpPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'main' },
  });

  const outputSize = S * S * 3 * 4;
  warpOutputBuffer = gpuDevice.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  warpReadBuffer = gpuDevice.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  uniformBuffer = gpuDevice.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  gpuSampler = gpuDevice.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
}

function dispatchWarp(bitmap, inv) {
  // Upload ImageBitmap to GPU texture
  const srcTexture = gpuDevice.createTexture({
    size: [bitmap.width, bitmap.height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  gpuDevice.queue.copyExternalImageToTexture(
    { source: bitmap },
    { texture: srcTexture },
    [bitmap.width, bitmap.height]
  );

  // Write inverse affine uniforms
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
    inv.a, inv.b, inv.c, 0,
    inv.d, inv.e, inv.f, 0,
    bitmap.width, bitmap.height, 0, 0,
  ]));

  // Dispatch compute shader
  const bindGroup = gpuDevice.createBindGroup({
    layout: warpPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: srcTexture.createView() },
      { binding: 1, resource: gpuSampler },
      { binding: 2, resource: { buffer: warpOutputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  });

  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(warpPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(S / 16), Math.ceil(S / 16));
  pass.end();
  gpuDevice.queue.submit([encoder.finish()]);
  // No sync needed -- same device, same queue, commands execute in order
  srcTexture.destroy();
}

// Canvas fallback for browsers without worker WebGPU
const warpCanvas = new OffscreenCanvas(S, S);
const warpCtx = warpCanvas.getContext('2d', { willReadFrequently: true });
const cpuInputBuffer = new Float32Array(S * S * 3);

function canvasWarp(bitmap, rect) {
  const inv = computeAffineParams(rect);
  if (!inv) return null;

  const detI = inv.a * inv.e - inv.b * inv.d;
  const fa = inv.e / detI, fb = -inv.b / detI;
  const fd = -inv.d / detI, fe = inv.a / detI;
  const fc = -(fa * inv.c + fb * inv.f);
  const ff = -(fd * inv.c + fe * inv.f);

  warpCtx.resetTransform();
  warpCtx.clearRect(0, 0, S, S);
  warpCtx.setTransform(fa, fd, fb, fe, fc, ff);
  warpCtx.drawImage(bitmap, 0, 0);
  warpCtx.resetTransform();

  const imageData = warpCtx.getImageData(0, 0, S, S);
  const rgba = imageData.data;
  for (let i = 0; i < S * S; i++) {
    cpuInputBuffer[i * 3]     = rgba[i * 4]     / 255;
    cpuInputBuffer[i * 3 + 1] = rgba[i * 4 + 1] / 255;
    cpuInputBuffer[i * 3 + 2] = rgba[i * 4 + 2] / 255;
  }

  return { cpuData: cpuInputBuffer, inv };
}

self.onmessage = async (e) => {
  const { type } = e.data;

  if (type === 'init') {
    try {
      // Create ONNX session with WebGPU backend
      // Model uses decomposed PReLU (Relu+Neg+Mul+Add) -- all GPU-native ops
      session = await ort.InferenceSession.create(e.data.modelUrl, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
        enableMemPattern: true,
      });

      // Get ONNX RT's WebGPU device and build our warp shader on it
      if (typeof navigator !== 'undefined' && navigator.gpu) {
        try {
          const onnxDevice = await ort.env.webgpu.device;
          await initGPU(onnxDevice);
          useGPUDirect = true;
        } catch (gpuErr) {
          console.warn('Face landmark worker: GPU warp unavailable, canvas fallback:', gpuErr.message);
        }
      }

      // Warmup run to discover output tensor names and sizes
      const warmup = new ort.Tensor('float32', new Float32Array(S * S * 3), [1, S, S, 3]);
      const results = await session.run({ [session.inputNames[0]]: warmup });

      // Map outputs by size:
      // Model outputs: Identity (1,1,1,1434) = landmarks, Identity_1 (1,1,1,1) = face score
      outputMap = {};
      for (const name of session.outputNames) {
        const data = results[name].data;
        if (data.length === LANDMARK_FLOATS) outputMap.landmarks = name;
        else if (data.length === 1 && !outputMap.faceFlag) outputMap.faceFlag = name;
      }

      console.log(`[face-landmark-worker] ready (GPU direct: ${useGPUDirect})`);
      self.postMessage({ type: 'ready', gpuDirect: useGPUDirect });
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }

  if (type === 'infer') {
    try {
      const { bitmap, rect, vw, vh } = e.data;

      const inv = computeAffineParams(rect);
      if (!inv) {
        bitmap.close();
        self.postMessage({ type: 'result', faceFlag: 0, landmarks: null });
        return;
      }

      let tensor;

      if (useGPUDirect) {
        // FULL GPU PATH: compute shader -> GPU buffer -> ONNX tensor (no CPU readback!)
        dispatchWarp(bitmap, inv);
        bitmap.close();

        // Create ONNX tensor directly from the GPU buffer -- zero copy
        tensor = ort.Tensor.fromGpuBuffer(warpOutputBuffer, {
          dataType: 'float32',
          dims: [1, S, S, 3],
        });
      } else if (useGPUWarp) {
        // GPU warp + readback: GPU does the affine transform, then we read back for WASM inference
        dispatchWarp(bitmap, inv);
        bitmap.close();

        const outputSize = S * S * 3 * 4;
        const encoder = gpuDevice.createCommandEncoder();
        encoder.copyBufferToBuffer(warpOutputBuffer, 0, warpReadBuffer, 0, outputSize);
        gpuDevice.queue.submit([encoder.finish()]);

        await warpReadBuffer.mapAsync(GPUMapMode.READ);
        const warpData = new Float32Array(warpReadBuffer.getMappedRange().slice(0));
        warpReadBuffer.unmap();

        tensor = new ort.Tensor('float32', warpData, [1, S, S, 3]);
      } else {
        // Canvas fallback
        const warpResult = canvasWarp(bitmap, rect);
        bitmap.close();
        if (!warpResult) {
          self.postMessage({ type: 'result', faceFlag: 0, landmarks: null });
          return;
        }
        tensor = new ort.Tensor('float32', warpResult.cpuData, [1, S, S, 3]);
      }

      const results = await session.run({ [session.inputNames[0]]: tensor });

      // Landmarks: flat array of 1434 floats, reshape to 478 x 3
      const rawLandmarks = outputMap.landmarks ? results[outputMap.landmarks].data : null;
      const faceFlag = outputMap.faceFlag ? results[outputMap.faceFlag].data[0] : 0;

      // Project 478 landmarks back to video space (only 1434 floats come to CPU)
      let projectedLandmarks = null;
      if (rawLandmarks && rawLandmarks.length === LANDMARK_FLOATS) {
        projectedLandmarks = new Float32Array(LANDMARK_FLOATS);
        for (let i = 0; i < NUM_LANDMARKS; i++) {
          const ox = rawLandmarks[i * 3];
          const oy = rawLandmarks[i * 3 + 1];
          const oz = rawLandmarks[i * 3 + 2];
          projectedLandmarks[i * 3]     = (inv.a * ox + inv.b * oy + inv.c) / vw;
          projectedLandmarks[i * 3 + 1] = (inv.d * ox + inv.e * oy + inv.f) / vh;
          projectedLandmarks[i * 3 + 2] = oz / S;
        }
      }

      // Send raw landmarks (model space) alongside projected ones for blendshape worker
      const rawCopy = rawLandmarks ? new Float32Array(rawLandmarks).buffer : null;
      const transferables = [];
      if (projectedLandmarks) transferables.push(projectedLandmarks.buffer);
      if (rawCopy) transferables.push(rawCopy);

      self.postMessage({
        type: 'result',
        faceFlag,
        landmarks: projectedLandmarks ? projectedLandmarks.buffer : null,
        rawLandmarks: rawCopy,
        modelSize: S,
      }, transferables);
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }
};
