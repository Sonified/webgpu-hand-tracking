// Face landmark inference worker: FULL GPU pipeline -- zero CPU readback for preprocessing.
// Compute shader warp -> GPU buffer -> ONNX tensor (same device) -> inference -> 478 landmarks out.

import * as ort from 'onnxruntime-web/webgpu';

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

const S = 256; // face landmark model input size
const NUM_LANDMARKS = 478;
const LANDMARK_FLOATS = NUM_LANDMARKS * 3; // 1434

// 146 landmark indices used as input to the blendshape model
// Source: mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc
const BLENDSHAPE_LANDMARK_INDICES = [
  0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54,
  55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95,
  103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152,
  153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178,
  181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282,
  283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314,
  317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374,
  375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397,
  398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473, 474,
  475, 476, 477,
];

let session = null;
let blendshapeSession = null;
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
      });

      // Get ONNX RT's WebGPU device and build our warp shader on it
      if (typeof navigator !== 'undefined' && navigator.gpu) {
        try {
          const onnxDevice = await ort.env.webgpu.device;
          await initGPU(onnxDevice);
          useGPUDirect = true;
          console.log('Face landmark worker: GPU direct path (zero-copy)');
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

      console.log('Face landmark output map:', outputMap,
        'output names:', session.outputNames,
        'output sizes:', session.outputNames.map(n => results[n].data.length));

      // Load blendshape model if URL provided
      if (e.data.blendshapeUrl) {
        blendshapeSession = await ort.InferenceSession.create(e.data.blendshapeUrl, {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all',
        });
        // Warmup
        const bsWarmup = new ort.Tensor('float32', new Float32Array(146 * 2), [1, 146, 2]);
        await blendshapeSession.run({ [blendshapeSession.inputNames[0]]: bsWarmup });
        console.log('Face blendshape model loaded');
      }

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

      // Run blendshape model if available -- takes 146 selected landmarks (x,y only)
      let blendshapes = null;
      if (blendshapeSession && rawLandmarks && rawLandmarks.length === LANDMARK_FLOATS) {
        const bsInput = new Float32Array(146 * 2);
        for (let i = 0; i < 146; i++) {
          const idx = BLENDSHAPE_LANDMARK_INDICES[i];
          bsInput[i * 2]     = rawLandmarks[idx * 3]     / S; // x normalized
          bsInput[i * 2 + 1] = rawLandmarks[idx * 3 + 1] / S; // y normalized
        }
        const bsTensor = new ort.Tensor('float32', bsInput, [1, 146, 2]);
        const bsResults = await blendshapeSession.run({ [blendshapeSession.inputNames[0]]: bsTensor });
        blendshapes = new Float32Array(bsResults[blendshapeSession.outputNames[0]].data);
      }

      const transferables = [];
      if (projectedLandmarks) transferables.push(projectedLandmarks.buffer);
      if (blendshapes) transferables.push(blendshapes.buffer);

      self.postMessage({
        type: 'result',
        faceFlag,
        landmarks: projectedLandmarks ? projectedLandmarks.buffer : null,
        blendshapes: blendshapes ? blendshapes.buffer : null,
      }, transferables);
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }
};
