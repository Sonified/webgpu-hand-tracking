// Landmark inference worker: receives ImageBitmap + rect, does GPU warp + inference.
// WebGPU compute shader does the affine warp + normalization -- no canvas, no JS pixel loops.

import * as ort from 'onnxruntime-web/webgpu';

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

const S = 224; // landmark model input size

let session = null;
let outputMap = null;

// WebGPU warp pipeline state
let gpuDevice = null;
let warpPipeline = null;
let outputBuffer = null;
let readBuffer = null;
let uniformBuffer = null;
let sampler = null;

const WGSL_SHADER = `
struct Uniforms {
  // Inverse affine: maps output pixel (x,y) to source pixel (sx,sy)
  // sx = a*x + b*y + c
  // sy = d*x + e*y + f
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

  // Inverse affine: output pixel -> source pixel
  let sx = u.a * fx + u.b * fy + u.c;
  let sy = u.d * fx + u.e * fy + u.f;

  // Normalize to [0,1] UV for sampling
  let uv = vec2f(sx / u.src_w, sy / u.src_h);

  // Sample with bilinear filtering (sampler handles it)
  let pixel = textureSampleLevel(srcTexture, srcSampler, uv, 0.0);

  // Write RGB to output (already [0,1] from texture)
  let idx = (y * ${S}u + x) * 3u;
  output[idx]      = pixel.r;
  output[idx + 1u] = pixel.g;
  output[idx + 2u] = pixel.b;
}
`;

async function initGPU() {
  const adapter = await navigator.gpu.requestAdapter();
  gpuDevice = await adapter.requestDevice();

  const shaderModule = gpuDevice.createShaderModule({ code: WGSL_SHADER });

  const bindGroupLayout = gpuDevice.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: undefined,
        buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  warpPipeline = gpuDevice.createComputePipeline({
    layout: gpuDevice.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: shaderModule, entryPoint: 'main' },
  });

  // Output buffer: 224 * 224 * 3 floats
  const outputSize = S * S * 3 * 4;
  outputBuffer = gpuDevice.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Read buffer for mapping back to CPU
  readBuffer = gpuDevice.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // Uniform buffer: 10 floats padded to 48 bytes (3 vec4s)
  uniformBuffer = gpuDevice.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  sampler = gpuDevice.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
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
  // Forward affine: source pixel -> output pixel
  const fa = S * (dy2 - dy1) * invDet;
  const fb = S * (dx1 - dx2) * invDet;
  const fd = S * (-dy1) * invDet;
  const fe = S * dx1 * invDet;
  const fc = -fa * p1x - fb * p1y;
  const ff = -fd * p1x - fe * p1y;

  // Inverse affine: output pixel -> source pixel
  const detM = fa * fe - fb * fd;
  const inv = {
    a: fe / detM, b: -fb / detM,
    c: -(fe * fc - fb * ff) / detM,
    d: -fd / detM, e: fa / detM,
    f: -(-fd * fc + fa * ff) / detM,
  };

  return inv;
}

function rotatedRectPoints(cx, cy, w, h, rotation) {
  const b = Math.cos(rotation) * 0.5;
  const a = Math.sin(rotation) * 0.5;
  const p0x = cx - a * h - b * w, p0y = cy + b * h - a * w;
  const p1x = cx + a * h - b * w, p1y = cy - b * h - a * w;
  return [[p0x, p0y], [p1x, p1y], [2 * cx - p0x, 2 * cy - p0y], [2 * cx - p1x, 2 * cy - p1y]];
}

async function gpuWarp(bitmap, rect) {
  const inv = computeAffineParams(rect);
  if (!inv) return null;

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
  const uniforms = new Float32Array([
    inv.a, inv.b, inv.c, 0, // vec4 (padded)
    inv.d, inv.e, inv.f, 0, // vec4 (padded)
    bitmap.width, bitmap.height, 0, 0, // vec4 (padded)
  ]);
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, uniforms);

  // Create bind group
  const bindGroup = gpuDevice.createBindGroup({
    layout: warpPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: srcTexture.createView() },
      { binding: 1, resource: sampler },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  });

  // Dispatch compute shader
  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(warpPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(S / 16), Math.ceil(S / 16));
  pass.end();

  // Copy output to read buffer
  encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, S * S * 3 * 4);
  gpuDevice.queue.submit([encoder.finish()]);

  // Read back result
  await readBuffer.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();

  // Clean up texture (created per frame)
  srcTexture.destroy();

  return { data: result, inv };
}

// Fallback: canvas-based crop (for browsers without worker WebGPU)
const warpCanvas = new OffscreenCanvas(S, S);
const warpCtx = warpCanvas.getContext('2d', { willReadFrequently: true });
const inputBuffer = new Float32Array(S * S * 3);

function canvasWarp(bitmap, rect) {
  const inv = computeAffineParams(rect);
  if (!inv) return null;

  // Reconstruct forward affine from inverse
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
    inputBuffer[i * 3]     = rgba[i * 4]     / 255;
    inputBuffer[i * 3 + 1] = rgba[i * 4 + 1] / 255;
    inputBuffer[i * 3 + 2] = rgba[i * 4 + 2] / 255;
  }

  return { data: inputBuffer, inv };
}

let useGPUWarp = false;

self.onmessage = async (e) => {
  const { type } = e.data;

  if (type === 'init') {
    try {
      // Try to init WebGPU for compute shader warp
      if (typeof navigator !== 'undefined' && navigator.gpu) {
        try {
          await initGPU();
          useGPUWarp = true;
          console.log('Worker: WebGPU compute warp enabled');
        } catch (gpuErr) {
          console.warn('Worker: WebGPU compute warp unavailable, using canvas fallback:', gpuErr.message);
        }
      }

      session = await ort.InferenceSession.create(e.data.modelUrl, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      });

      const warmup = new ort.Tensor('float32', new Float32Array(S * S * 3), [1, S, S, 3]);
      const results = await session.run({ [session.inputNames[0]]: warmup });

      outputMap = {};
      for (const name of session.outputNames) {
        const size = results[name].data.length;
        if (size === 63 && !outputMap.landmarks) outputMap.landmarks = name;
        else if (size === 63) outputMap.worldLandmarks = name;
        else if (size === 1 && !outputMap.handFlag) outputMap.handFlag = name;
        else if (size === 1) outputMap.handedness = name;
      }

      self.postMessage({ type: 'ready', gpuWarp: useGPUWarp });
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }

  if (type === 'infer') {
    try {
      const { bitmap, rect, vw, vh } = e.data;

      // GPU warp or canvas fallback
      const warpResult = useGPUWarp
        ? await gpuWarp(bitmap, rect)
        : canvasWarp(bitmap, rect);

      bitmap.close();

      if (!warpResult) {
        self.postMessage({ type: 'result', handFlag: 0, landmarks: null });
        return;
      }

      const { data, inv } = warpResult;

      const tensor = new ort.Tensor('float32', data, [1, S, S, 3]);
      const results = await session.run({ [session.inputNames[0]]: tensor });

      const rawLandmarks = outputMap.landmarks ? results[outputMap.landmarks].data : null;
      const handFlag = outputMap.handFlag ? results[outputMap.handFlag].data[0] : 0;
      const handedness = outputMap.handedness ? results[outputMap.handedness].data[0] : 0;

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
