// Face detection worker: receives ImageBitmap, does GPU letterbox + inference + decode + NMS.
// Returns detections with keypoints, ready for downstream processing.

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.webgpu.min.mjs';
import { generateFaceAnchors, decodeFaceDetections } from './face-anchors.js';
import { weightedNMS } from './face-nms.js';

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

const FACE_SIZE = 128;

let session = null;
let anchors = null;

// WebGPU letterbox pipeline (shared device with ONNX RT for zero-copy path)
let gpuDevice = null;
let letterboxPipeline = null;
let outputBuffer = null;
let readBuffer = null;
let uniformBuffer = null;
let sampler = null;
let useGPU = false;
let useGPUDirect = false; // true = shared device, fromGpuBuffer, no readback

const WGSL_LETTERBOX = `
struct Uniforms {
  scale: f32,
  offset_x: f32,
  offset_y: f32,
  src_w: f32,
  src_h: f32,
}

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> u: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x;
  let y = gid.y;
  if (x >= ${FACE_SIZE}u || y >= ${FACE_SIZE}u) { return; }

  let fx = f32(x) + 0.5;
  let fy = f32(y) + 0.5;

  // Undo letterbox: output pixel -> source pixel
  let sx = (fx - u.offset_x) / u.scale;
  let sy = (fy - u.offset_y) / u.scale;

  // Check if inside source bounds
  var r = 0.0; var g = 0.0; var b = 0.0;
  if (sx >= 0.0 && sx < u.src_w && sy >= 0.0 && sy < u.src_h) {
    let uv = vec2f(sx / u.src_w, sy / u.src_h);
    let pixel = textureSampleLevel(srcTexture, srcSampler, uv, 0.0);
    r = pixel.r;
    g = pixel.g;
    b = pixel.b;
  }

  let idx = (y * ${FACE_SIZE}u + x) * 3u;
  output[idx]      = r;
  output[idx + 1u] = g;
  output[idx + 2u] = b;
}
`;

async function initGPU(device) {
  gpuDevice = device || (await (await navigator.gpu.requestAdapter()).requestDevice());

  const shaderModule = gpuDevice.createShaderModule({ code: WGSL_LETTERBOX });

  letterboxPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'main' },
  });

  const outputSize = FACE_SIZE * FACE_SIZE * 3 * 4;
  outputBuffer = gpuDevice.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  readBuffer = gpuDevice.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  uniformBuffer = gpuDevice.createBuffer({
    size: 32, // 5 floats padded to 32 bytes
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  sampler = gpuDevice.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
}

async function gpuLetterbox(bitmap) {
  const srcW = bitmap.width;
  const srcH = bitmap.height;
  const scale = FACE_SIZE / Math.max(srcW, srcH);
  const dstW = Math.round(srcW * scale);
  const dstH = Math.round(srcH * scale);
  const offsetX = (FACE_SIZE - dstW) / 2;
  const offsetY = (FACE_SIZE - dstH) / 2;

  // Upload bitmap
  const srcTexture = gpuDevice.createTexture({
    size: [srcW, srcH],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  gpuDevice.queue.copyExternalImageToTexture(
    { source: bitmap },
    { texture: srcTexture },
    [srcW, srcH]
  );

  // Write uniforms (scale in pixels, not normalized)
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
    scale, offsetX, offsetY, srcW, srcH, 0, 0, 0,
  ]));

  const bindGroup = gpuDevice.createBindGroup({
    layout: letterboxPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: srcTexture.createView() },
      { binding: 1, resource: sampler },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  });

  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(letterboxPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(FACE_SIZE / 16), Math.ceil(FACE_SIZE / 16));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, FACE_SIZE * FACE_SIZE * 3 * 4);
  gpuDevice.queue.submit([encoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  srcTexture.destroy();

  const letterbox = {
    scaleX: dstW / FACE_SIZE,
    scaleY: dstH / FACE_SIZE,
    offsetX: offsetX / FACE_SIZE,
    offsetY: offsetY / FACE_SIZE,
  };

  return { data, letterbox };
}

// GPU-direct letterbox: stays on GPU, no CPU readback. Returns letterbox only.
function gpuLetterboxDirect(bitmap) {
  const srcW = bitmap.width;
  const srcH = bitmap.height;
  const scale = FACE_SIZE / Math.max(srcW, srcH);
  const dstW = Math.round(srcW * scale);
  const dstH = Math.round(srcH * scale);
  const offsetX = (FACE_SIZE - dstW) / 2;
  const offsetY = (FACE_SIZE - dstH) / 2;

  const srcTexture = gpuDevice.createTexture({
    size: [srcW, srcH],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  gpuDevice.queue.copyExternalImageToTexture(
    { source: bitmap },
    { texture: srcTexture },
    [srcW, srcH]
  );

  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
    scale, offsetX, offsetY, srcW, srcH, 0, 0, 0,
  ]));

  const bindGroup = gpuDevice.createBindGroup({
    layout: letterboxPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: srcTexture.createView() },
      { binding: 1, resource: sampler },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  });

  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(letterboxPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(FACE_SIZE / 16), Math.ceil(FACE_SIZE / 16));
  pass.end();
  gpuDevice.queue.submit([encoder.finish()]);
  srcTexture.destroy();

  return {
    scaleX: dstW / FACE_SIZE,
    scaleY: dstH / FACE_SIZE,
    offsetX: offsetX / FACE_SIZE,
    offsetY: offsetY / FACE_SIZE,
  };
}

// Canvas fallback
const faceCanvas = new OffscreenCanvas(FACE_SIZE, FACE_SIZE);
const faceCtx = faceCanvas.getContext('2d', { willReadFrequently: true });

function canvasLetterbox(bitmap) {
  const srcW = bitmap.width;
  const srcH = bitmap.height;
  const scale = FACE_SIZE / Math.max(srcW, srcH);
  const dstW = Math.round(srcW * scale);
  const dstH = Math.round(srcH * scale);
  const offsetX = (FACE_SIZE - dstW) / 2;
  const offsetY = (FACE_SIZE - dstH) / 2;

  faceCtx.fillStyle = '#000';
  faceCtx.fillRect(0, 0, FACE_SIZE, FACE_SIZE);
  faceCtx.drawImage(bitmap, offsetX, offsetY, dstW, dstH);

  const imageData = faceCtx.getImageData(0, 0, FACE_SIZE, FACE_SIZE);
  const rgba = imageData.data;
  const data = new Float32Array(FACE_SIZE * FACE_SIZE * 3);
  for (let i = 0; i < FACE_SIZE * FACE_SIZE; i++) {
    data[i * 3]     = rgba[i * 4]     / 255;
    data[i * 3 + 1] = rgba[i * 4 + 1] / 255;
    data[i * 3 + 2] = rgba[i * 4 + 2] / 255;
  }

  return {
    data,
    letterbox: {
      scaleX: dstW / FACE_SIZE,
      scaleY: dstH / FACE_SIZE,
      offsetX: offsetX / FACE_SIZE,
      offsetY: offsetY / FACE_SIZE,
    },
  };
}

self.onmessage = async (e) => {
  const { type } = e.data;

  if (type === 'init') {
    try {
      // Generate anchors
      anchors = generateFaceAnchors();

      // Create ONNX session FIRST -- it creates its own WebGPU device that we can share
      session = await ort.InferenceSession.create(e.data.modelUrl, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
        enableMemPattern: true,
      });

      // Now grab ONNX RT's device and build our compute shader on it (zero-copy path)
      if (typeof navigator !== 'undefined' && navigator.gpu) {
        try {
          const onnxDevice = await ort.env.webgpu.device;
          await initGPU(onnxDevice);
          useGPUDirect = true;
          useGPU = true;
          console.log('Face worker: GPU direct path enabled (zero CPU readback, shared device)');
        } catch (gpuErr) {
          console.warn('Face worker: GPU direct unavailable, trying standalone GPU:', gpuErr.message);
          try {
            await initGPU();
            useGPU = true;
            console.log('Face worker: WebGPU letterbox enabled (standalone device, with readback)');
          } catch (err2) {
            console.warn('Face worker: GPU letterbox unavailable:', err2.message);
          }
        }
      }

      // Warmup
      const warmup = new ort.Tensor('float32', new Float32Array(FACE_SIZE * FACE_SIZE * 3), [1, FACE_SIZE, FACE_SIZE, 3]);
      await session.run({ [session.inputNames[0]]: warmup });

      self.postMessage({ type: 'ready', gpuLetterbox: useGPU, gpuDirect: useGPUDirect });
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }

  if (type === 'detect') {
    try {
      const { bitmap } = e.data;

      let input, letterbox;

      if (useGPUDirect) {
        // FULL GPU PATH: compute shader -> GPU buffer -> ONNX tensor (no CPU readback!)
        letterbox = gpuLetterboxDirect(bitmap);
        bitmap.close();
        input = ort.Tensor.fromGpuBuffer(outputBuffer, {
          dataType: 'float32',
          dims: [1, FACE_SIZE, FACE_SIZE, 3],
        });
      } else if (useGPU) {
        const result = await gpuLetterbox(bitmap);
        bitmap.close();
        letterbox = result.letterbox;
        input = new ort.Tensor('float32', result.data, [1, FACE_SIZE, FACE_SIZE, 3]);
      } else {
        const result = canvasLetterbox(bitmap);
        bitmap.close();
        letterbox = result.letterbox;
        input = new ort.Tensor('float32', result.data, [1, FACE_SIZE, FACE_SIZE, 3]);
      }

      // Run inference
      const results = await session.run({ [session.inputNames[0]]: input });

      // Decode -- BlazeFace outputs 16 values per anchor (not 18 like palm)
      let regressors, scores;
      for (const name of session.outputNames) {
        const t = results[name];
        if (t.dims[t.dims.length - 1] === 16) regressors = t.data;
        else scores = t.data;
      }

      let detections = decodeFaceDetections(regressors, scores, anchors);
      detections = weightedNMS(detections);

      // Send detections + letterbox back
      self.postMessage({
        type: 'detections',
        detections,
        letterbox,
      });
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }
};
