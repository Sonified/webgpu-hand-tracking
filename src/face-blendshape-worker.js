// Face blendshape worker: takes 146 landmark points, outputs 52 expression coefficients.
// Runs independently from the landmark worker so it doesn't block the next frame.

import * as ort from '../vendor/onnxruntime-web/ort.webgpu.min.mjs';

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = new URL('../vendor/onnxruntime-web/', import.meta.url).href;

let session = null;
let running = false;

// 146 landmark indices used as input to the blendshape model
// Source: mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc
const INDICES = [
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

self.onmessage = async (e) => {
  const { type } = e.data;

  if (type === 'init') {
    try {
      session = await ort.InferenceSession.create(e.data.modelUrl, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
        enableMemPattern: true,
      });

      // Warmup
      const warmup = new ort.Tensor('float32', new Float32Array(146 * 2), [1, 146, 2]);
      await session.run({ [session.inputNames[0]]: warmup });

      console.log('[blendshape-worker] ready');
      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }

  if (type === 'infer') {
    if (running) return; // skip if previous inference still running
    running = true;
    try {
      // rawLandmarks is a Float32Array of 1434 floats (478 * 3) in model space
      const raw = new Float32Array(e.data.rawLandmarks);
      const S = e.data.modelSize; // 256

      const input = new Float32Array(146 * 2);
      for (let i = 0; i < 146; i++) {
        const idx = INDICES[i];
        input[i * 2]     = raw[idx * 3]     / S;
        input[i * 2 + 1] = raw[idx * 3 + 1] / S;
      }

      const tensor = new ort.Tensor('float32', input, [1, 146, 2]);
      const results = await session.run({ [session.inputNames[0]]: tensor });
      const blendshapes = new Float32Array(results[session.outputNames[0]].data);

      self.postMessage({
        type: 'result',
        blendshapes: blendshapes.buffer,
      }, [blendshapes.buffer]);
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    } finally {
      running = false;
    }
  }
};
