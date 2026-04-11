// Hand landmark worker: runs MediaPipe HandLandmarker off main thread
// Key technique: receives ImageBitmap via transferable, returns 21 3D landmarks
importScripts('mediapipe-vision.js');

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task';

let landmarker = null;

async function init(numHands) {
  const vision = await $mediapipe.FilesetResolver.forVisionTasks(WASM_URL);
  landmarker = await $mediapipe.HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numHands: numHands || 1,
    minHandDetectionConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });
  self.postMessage({ type: 'init', ok: true });
}

self.onmessage = async (e) => {
  const { type, image, timestamp, numHands } = e.data;
  if (type === 'init') { await init(numHands); return; }
  if (type === 'detect' && landmarker) {
    const result = landmarker.detectForVideo(image, timestamp);
    image.close(); // Release the transferred ImageBitmap
    let landmarks = null;
    if (result.landmarks && result.landmarks.length > 0) {
      landmarks = result.landmarks.map(hand => hand.map(p => ({ x: p.x, y: p.y, z: p.z })));
    }
    self.postMessage({ type: 'result', landmarks });
  }
};
