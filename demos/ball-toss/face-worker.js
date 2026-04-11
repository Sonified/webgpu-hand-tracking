// Face detection worker: runs MediaPipe FaceDetector off main thread
// Key technique: receives ImageBitmap via transferable, returns face keypoints
importScripts('mediapipe-vision.js');

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite';

let detector = null;

async function init() {
  const vision = await $mediapipe.FilesetResolver.forVisionTasks(WASM_URL);
  detector = await $mediapipe.FaceDetector.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    minDetectionConfidence: 0.3,
  });
  self.postMessage({ type: 'init', ok: true });
}

self.onmessage = async (e) => {
  const { type, image, timestamp } = e.data;
  if (type === 'init') { await init(); return; }
  if (type === 'detect' && detector) {
    const result = detector.detectForVideo(image, timestamp);
    image.close(); // Release the transferred ImageBitmap
    let face = null;
    if (result.detections && result.detections.length > 0) {
      face = { keypoints: result.detections[0].keypoints.map(k => ({ x: k.x, y: k.y })) };
    }
    self.postMessage({ type: 'result', face });
  }
};
