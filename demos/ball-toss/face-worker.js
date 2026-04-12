// MediaPipe face worker: holds BOTH FaceDetector (BlazeFace, 6kp, fast)
// and FaceLandmarker (FaceMesh, 478pt, precise) so the demo can A/B them
// without tearing down the worker. Mode is set on init and switchable via
// {type:'setMode'}. Each frame's detect call branches on the current mode
// and returns the same {keypoints:[leftEye, rightEye]} shape so the main
// thread's handleFaceResult doesn't need to know which model produced it.
importScripts('mediapipe-vision.js');

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const DETECTOR_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite';
const LANDMARKER_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

let detector = null;
let landmarker = null;
let mode = 'detector';

async function init(initialMode) {
  if (initialMode) mode = initialMode;
  const vision = await $mediapipe.FilesetResolver.forVisionTasks(WASM_URL);
  detector = await $mediapipe.FaceDetector.createFromOptions(vision, {
    baseOptions: { modelAssetPath: DETECTOR_MODEL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    minDetectionConfidence: 0.3,
  });
  landmarker = await $mediapipe.FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: LANDMARKER_MODEL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numFaces: 1,
    minFaceDetectionConfidence: 0.3,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
  self.postMessage({ type: 'init', ok: true });
}

self.onmessage = async (e) => {
  const { type, image, timestamp } = e.data;
  if (type === 'init') { await init(e.data.mode); return; }
  if (type === 'setMode') { mode = e.data.mode; return; }
  if (type === 'detect') {
    let face = null;
    if (mode === 'landmark' && landmarker) {
      const result = landmarker.detectForVideo(image, timestamp);
      if (result.faceLandmarks && result.faceLandmarks.length > 0) {
        const lm = result.faceLandmarks[0];
        // Inner eye corners: 263 (left), 33 (right). Same indices the WebGPU
        // landmark path uses, so the parallax math gets identical inputs and
        // we're comparing inference backends, not landmark conventions.
        face = { keypoints: [{ x: lm[263].x, y: lm[263].y }, { x: lm[33].x, y: lm[33].y }] };
      }
    } else if (detector) {
      const result = detector.detectForVideo(image, timestamp);
      if (result.detections && result.detections.length > 0) {
        face = { keypoints: result.detections[0].keypoints.map(k => ({ x: k.x, y: k.y })) };
      }
    }
    image.close();
    self.postMessage({ type: 'result', face });
  }
};
