// Model URL resolution: local in dev, CDN in production.

const CDN_BASE = 'https://models.now.audio';
const LOCAL_BASE = '/models';

const isLocal = typeof location !== 'undefined' && location.hostname === 'localhost';
const BASE = isLocal ? LOCAL_BASE : CDN_BASE;

export const PALM_MODEL_URL = `${BASE}/palm_detection_lite.onnx`;
export const HAND_LANDMARK_URL = `${BASE}/hand_landmark_full.onnx`;
export const FACE_DETECTOR_URL = `${BASE}/face_detector.onnx`;
export const FACE_LANDMARK_URL = `${BASE}/face_landmarks_detector.onnx`;
