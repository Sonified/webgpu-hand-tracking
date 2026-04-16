// Face tracking pipeline: ALL inference in workers. Main thread is pure orchestration.
// - Face detection: dedicated worker with GPU letterbox
// - Landmark inference: one worker with GPU affine warp (single face for now)

import { faceDetectionToRect } from './face-nms.js';
import { FACE_DETECTOR_URL, FACE_LANDMARK_URL, FACE_BLENDSHAPE_URL } from './model-urls.js';
const FACE_FLAG_THRESHOLD = 0.5;

// Rate-limited logger
function makeLogger(intervalMs = 2000) {
  let lastLog = 0;
  return function(msg, ...args) {
    const now = performance.now();
    if (now - lastLog > intervalMs) {
      console.log(msg, ...args);
      lastLog = now;
    }
  };
}
const logDetect = makeLogger(2000);
const logSlot = makeLogger(2000);
const logLandmark = makeLogger(2000);

/**
 * Wraps the face detection worker.
 */
class FaceDetectionWorker {
  constructor() {
    this.worker = new Worker(
      new URL('./face-detection-worker-wgsl.js', import.meta.url),
      { type: 'module' }
    );
    this.pendingResolve = null;
    this.worker.onmessage = (e) => this._onMessage(e);
  }

  init(modelUrl) {
    return new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => {
        if (e.data.type === 'ready') {
          this.worker.onmessage = (ev) => this._onMessage(ev);
          resolve();
        } else if (e.data.type === 'error') {
          reject(new Error(e.data.message));
        }
      };
      this.worker.postMessage({ type: 'init', modelUrl });
    });
  }

  detect(frame) {
    return new Promise((resolve) => {
      this.pendingResolve = resolve;
      this.worker.postMessage({ type: 'detect', frame }, [frame]);
    });
  }

  _onMessage(e) {
    if (e.data.type === 'detections' && this.pendingResolve) {
      const resolve = this.pendingResolve;
      this.pendingResolve = null;
      resolve({ detections: e.data.detections, letterbox: e.data.letterbox });
    } else if (e.data.type === 'error') {
      console.error('Face detection worker error:', e.data.message);
      if (this.pendingResolve) {
        this.pendingResolve({ detections: [], letterbox: {} });
        this.pendingResolve = null;
      }
    }
  }
}

/**
 * Wraps the face landmark inference worker.
 */
class FaceLandmarkWorker {
  constructor() {
    this.worker = new Worker(
      new URL('./face-landmark-worker-wgsl.js', import.meta.url),
      { type: 'module' }
    );
    this.pendingResolve = null;
    this.worker.onmessage = (e) => this._onMessage(e);
  }

  init(modelUrl, blendshapeUrl) {
    return new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => {
        if (e.data.type === 'ready') {
          this.worker.onmessage = (ev) => this._onMessage(ev);
          resolve();
        } else if (e.data.type === 'error') {
          reject(new Error(e.data.message));
        }
      };
      this.worker.postMessage({ type: 'init', modelUrl, blendshapeUrl });
    });
  }

  infer(frame, rect, vw, vh, runBlendshapes = true) {
    return new Promise((resolve) => {
      this.pendingResolve = resolve;
      this.worker.postMessage(
        { type: 'infer', frame, rect, vw, vh, runBlendshapes },
        [frame]
      );
    });
  }

  _onMessage(e) {
    if (e.data.type === 'result' && this.pendingResolve) {
      const resolve = this.pendingResolve;
      this.pendingResolve = null;

      let landmarks = [];
      if (e.data.landmarks) {
        const flat = new Float32Array(e.data.landmarks);
        for (let i = 0; i < 478; i++) {
          landmarks.push({
            x: flat[i * 3],
            y: flat[i * 3 + 1],
            z: flat[i * 3 + 2],
          });
        }
      }

      resolve({
        landmarks,
        faceFlag: e.data.faceFlag,
        rawLandmarks: e.data.rawLandmarks,
        modelSize: e.data.modelSize,
      });
    } else if (e.data.type === 'error') {
      console.error('Face landmark worker error:', e.data.message);
      if (this.pendingResolve) {
        this.pendingResolve({ landmarks: [], faceFlag: 0 });
        this.pendingResolve = null;
      }
    }
  }
}

/**
 * Wraps the blendshape inference worker.
 */
class BlendshapeWorker {
  constructor() {
    this.worker = new Worker(
      new URL('./face-blendshape-worker.js', import.meta.url),
      { type: 'module' }
    );
    this.pendingResolve = null;
    this.worker.onmessage = (e) => this._onMessage(e);
    this.lastBlendshapes = null; // cache latest result
  }

  init(modelUrl) {
    return new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => {
        if (e.data.type === 'ready') {
          this.worker.onmessage = (ev) => this._onMessage(ev);
          resolve();
        } else if (e.data.type === 'error') {
          reject(new Error(e.data.message));
        }
      };
      this.worker.postMessage({ type: 'init', modelUrl });
    });
  }

  // Fire and forget -- don't await, just grab latest result
  infer(rawLandmarks, modelSize) {
    this.worker.postMessage(
      { type: 'infer', rawLandmarks, modelSize },
      [rawLandmarks]
    );
  }

  _onMessage(e) {
    if (e.data.type === 'result') {
      this.lastBlendshapes = new Float32Array(e.data.blendshapes);
    } else if (e.data.type === 'error') {
      console.error('Blendshape worker error:', e.data.message);
    }
  }
}

export class FaceTracker {
  constructor(numFaces = 1) {
    this.numFaces = numFaces;
    this.detectWorker = new FaceDetectionWorker();
    this.landmarkWorkers = [];
    this.blendshapeWorker = new BlendshapeWorker();
    this.slots = [];
    for (let i = 0; i < numFaces; i++) {
      const worker = new FaceLandmarkWorker();
      this.landmarkWorkers.push(worker);
      this.slots.push({ index: i, worker, active: false, rect: null, landmarks: null });
    }
    this.ready = false;
    this.running = false;
    this.detecting = false;
    this.pendingDetections = null;
  }

  async init(onStatus) {
    // Init workers sequentially (WebGPU EP requires it)
    onStatus?.('Loading face detection worker...');
    await this.detectWorker.init(FACE_DETECTOR_URL);

    for (let i = 0; i < this.landmarkWorkers.length; i++) {
      onStatus?.(`Loading face landmark worker ${i}...`);
      await this.landmarkWorkers[i].init(FACE_LANDMARK_URL);
    }

    onStatus?.('Loading blendshape worker...');
    await this.blendshapeWorker.init(FACE_BLENDSHAPE_URL);

    console.log(`All face workers ready (${this.numFaces} face slots) -- main thread is pure orchestration`);
    this.ready = true;
    onStatus?.('Ready');
  }

  async processFrame(video, { runBlendshapes = true } = {}) {
    if (!this.ready || this.running) return { faces: [] };
    this.running = true;

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    try {
      // If pending detections arrived from a previous detect, assign them now
      if (this.pendingDetections) {
        const { detections, letterbox } = this.pendingDetections;
        this.pendingDetections = null;

        const emptySlots = this.slots.filter(s => !s.active);
        for (const det of detections) {
          if (emptySlots.length === 0) break;

          det.cx = (det.cx - letterbox.offsetX) / letterbox.scaleX;
          det.cy = (det.cy - letterbox.offsetY) / letterbox.scaleY;
          det.w = det.w / letterbox.scaleX;
          det.h = det.h / letterbox.scaleY;
          for (const kp of det.keypoints) {
            kp.x = (kp.x - letterbox.offsetX) / letterbox.scaleX;
            kp.y = (kp.y - letterbox.offsetY) / letterbox.scaleY;
          }

          if (det.cy < -0.1 || det.cy > 1.1 || det.cx < -0.1 || det.cx > 1.1) continue;

          const detPx = det.cx * vw;
          const detPy = det.cy * vh;
          const overlapsTracked = this.slots.some(s => {
            if (!s.active) return false;
            const dx = s.rect.cx - detPx;
            const dy = s.rect.cy - detPy;
            return Math.sqrt(dx * dx + dy * dy) < s.rect.w * 0.5;
          });
          if (overlapsTracked) continue;

          const rect = faceDetectionToRect(det, vw, vh);
          const slot = emptySlots.shift();
          slot.active = true;
          slot.rect = rect;
          logSlot(`[new face] slot ${slot.index} cx=${rect.cx.toFixed(0)} cy=${rect.cy.toFixed(0)}`);
        }
      }

      // Fire off face detection async if there are empty slots and we're not already detecting
      const hasEmptySlots = this.slots.some(s => !s.active);
      if (hasEmptySlots && !this.detecting) {
        this.detecting = true;
        const frame = new VideoFrame(video);
        this.detectWorker.detect(frame).then(result => {
          this.detecting = false;
          if (result.detections.length > 0) {
            logDetect(`[face detect] ${result.detections.length} detections`);
            this.pendingDetections = result;
          }
        }).catch(() => { this.detecting = false; });
      }

      // Landmark inference for active slot
      const results = await Promise.all(this.slots.map(async (slot) => {
        if (!slot.active) return null;

        const frame = new VideoFrame(video);
        const result = await slot.worker.infer(frame, slot.rect, vw, vh);

        if (result.faceFlag > FACE_FLAG_THRESHOLD) {
          slot.landmarks = result.landmarks;
          slot.rect = this.landmarksToRect(result.landmarks, vw, vh);

          // Fire blendshapes async -- don't wait, grab latest from previous frame
          if (runBlendshapes && result.rawLandmarks) {
            this.blendshapeWorker.infer(result.rawLandmarks, result.modelSize);
          }

          return {
            landmarks: result.landmarks,
            blendshapes: this.blendshapeWorker.lastBlendshapes,
          };
        } else {
          slot.active = false;
          slot.landmarks = null;
          return null;
        }
      }));

      logLandmark(`[tracking] slots: ${this.slots.map(s => s.active ? '1' : '0').join(',')}`);

      return {
        faces: results.filter(Boolean),
        debug: {
          rects: this.slots.filter(s => s.rect).map(s => s.rect),
        },
      };
    } catch (err) {
      console.error('processFrame error:', err.message, err.stack);
      return { faces: [] };
    } finally {
      this.running = false;
    }
  }

  landmarksToRect(landmarks, imgW, imgH) {
    // Rotation from eye-to-eye angle
    // Landmark 33 = right eye inner corner, 263 = left eye inner corner
    const rightEye = landmarks[33];
    const leftEye = landmarks[263];
    const angle = Math.atan2(leftEye.y - rightEye.y, leftEye.x - rightEye.x);

    // Bounding box of all 478 landmarks
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const lm of landmarks) {
      const px = lm.x * imgW;
      const py = lm.y * imgH;
      minX = Math.min(minX, px);
      minY = Math.min(minY, py);
      maxX = Math.max(maxX, px);
      maxY = Math.max(maxY, py);
    }

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const bw = maxX - minX;
    const bh = maxY - minY;
    const size = Math.max(bw, bh) * 1.5;

    return { cx, cy, w: size, h: size, angle };
  }
}
