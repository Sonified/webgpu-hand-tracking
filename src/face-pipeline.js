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
      new URL('./face-detection-worker.js', import.meta.url),
      { type: 'module' }
    );
    this.pendingResolve = null;
    this.worker.onmessage = (e) => this._onMessage(e);
  }

  init(modelUrl) {
    return new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => {
        if (e.data.type === 'ready') {
          console.log('Face detection worker ready, GPU letterbox:', e.data.gpuLetterbox);
          this.worker.onmessage = (ev) => this._onMessage(ev);
          resolve();
        } else if (e.data.type === 'error') {
          reject(new Error(e.data.message));
        }
      };
      this.worker.postMessage({ type: 'init', modelUrl });
    });
  }

  detect(bitmap) {
    return new Promise((resolve) => {
      this.pendingResolve = resolve;
      this.worker.postMessage({ type: 'detect', bitmap }, [bitmap]);
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
      new URL('./face-landmark-worker.js', import.meta.url),
      { type: 'module' }
    );
    this.pendingResolve = null;
    this.worker.onmessage = (e) => this._onMessage(e);
  }

  init(modelUrl, blendshapeUrl) {
    return new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => {
        if (e.data.type === 'ready') {
          console.log('Face landmark worker ready, GPU warp:', e.data.gpuWarp);
          this.worker.onmessage = (ev) => this._onMessage(ev);
          resolve();
        } else if (e.data.type === 'error') {
          reject(new Error(e.data.message));
        }
      };
      this.worker.postMessage({ type: 'init', modelUrl, blendshapeUrl });
    });
  }

  infer(bitmap, rect, vw, vh) {
    return new Promise((resolve) => {
      this.pendingResolve = resolve;
      this.worker.postMessage(
        { type: 'infer', bitmap, rect, vw, vh },
        [bitmap]
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

      let blendshapes = null;
      if (e.data.blendshapes) {
        blendshapes = new Float32Array(e.data.blendshapes);
      }

      resolve({
        landmarks,
        faceFlag: e.data.faceFlag,
        blendshapes,
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

export class FaceTracker {
  constructor() {
    this.detectWorker = new FaceDetectionWorker();
    this.landmarkWorker = new FaceLandmarkWorker();
    this.slots = [
      { index: 0, worker: this.landmarkWorker, active: false, rect: null, landmarks: null },
    ];
    this.ready = false;
    this.running = false;
    this.detecting = false;       // is detection worker currently busy?
    this.pendingDetections = null; // stashed results from last face detect
  }

  async init(onStatus) {
    // Init workers sequentially (WebGPU EP requires it)
    onStatus?.('Loading face detection worker...');
    await this.detectWorker.init(FACE_DETECTOR_URL);

    onStatus?.('Loading face landmark worker...');
    await this.landmarkWorker.init(FACE_LANDMARK_URL, FACE_BLENDSHAPE_URL);

    console.log('All face workers ready -- main thread is pure orchestration');
    this.ready = true;
    onStatus?.('Ready');
  }

  async processFrame(video) {
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
        createImageBitmap(video).then(bitmap => {
          this.detectWorker.detect(bitmap).then(result => {
            this.detecting = false;
            if (result.detections.length > 0) {
              logDetect(`[face detect] ${result.detections.length} detections`);
              this.pendingDetections = result;
            }
          }).catch(() => { this.detecting = false; });
        });
      }

      // Landmark inference for active slot
      const results = await Promise.all(this.slots.map(async (slot) => {
        if (!slot.active) return null;

        const bitmap = await createImageBitmap(video);
        const result = await slot.worker.infer(bitmap, slot.rect, vw, vh);

        if (result.faceFlag > FACE_FLAG_THRESHOLD) {
          slot.landmarks = result.landmarks;
          slot.rect = this.landmarksToRect(result.landmarks, vw, vh);
          return { landmarks: result.landmarks, blendshapes: result.blendshapes };
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
