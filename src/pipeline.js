// Hand tracking pipeline: ALL inference in workers. Main thread is pure orchestration.
// - Palm detection: dedicated worker with GPU letterbox
// - Landmark inference: two parallel workers with GPU affine warp

import { detectionToRect } from './nms.js';
import { PALM_MODEL_URL, HAND_LANDMARK_URL as LANDMARK_MODEL_URL } from './model-urls.js';
const HAND_FLAG_THRESHOLD = 0.5;

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
const logPalm = makeLogger(2000);
const logSlot = makeLogger(2000);
const logLandmark = makeLogger(2000);

/**
 * Wraps the palm detection worker.
 */
class PalmWorker {
  constructor() {
    this.worker = new Worker(
      new URL('./palm-worker-wgsl.js', import.meta.url),
      { type: 'module' }
    );
    this.pendingResolve = null;
    this.worker.onmessage = (e) => this._onMessage(e);
    this.worker.onerror = (e) => console.error('[PalmWorker] uncaught worker error:', e.message, e);
  }

  init(modelUrl) {
    return new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => {
        if (e.data.type === 'ready') {
          this.worker.onmessage = (ev) => this._onMessage(ev);
          resolve();
        } else if (e.data.type === 'error') {
          console.error('[PalmWorker] reported error:', e.data.message);
          reject(new Error(e.data.message));
        }
      };
      this.worker.onerror = (e) => {
        console.error('[PalmWorker] worker crashed:', e.message, e);
        reject(new Error(`Worker crashed: ${e.message}`));
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
      console.error('Palm worker error:', e.data.message);
      if (this.pendingResolve) {
        this.pendingResolve({ detections: [], letterbox: {} });
        this.pendingResolve = null;
      }
    }
  }
}

/**
 * Wraps a landmark inference worker.
 */
class LandmarkWorker {
  constructor() {
    this.worker = new Worker(
      new URL('./landmark-worker-wgsl.js', import.meta.url),
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
        for (let i = 0; i < 21; i++) {
          landmarks.push({
            x: flat[i * 3],
            y: flat[i * 3 + 1],
            z: flat[i * 3 + 2],
          });
        }
      }

      resolve({
        landmarks,
        handFlag: e.data.handFlag,
        handedness: e.data.handedness,
      });
    } else if (e.data.type === 'error') {
      console.error('Landmark worker error:', e.data.message);
      if (this.pendingResolve) {
        this.pendingResolve({ landmarks: [], handFlag: 0, handedness: null });
        this.pendingResolve = null;
      }
    }
  }
}

export class HandTracker {
  constructor() {
    this.palmWorker = new PalmWorker();
    this.landmarkWorkers = [new LandmarkWorker(), new LandmarkWorker()];
    this.slots = [
      { index: 0, worker: this.landmarkWorkers[0], active: false, rect: null, landmarks: null },
      { index: 1, worker: this.landmarkWorkers[1], active: false, rect: null, landmarks: null },
    ];
    this.ready = false;
    this.running = false;
    this.palmDetecting = false;   // is palm worker currently busy?
    this.pendingDetections = null; // stashed results from last palm detect
  }

  async init(onStatus) {
    // Init all three workers sequentially (WebGPU EP requires it)
    onStatus?.('Loading palm detection worker...');
    await this.palmWorker.init(PALM_MODEL_URL);

    onStatus?.('Loading landmark worker 0...');
    await this.landmarkWorkers[0].init(LANDMARK_MODEL_URL);
    onStatus?.('Loading landmark worker 1...');
    await this.landmarkWorkers[1].init(LANDMARK_MODEL_URL);

    console.log('All workers ready -- main thread is pure orchestration');
    this.ready = true;
    onStatus?.('Ready');
  }

  async processFrame(video) {
    if (!this.ready || this.running) return { hands: [] };
    this.running = true;

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    try {
      // If pending detections arrived from a previous palm detect, assign them now
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

          const rect = detectionToRect(det, vw, vh);
          const slot = emptySlots.shift();
          slot.active = true;
          slot.rect = rect;
          logSlot(`[new hand] slot ${slot.index} cx=${rect.cx.toFixed(0)} cy=${rect.cy.toFixed(0)}`);
        }
      }

      // Fire off palm detection async if there are empty slots and we're not already detecting
      const hasEmptySlots = this.slots.some(s => !s.active);
      if (hasEmptySlots && !this.palmDetecting) {
        this.palmDetecting = true;
        createImageBitmap(video).then(bitmap => {
          this.palmWorker.detect(bitmap).then(result => {
            this.palmDetecting = false;
            if (result.detections.length > 0) {
              logPalm(`[palm] ${result.detections.length} detections`);
              this.pendingDetections = result;
            }
          }).catch(() => { this.palmDetecting = false; });
        });
      }

      // Parallel landmark inference via Promise.all
      const results = await Promise.all(this.slots.map(async (slot) => {
        if (!slot.active) return null;

        const bitmap = await createImageBitmap(video);
        const result = await slot.worker.infer(bitmap, slot.rect, vw, vh);

        if (result.handFlag > HAND_FLAG_THRESHOLD) {
          slot.landmarks = result.landmarks;
          slot.rect = this.landmarksToRect(result.landmarks, vw, vh);
          return { landmarks: result.landmarks, handedness: result.handedness };
        } else {
          slot.active = false;
          slot.landmarks = null;
          return null;
        }
      }));

      // Stable handedness: slot 0 = Left, slot 1 = Right
      // If both active and labels are swapped, swap slot tracking data
      if (this.slots[0].active && this.slots[1].active &&
          results[0]?.handedness === 'Right' && results[1]?.handedness === 'Left') {
        [this.slots[0].rect, this.slots[1].rect] = [this.slots[1].rect, this.slots[0].rect];
        [this.slots[0].landmarks, this.slots[1].landmarks] = [this.slots[1].landmarks, this.slots[0].landmarks];
        [results[0], results[1]] = [results[1], results[0]];
      }

      logLandmark(`[tracking] slots: ${this.slots.map(s => s.active ? (s === this.slots[0] ? 'L' : 'R') : '_').join(',')}`);

      return {
        hands: results.filter(Boolean),
        debug: {
          rects: this.slots.filter(s => s.rect).map(s => s.rect),
        },
      };
    } catch (err) {
      console.error('processFrame error:', err.message, err.stack);
      return { hands: [] };
    } finally {
      this.running = false;
    }
  }

  landmarksToRect(landmarks, imgW, imgH) {
    const wrist = landmarks[0];
    const indexMcp = landmarks[5];
    const middleMcp = landmarks[9];
    const ringMcp = landmarks[13];

    const tx = 0.25 * (indexMcp.x + ringMcp.x) + 0.5 * middleMcp.x;
    const ty = 0.25 * (indexMcp.y + ringMcp.y) + 0.5 * middleMcp.y;

    const rotation = Math.PI / 2 - Math.atan2(wrist.y - ty, tx - wrist.x);
    const angle = rotation - 2 * Math.PI * Math.floor((rotation + Math.PI) / (2 * Math.PI));

    const stableIds = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];
    const pts = stableIds.map(i => [landmarks[i].x * imgW, landmarks[i].y * imgH]);

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of pts) {
      minX = Math.min(minX, x); minY = Math.min(minY, y);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    }
    const acx = (minX + maxX) / 2;
    const acy = (minY + maxY) / 2;

    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    let rMinX = Infinity, rMinY = Infinity, rMaxX = -Infinity, rMaxY = -Infinity;
    for (const [x, y] of pts) {
      const dx = x - acx, dy = y - acy;
      const rx = dx * cos + dy * sin;
      const ry = -dx * sin + dy * cos;
      rMinX = Math.min(rMinX, rx); rMinY = Math.min(rMinY, ry);
      rMaxX = Math.max(rMaxX, rx); rMaxY = Math.max(rMaxY, ry);
    }

    const projCx = (rMinX + rMaxX) / 2;
    const projCy = (rMinY + rMaxY) / 2;
    const cx = cos * projCx - sin * projCy + acx;
    const cy = sin * projCx + cos * projCy + acy;

    const width = rMaxX - rMinX;
    const height = rMaxY - rMinY;
    const size = 2 * Math.max(width, height);

    const shiftCx = cx + 0.1 * height * sin;
    const shiftCy = cy - 0.1 * height * cos;

    return { cx: shiftCx, cy: shiftCy, w: size, h: size, angle };
  }
}
