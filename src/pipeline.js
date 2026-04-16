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

  infer(frame, rect, vw, vh) {
    return new Promise((resolve) => {
      this.pendingResolve = resolve;
      this.worker.postMessage(
        { type: 'infer', frame, rect, vw, vh },
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
      { index: 0, worker: this.landmarkWorkers[0], active: false, rect: null, landmarks: null, lastRect: null },
      { index: 1, worker: this.landmarkWorkers[1], active: false, rect: null, landmarks: null, lastRect: null },
    ];
    this.ready = false;
    this.running = false;
    this.palmDetecting = false;   // is palm worker currently busy?
    this.pendingDetections = null; // stashed results from last palm detect
    this.dupeFrames = 0;          // consecutive frames both slots track the same hand
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
          let overlapsTracked = false;
          for (const s of this.slots) {
            if (!s.active) continue;
            const dx = s.rect.cx - detPx;
            const dy = s.rect.cy - detPy;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const thresh = s.rect.w * 0.5;
            if (dist < thresh) {
              overlapsTracked = true;
              logPalm(`[palm-overlap] det(${detPx.toFixed(0)},${detPy.toFixed(0)}) overlaps slot ${s.index} rect(${s.rect.cx.toFixed(0)},${s.rect.cy.toFixed(0)}) dist=${dist.toFixed(0)} thresh=${thresh.toFixed(0)}`);
              break;
            }
          }
          if (overlapsTracked) continue;

          const rect = detectionToRect(det, vw, vh);
          // Assign to the empty slot whose last-known position is closest.
          // This gives spatial continuity: a returning hand goes back to its slot.
          let bestSlot = emptySlots[0];
          if (emptySlots.length > 1) {
            let bestDist = Infinity;
            for (const s of emptySlots) {
              if (s.lastRect) {
                const dx = s.lastRect.cx - detPx;
                const dy = s.lastRect.cy - detPy;
                const d = dx * dx + dy * dy;
                if (d < bestDist) { bestDist = d; bestSlot = s; }
              }
            }
          }
          emptySlots.splice(emptySlots.indexOf(bestSlot), 1);
          bestSlot.active = true;
          bestSlot.rect = rect;
          logSlot(`[new hand] slot ${bestSlot.index} cx=${rect.cx.toFixed(0)} cy=${rect.cy.toFixed(0)}`);
        }
      }

      // Fire off palm detection async if there are empty slots and we're not already detecting
      const hasEmptySlots = this.slots.some(s => !s.active);
      if (hasEmptySlots && !this.palmDetecting) {
        this.palmDetecting = true;
        const frame = new VideoFrame(video);
        this.palmWorker.detect(frame).then(result => {
          this.palmDetecting = false;
          if (result.detections.length > 0) {
            logPalm(`[palm] ${result.detections.length} detections`);
            this.pendingDetections = result;
          }
        }).catch(() => { this.palmDetecting = false; });
      }

      // Parallel landmark inference via Promise.all
      const results = await Promise.all(this.slots.map(async (slot) => {
        if (!slot.active) return null;

        const frame = new VideoFrame(video);
        const result = await slot.worker.infer(frame, slot.rect, vw, vh);

        if (result.handFlag > HAND_FLAG_THRESHOLD) {
          slot.landmarks = result.landmarks;
          slot.rect = this.landmarksToRect(result.landmarks, vw, vh);
          return { landmarks: result.landmarks, handedness: result.handedness };
        } else {
          slot.lastRect = slot.rect; // remember where this hand was
          slot.active = false;
          slot.landmarks = null;
          return null;
        }
      }));

      // Slots are assigned by screen position (slot 0 = left side, slot 1 = right).
      // No runtime handedness swap -- avoids the visible flip when hands leave/return.

      // Detect double-mapped hands: both slots locked onto the same physical hand.
      // Don't kill either slot (could be prayer hands). Instead, run palm detection
      // in the background looking for the separated hand. When a new detection appears
      // that doesn't overlap either current slot, reassign the duplicate slot to it.
      if (this.slots[0].active && this.slots[1].active &&
          results[0] && results[1] &&
          results[0].handedness === results[1].handedness) {
        const r0 = this.slots[0].rect, r1 = this.slots[1].rect;
        const dx = r0.cx - r1.cx, dy = r0.cy - r1.cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const overlap = dist < Math.max(r0.w, r1.w) * 0.6;
        if (overlap) {
          this.dupeFrames++;
          // After 3 frames of confirmed duplicate, start background palm detection
          // to look for the separated hand (even though both slots are "active")
          if (this.dupeFrames >= 3 && !this.palmDetecting) {
            this.palmDetecting = true;
            const frame = new VideoFrame(video);
            this.palmWorker.detect(frame).then(result => {
              this.palmDetecting = false;
              if (!result.detections.length) return;
              // Find a detection that doesn't overlap either current slot
              for (const det of result.detections) {
                const detX = det.cx * vw, detY = det.cy * vh;
                const d0 = Math.sqrt((detX - this.slots[0].rect.cx) ** 2 + (detY - this.slots[0].rect.cy) ** 2);
                const d1 = Math.sqrt((detX - this.slots[1].rect.cx) ** 2 + (detY - this.slots[1].rect.cy) ** 2);
                const minSep = Math.max(this.slots[0].rect.w, this.slots[1].rect.w) * 0.5;
                if (d0 > minSep || d1 > minSep) {
                  // Found a separated hand -- reassign the duplicate slot (slot 1)
                  const rect = detectionToRect(det, vw, vh);
                  this.slots[1].rect = rect;
                  logSlot(`[dedup] found separated hand, reassigning slot 1 cx=${rect.cx.toFixed(0)} cy=${rect.cy.toFixed(0)}`);
                  this.dupeFrames = 0;
                  break;
                }
              }
            }).catch(() => { this.palmDetecting = false; });
          }
        } else {
          this.dupeFrames = 0;
        }
      } else {
        this.dupeFrames = 0;
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
