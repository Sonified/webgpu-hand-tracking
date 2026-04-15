// Hand tracking pipeline: single GPU device via vision-worker.
// Main thread is pure orchestration.

import { detectionToRect } from './nms.js';
const HAND_FLAG_THRESHOLD = 0.5;

function makeLogger(intervalMs = 2000) {
  let lastLog = 0;
  return function(msg, ...args) {
    const now = performance.now();
    if (now - lastLog > intervalMs) { console.log(msg, ...args); lastLog = now; }
  };
}
const logPalm = makeLogger(2000);
const logSlot = makeLogger(2000);
const logLandmark = makeLogger(2000);

// ── Shared vision worker with reqId-based routing ──

let _worker = null;
let _workerReady = false;
let _workerInitPromise = null;
const _pending = new Map(); // reqId -> { resolve }
let _nextId = 0;

function getWorker() {
  if (_worker) return _worker;
  _worker = new Worker(new URL('./vision-worker.js', import.meta.url), { type: 'module' });
  _worker.onmessage = (e) => {
    const { reqId } = e.data;
    if (reqId != null && _pending.has(reqId)) {
      _pending.get(reqId).resolve(e.data);
      _pending.delete(reqId);
    }
  };
  _worker.onerror = (e) => console.error('[vision-worker] error:', e.message);
  return _worker;
}

export async function ensureWorkerReady() {
  if (_workerReady) return;
  if (_workerInitPromise) return _workerInitPromise;
  const w = getWorker();
  _workerInitPromise = new Promise((resolve, reject) => {
    const orig = w.onmessage;
    w.onmessage = (e) => {
      if (e.data.type === 'ready') {
        _workerReady = true;
        w.onmessage = orig;
        resolve();
      } else if (e.data.type === 'error') {
        reject(new Error(e.data.message));
      }
    };
    w.postMessage({ type: 'init' });
  });
  return _workerInitPromise;
}

export function workerCall(msgType, data, transfers) {
  const reqId = _nextId++;
  return new Promise((resolve) => {
    _pending.set(reqId, { resolve });
    getWorker().postMessage({ type: msgType, reqId, ...data }, transfers || []);
  });
}

// ── Pipeline wrappers (same interface as before) ──

class PalmWorker {
  async init() { await ensureWorkerReady(); }
  detect(bitmap) {
    return workerCall('palmDetect', { bitmap }, [bitmap]);
  }
}

class LandmarkWorker {
  async init() { await ensureWorkerReady(); }
  infer(bitmap, rect, vw, vh) {
    return workerCall('handLandmark', { bitmap, rect, vw, vh }, [bitmap]).then(data => {
      let landmarks = [];
      if (data.landmarks) {
        const flat = new Float32Array(data.landmarks);
        for (let i = 0; i < 21; i++) {
          landmarks.push({ x: flat[i*3], y: flat[i*3+1], z: flat[i*3+2] });
        }
      }
      return { landmarks, handFlag: data.handFlag, handedness: data.handedness };
    });
  }
}

export class HandTracker {
  constructor() {
    this.palmWorker = new PalmWorker();
    // Two logical landmark "workers" but they share the same physical worker+device
    this.landmarkWorkers = [new LandmarkWorker(), new LandmarkWorker()];
    this.slots = [
      { index: 0, worker: this.landmarkWorkers[0], active: false, rect: null, landmarks: null },
      { index: 1, worker: this.landmarkWorkers[1], active: false, rect: null, landmarks: null },
    ];
    this.ready = false;
    this.running = false;
    this.palmDetecting = false;
    this.pendingDetections = null;
  }

  async init(onStatus) {
    onStatus?.('Loading vision worker (all models)...');
    await this.palmWorker.init();
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
          const detPx = det.cx * vw, detPy = det.cy * vh;
          const overlapsTracked = this.slots.some(s => {
            if (!s.active) return false;
            const dx = s.rect.cx - detPx, dy = s.rect.cy - detPy;
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

      // Both hand landmarks can fire concurrently -- same device handles them
      // via the GPU command queue (no JS-level serialization needed)
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

      if (this.slots[0].active && this.slots[1].active &&
          results[0]?.handedness === 'Right' && results[1]?.handedness === 'Left') {
        [this.slots[0].rect, this.slots[1].rect] = [this.slots[1].rect, this.slots[0].rect];
        [this.slots[0].landmarks, this.slots[1].landmarks] = [this.slots[1].landmarks, this.slots[0].landmarks];
        [results[0], results[1]] = [results[1], results[0]];
      }

      logLandmark(`[tracking] slots: ${this.slots.map(s => s.active ? (s === this.slots[0] ? 'L' : 'R') : '_').join(',')}`);

      return {
        hands: results.filter(Boolean),
        debug: { rects: this.slots.filter(s => s.rect).map(s => s.rect) },
      };
    } catch (err) {
      console.error('processFrame error:', err.message, err.stack);
      return { hands: [] };
    } finally {
      this.running = false;
    }
  }

  landmarksToRect(landmarks, imgW, imgH) {
    const wrist = landmarks[0], indexMcp = landmarks[5], middleMcp = landmarks[9], ringMcp = landmarks[13];
    const tx = 0.25 * (indexMcp.x + ringMcp.x) + 0.5 * middleMcp.x;
    const ty = 0.25 * (indexMcp.y + ringMcp.y) + 0.5 * middleMcp.y;
    const rotation = Math.PI / 2 - Math.atan2(wrist.y - ty, tx - wrist.x);
    const angle = rotation - 2 * Math.PI * Math.floor((rotation + Math.PI) / (2 * Math.PI));
    const stableIds = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];
    const pts = stableIds.map(i => [landmarks[i].x * imgW, landmarks[i].y * imgH]);
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of pts) { minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); }
    const acx = (minX + maxX) / 2, acy = (minY + maxY) / 2;
    const cos = Math.cos(angle), sin = Math.sin(angle);
    let rMinX = Infinity, rMinY = Infinity, rMaxX = -Infinity, rMaxY = -Infinity;
    for (const [x, y] of pts) { const dx = x - acx, dy = y - acy; const rx = dx * cos + dy * sin; const ry = -dx * sin + dy * cos; rMinX = Math.min(rMinX, rx); rMinY = Math.min(rMinY, ry); rMaxX = Math.max(rMaxX, rx); rMaxY = Math.max(rMaxY, ry); }
    const projCx = (rMinX + rMaxX) / 2, projCy = (rMinY + rMaxY) / 2;
    const cx = cos * projCx - sin * projCy + acx, cy = sin * projCx + cos * projCy + acy;
    const width = rMaxX - rMinX, height = rMaxY - rMinY;
    const size = 2 * Math.max(width, height);
    return { cx: cx + 0.1 * height * sin, cy: cy - 0.1 * height * cos, w: size, h: size, angle };
  }
}
