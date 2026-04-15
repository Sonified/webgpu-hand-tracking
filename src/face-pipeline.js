// Face tracking pipeline: shares vision-worker with hand pipeline (one GPU device).

import { faceDetectionToRect } from './face-nms.js';
import { FACE_BLENDSHAPE_URL } from './model-urls.js';
import { ensureWorkerReady, workerCall } from './pipeline.js';
const FACE_FLAG_THRESHOLD = 0.5;

function makeLogger(intervalMs = 2000) {
  let lastLog = 0;
  return function(msg, ...args) {
    const now = performance.now();
    if (now - lastLog > intervalMs) { console.log(msg, ...args); lastLog = now; }
  };
}
const logDetect = makeLogger(2000);
const logSlot = makeLogger(2000);
const logLandmark = makeLogger(2000);

class FaceDetectionWorker {
  async init() { await ensureWorkerReady(); }
  detect(bitmap) {
    return workerCall('faceDetect', { bitmap }, [bitmap]);
  }
}

class FaceLandmarkWorker {
  async init() { await ensureWorkerReady(); }
  infer(bitmap, rect, vw, vh) {
    return workerCall('faceLandmark', { bitmap, rect, vw, vh }, [bitmap]).then(data => {
      let landmarks = [];
      if (data.landmarks) {
        const flat = new Float32Array(data.landmarks);
        for (let i = 0; i < 478; i++) {
          landmarks.push({ x: flat[i*3], y: flat[i*3+1], z: flat[i*3+2] });
        }
      }
      return { landmarks, faceFlag: data.faceFlag, rawLandmarks: data.rawLandmarks, modelSize: data.modelSize };
    });
  }
}

class BlendshapeWorker {
  constructor() {
    this.worker = new Worker(new URL('./face-blendshape-worker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e) => this._onMessage(e);
    this.lastBlendshapes = null;
  }
  init(modelUrl) {
    return new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => {
        if (e.data.type === 'ready') { this.worker.onmessage = (ev) => this._onMessage(ev); resolve(); }
        else if (e.data.type === 'error') reject(new Error(e.data.message));
      };
      this.worker.postMessage({ type: 'init', modelUrl });
    });
  }
  infer(rawLandmarks, modelSize) {
    this.worker.postMessage({ type: 'infer', rawLandmarks, modelSize }, [rawLandmarks]);
  }
  _onMessage(e) {
    if (e.data.type === 'result') this.lastBlendshapes = new Float32Array(e.data.blendshapes);
    else if (e.data.type === 'error') console.error('Blendshape worker error:', e.data.message);
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
    onStatus?.('Loading face models (shared GPU)...');
    await this.detectWorker.init();
    onStatus?.('Loading blendshape worker...');
    await this.blendshapeWorker.init(FACE_BLENDSHAPE_URL);
    console.log(`All face workers ready (${this.numFaces} face slots) -- main thread is pure orchestration`);
    this.ready = true;
    onStatus?.('Ready');
  }

  async processFrame(video, { runBlendshapes = true } = {}) {
    if (!this.ready || this.running) return { faces: [] };
    this.running = true;
    const vw = video.videoWidth, vh = video.videoHeight;

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
          for (const kp of det.keypoints) { kp.x = (kp.x - letterbox.offsetX) / letterbox.scaleX; kp.y = (kp.y - letterbox.offsetY) / letterbox.scaleY; }
          if (det.cy < -0.1 || det.cy > 1.1 || det.cx < -0.1 || det.cx > 1.1) continue;
          const detPx = det.cx * vw, detPy = det.cy * vh;
          const overlapsTracked = this.slots.some(s => { if (!s.active) return false; const dx = s.rect.cx - detPx, dy = s.rect.cy - detPy; return Math.sqrt(dx*dx + dy*dy) < s.rect.w * 0.5; });
          if (overlapsTracked) continue;
          const rect = faceDetectionToRect(det, vw, vh);
          const slot = emptySlots.shift();
          slot.active = true; slot.rect = rect;
          logSlot(`[new face] slot ${slot.index} cx=${rect.cx.toFixed(0)} cy=${rect.cy.toFixed(0)}`);
        }
      }

      const hasEmptySlots = this.slots.some(s => !s.active);
      if (hasEmptySlots && !this.detecting) {
        this.detecting = true;
        createImageBitmap(video).then(bitmap => {
          this.detectWorker.detect(bitmap).then(result => {
            this.detecting = false;
            if (result.detections.length > 0) { logDetect(`[face detect] ${result.detections.length} detections`); this.pendingDetections = result; }
          }).catch(() => { this.detecting = false; });
        });
      }

      const results = await Promise.all(this.slots.map(async (slot) => {
        if (!slot.active) return null;
        const bitmap = await createImageBitmap(video);
        const result = await slot.worker.infer(bitmap, slot.rect, vw, vh);
        if (result.faceFlag > FACE_FLAG_THRESHOLD) {
          slot.landmarks = result.landmarks;
          slot.rect = this.landmarksToRect(result.landmarks, vw, vh);
          if (runBlendshapes && result.rawLandmarks) this.blendshapeWorker.infer(result.rawLandmarks, result.modelSize);
          return { landmarks: result.landmarks, blendshapes: this.blendshapeWorker.lastBlendshapes };
        } else { slot.active = false; slot.landmarks = null; return null; }
      }));

      logLandmark(`[tracking] slots: ${this.slots.map(s => s.active ? '1' : '0').join(',')}`);
      return { faces: results.filter(Boolean), debug: { rects: this.slots.filter(s => s.rect).map(s => s.rect) } };
    } catch (err) { console.error('processFrame error:', err.message, err.stack); return { faces: [] }; }
    finally { this.running = false; }
  }

  landmarksToRect(landmarks, imgW, imgH) {
    const rightEye = landmarks[33], leftEye = landmarks[263];
    const angle = Math.atan2(leftEye.y - rightEye.y, leftEye.x - rightEye.x);
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const lm of landmarks) { const px = lm.x * imgW, py = lm.y * imgH; minX = Math.min(minX, px); minY = Math.min(minY, py); maxX = Math.max(maxX, px); maxY = Math.max(maxY, py); }
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    const size = Math.max(maxX - minX, maxY - minY) * 1.5;
    return { cx, cy, w: size, h: size, angle };
  }
}
