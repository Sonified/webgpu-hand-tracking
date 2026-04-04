// Entry point: webcam setup, face pipeline init, render loop.

import { FaceTracker } from './face-pipeline.js';

const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const fpsEl = document.getElementById('fps');

const tracker = new FaceTracker();
const showDebugCheckbox = document.getElementById('showDebug');

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      resolve();
    };
  });
}

function drawFaces(faces) {
  for (const face of faces) {
    const lm = face.landmarks;
    if (!lm || lm.length === 0) continue;

    const scaleX = overlay.width;
    const scaleY = overlay.height;

    // Draw all 478 landmarks as small dots
    for (let i = 0; i < lm.length; i++) {
      const x = lm[i].x * scaleX;
      const y = lm[i].y * scaleY;
      ctx.beginPath();
      ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
      // Nose tip (landmark 1) in yellow, everything else cyan
      ctx.fillStyle = i === 1 ? '#ff0' : '#0ff';
      ctx.fill();
    }
  }
}

function drawDebug(debug) {
  if (!debug.rects) return;
  for (const rect of debug.rects) {
    if (!rect) continue;
    const { cx, cy, w, h, angle } = rect;

    // Draw rotated rect on overlay
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.strokeStyle = 'rgba(255, 255, 0, 0.6)';
    ctx.lineWidth = 2;
    ctx.strokeRect(-w / 2, -h / 2, w, h);
    ctx.restore();

    // Crop preview in bottom-right corner
    const previewSize = 128;
    const px = 8; // draws on left in canvas, appears on right after CSS scaleX(-1)
    const py = overlay.height - previewSize - 8;

    ctx.save();
    ctx.beginPath();
    ctx.rect(px, py, previewSize, previewSize);
    ctx.clip();
    // Counter-rotate so face appears upright (what the model sees)
    ctx.translate(px + previewSize / 2, py + previewSize / 2);
    const scale = previewSize / Math.max(w, h);
    ctx.scale(scale, scale);
    ctx.rotate(-angle);
    ctx.translate(-cx, -cy);
    ctx.drawImage(video, 0, 0);
    ctx.restore();

    ctx.strokeStyle = 'rgba(255, 255, 0, 0.8)';
    ctx.lineWidth = 1;
    ctx.strokeRect(px, py, previewSize, previewSize);
  }
}

// FPS tracking
let frameCount = 0;
let lastFpsTime = performance.now();

async function loop() {
  if (video.readyState < 2) {
    requestAnimationFrame(loop);
    return;
  }

  try {
    const t0 = performance.now();
    const result = await tracker.processFrame(video, {
      runBlendshapes: document.getElementById('enableBlendshapes').checked,
    });
    const dt = performance.now() - t0;

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Draw debug info if checkbox is checked
    if (result.debug && showDebugCheckbox.checked) {
      drawDebug(result.debug);
    }

    // Draw face landmarks if available
    if (result.faces.length > 0) {
      drawFaces(result.faces);
    }

    // FPS + round-trip timing (update ~1/sec)
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime > 1000) {
      const fps = (frameCount / (now - lastFpsTime)) * 1000;
      fpsEl.textContent = `${fps.toFixed(0)} fps | ${dt.toFixed(1)}ms`;
      console.log(`[perf] ${fps.toFixed(0)} fps | ${result.faces.length} faces | processFrame ${dt.toFixed(2)}ms`);
      frameCount = 0;
      lastFpsTime = now;
    }
  } catch (err) {
    console.error('Frame error:', err);
  }

  requestAnimationFrame(loop);
}

async function main() {
  try {
    console.log('[main] starting face tracking');

    statusEl.textContent = 'Requesting camera...';
    await setupCamera();
    console.log('[main] camera ready:', video.videoWidth, 'x', video.videoHeight);

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    if (!crossOriginIsolated) {
      console.warn('[main] Missing COOP/COEP headers -- tracking disabled, camera still works');
      const cmd = 'npm run dev';
      statusEl.innerHTML = `
        <span style="color:#f90">WebGPU requires security headers, run the following command to start the dev server:</span>
        <code style="margin-left:6px">${cmd}</code>
        <button id="copyBtn" style="margin-left:6px; padding:4px 12px; cursor:pointer; font-size:0.8rem; width:60px; height:28px; vertical-align:middle; animation:pulse 2s infinite; background:#222; color:#0f0; border:1px solid #0f0; border-radius:4px; font-family:monospace">Copy</button>
        <style>@keyframes pulse{0%,100%{box-shadow:0 0 4px #0f0}50%{box-shadow:0 0 12px #0f0}}</style>
      `;
      document.getElementById('copyBtn').onclick = () => {
        navigator.clipboard.writeText(cmd);
        const btn = document.getElementById('copyBtn');
        btn.textContent = '\u2713';
        btn.style.fontSize = '1.2rem';
        btn.style.animation = 'none';
        btn.style.boxShadow = '0 0 8px #0f0';
        setTimeout(() => { btn.textContent = 'Copy'; btn.style.fontSize = '0.8rem'; btn.style.animation = 'pulse 2s infinite'; btn.style.boxShadow = ''; }, 1500);
      };
      return;
    }

    await tracker.init((msg) => {
      console.log('[init]', msg);
      statusEl.textContent = msg;
    });

    console.log('[main] face tracker ready, starting loop');
    statusEl.textContent = 'Tracking...';
    loop();
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error('[main] fatal:', err);
  }
}

main();
