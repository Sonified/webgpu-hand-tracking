// Entry point: webcam setup, hand + face tracking, render loop.

import { HandTracker } from './pipeline.js';
import { FaceTracker } from './face-pipeline.js';

const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const fpsEl = document.getElementById('fps');
const trackHandsEl = document.getElementById('trackHands');
const trackFaceEl = document.getElementById('trackFace');

const handTracker = new HandTracker();
const faceTracker = new FaceTracker();
let handReady = false;
let faceReady = false;

// Hand landmark connections for drawing skeleton
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],       // thumb
  [0,5],[5,6],[6,7],[7,8],       // index
  [5,9],[9,10],[10,11],[11,12],  // middle
  [9,13],[13,14],[14,15],[15,16],// ring
  [13,17],[17,18],[18,19],[19,20],// pinky
  [0,17],                        // palm base
];

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

function drawHands(hands) {
  for (const hand of hands) {
    const lm = hand.landmarks;
    if (!lm || lm.length === 0) continue;

    const scaleX = overlay.width;
    const scaleY = overlay.height;

    ctx.strokeStyle = 'rgba(0, 255, 100, 0.8)';
    ctx.lineWidth = 2;
    for (const [a, b] of CONNECTIONS) {
      if (a >= lm.length || b >= lm.length) continue;
      ctx.beginPath();
      ctx.moveTo(lm[a].x * scaleX, lm[a].y * scaleY);
      ctx.lineTo(lm[b].x * scaleX, lm[b].y * scaleY);
      ctx.stroke();
    }

    for (let i = 0; i < lm.length; i++) {
      const x = lm[i].x * scaleX;
      const y = lm[i].y * scaleY;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fillStyle = i === 0 ? '#ff0' : '#0f0';
      ctx.fill();
    }
  }
}

function drawFaces(faces) {
  for (const face of faces) {
    const lm = face.landmarks;
    if (!lm || lm.length === 0) continue;

    const scaleX = overlay.width;
    const scaleY = overlay.height;

    for (let i = 0; i < lm.length; i++) {
      const x = lm[i].x * scaleX;
      const y = lm[i].y * scaleY;
      ctx.beginPath();
      ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
      ctx.fillStyle = i === 1 ? '#ff0' : '#0ff';
      ctx.fill();
    }
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

    // Run active trackers in parallel
    const promises = [];
    if (trackHandsEl.checked && handReady) promises.push(handTracker.processFrame(video));
    if (trackFaceEl.checked && faceReady) promises.push(faceTracker.processFrame(video));

    const results = await Promise.all(promises);
    const dt = performance.now() - t0;

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    let handCount = 0;
    let faceCount = 0;

    for (const result of results) {
      if (result.hands) {
        handCount = result.hands.length;
        if (handCount > 0) drawHands(result.hands);
      }
      if (result.faces) {
        faceCount = result.faces.length;
        if (faceCount > 0) drawFaces(result.faces);
      }
    }

    // FPS + round-trip timing (update ~1/sec)
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime > 1000) {
      const fps = (frameCount / (now - lastFpsTime)) * 1000;
      fpsEl.textContent = `${fps.toFixed(0)} fps | ${dt.toFixed(1)}ms`;
      const parts = [`${fps.toFixed(0)} fps`];
      if (trackHandsEl.checked) parts.push(`${handCount} hands`);
      if (trackFaceEl.checked) parts.push(`${faceCount} faces`);
      parts.push(`${dt.toFixed(2)}ms`);
      console.log(`[perf] ${parts.join(' | ')}`);
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
    console.log('[main] starting');

    statusEl.textContent = 'Requesting camera...';
    await setupCamera();
    console.log('[main] camera ready:', video.videoWidth, 'x', video.videoHeight);

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    // WebGPU in Workers requires cross-origin isolation headers
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

    // Init hand tracker (checked by default)
    statusEl.textContent = 'Loading hand tracker...';
    await handTracker.init((msg) => {
      console.log('[hand init]', msg);
      statusEl.textContent = msg;
    });
    handReady = true;
    console.log('[main] hand tracker ready');

    // Init face tracker in background
    statusEl.textContent = 'Loading face tracker...';
    await faceTracker.init((msg) => {
      console.log('[face init]', msg);
      statusEl.textContent = msg;
    });
    faceReady = true;
    console.log('[main] face tracker ready');

    document.getElementById('controls').style.display = 'flex';
    statusEl.textContent = 'Tracking...';
    loop();
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error('[main] fatal:', err);
  }
}

main();
