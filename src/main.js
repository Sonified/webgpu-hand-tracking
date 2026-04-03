// Entry point: webcam setup, pipeline init, render loop.

import { HandTracker } from './pipeline.js';

const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const fpsEl = document.getElementById('fps');

const tracker = new HandTracker();

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

function drawDebug(debug) {
  // Debug rects hidden -- uncomment to show bounding boxes
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
    const result = await tracker.processFrame(video);
    const dt = performance.now() - t0;

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Always draw debug info (rects + crop preview)
    if (result.debug) {
      drawDebug(result.debug);
    }

    // Draw hand skeletons if landmarks are available
    if (result.hands.length > 0) {
      drawHands(result.hands);
    }

    // FPS + round-trip timing (update ~1/sec)
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime > 1000) {
      const fps = (frameCount / (now - lastFpsTime)) * 1000;
      fpsEl.textContent = `${fps.toFixed(0)} fps | ${dt.toFixed(1)}ms`;
      console.log(`[perf] ${fps.toFixed(0)} fps | ${result.hands.length} hands | processFrame ${dt.toFixed(2)}ms`);
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

    await tracker.init((msg) => {
      console.log('[init]', msg);
      statusEl.textContent = msg;
    });

    console.log('[main] tracker ready, starting loop');
    statusEl.textContent = 'Tracking...';
    loop();
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error('[main] fatal:', err);
  }
}

main();
