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
const enableBlendshapesEl = document.getElementById('enableBlendshapes');
const showBlendshapesEl = document.getElementById('showBlendshapes');
const showBsLabel = document.getElementById('showBsLabel');
const numFacesEl = document.getElementById('numFaces');
const numFacesLabel = document.getElementById('numFacesLabel');
const blendshapePanel = document.getElementById('blendshapePanel');

// Restore checkbox/select state from localStorage (before constructors that read them)
if (localStorage.getItem('trackHands') !== null) trackHandsEl.checked = localStorage.getItem('trackHands') === 'true';
if (localStorage.getItem('trackFace') !== null) trackFaceEl.checked = localStorage.getItem('trackFace') === 'true';
if (localStorage.getItem('enableBlendshapes') !== null) enableBlendshapesEl.checked = localStorage.getItem('enableBlendshapes') === 'true';
if (localStorage.getItem('showBlendshapes') !== null) showBlendshapesEl.checked = localStorage.getItem('showBlendshapes') === 'true';
if (localStorage.getItem('numFaces') !== null) numFacesEl.value = localStorage.getItem('numFaces');

const handTracker = new HandTracker();
let faceTracker = new FaceTracker(parseInt(numFacesEl.value) || 1);
let handReady = false;
let faceReady = false;
trackHandsEl.addEventListener('change', () => localStorage.setItem('trackHands', trackHandsEl.checked));
numFacesEl.addEventListener('change', async () => {
  localStorage.setItem('numFaces', numFacesEl.value);
  faceReady = false;
  statusEl.textContent = 'Reinitializing face tracker...';
  faceTracker = new FaceTracker(parseInt(numFacesEl.value) || 1);
  await faceTracker.init((msg) => { statusEl.textContent = msg; });
  faceReady = true;
  statusEl.textContent = 'Tracking...';
});
trackFaceEl.addEventListener('change', () => {
  localStorage.setItem('trackFace', trackFaceEl.checked);
  updateBsDeps();
});
enableBlendshapesEl.addEventListener('change', () => {
  localStorage.setItem('enableBlendshapes', enableBlendshapesEl.checked);
  updateBsDeps();
});
showBlendshapesEl.addEventListener('change', () => {
  localStorage.setItem('showBlendshapes', showBlendshapesEl.checked);
  updateBsDeps();
});

function updateBsDeps() {
  const faceOn = trackFaceEl.checked;
  numFacesEl.disabled = !faceOn;
  enableBlendshapesEl.disabled = !faceOn;
  showBlendshapesEl.disabled = !faceOn || !enableBlendshapesEl.checked;
  if (!faceOn) {
    enableBlendshapesEl.checked = false;
    showBlendshapesEl.checked = false;
    localStorage.setItem('enableBlendshapes', 'false');
    localStorage.setItem('showBlendshapes', 'false');
  }
  if (!enableBlendshapesEl.checked) {
    showBlendshapesEl.checked = false;
    localStorage.setItem('showBlendshapes', 'false');
  }
  showBsLabel.style.display = enableBlendshapesEl.checked ? '' : 'none';
  blendshapePanel.style.display = (showBlendshapesEl.checked && enableBlendshapesEl.checked && faceOn) ? 'block' : 'none';
}
updateBsDeps();

const BLENDSHAPE_NAMES = [
  '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp',
  'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff',
  'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft',
  'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
  'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
  'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight',
  'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
  'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
  'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight',
  'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft',
  'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft',
  'mouthPressRight', 'mouthPucker', 'mouthRight',
  'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
  'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight',
  'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft',
  'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight',
];

let blendshapeBars = [];
function initBlendshapePanel() {
  let html = '';
  for (let i = 1; i < 52; i++) {
    html += `<div style="display:flex; align-items:center; margin-bottom:2px;">
      <span style="width:130px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#888">${BLENDSHAPE_NAMES[i]}</span>
      <div style="flex:1; height:10px; background:#222; border-radius:2px; overflow:hidden;">
        <div id="bs${i}" style="height:100%; width:0%; background:#0ff; transition:width 0.05s;"></div>
      </div>
    </div>`;
  }
  blendshapePanel.innerHTML = html;
  for (let i = 1; i < 52; i++) {
    blendshapeBars.push(document.getElementById(`bs${i}`));
  }
}
initBlendshapePanel();

function updateBlendshapes(blendshapes) {
  if (!blendshapes) return;
  for (let i = 0; i < blendshapeBars.length; i++) {
    const val = Math.min(1, Math.max(0, blendshapes[i + 1]));
    blendshapeBars[i].style.width = `${(val * 100).toFixed(0)}%`;
  }
}

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

    const isLeft = hand.handedness === 'Left';
    const lineColor = isLeft ? 'rgba(0, 255, 100, 0.8)' : 'rgba(100, 150, 255, 0.8)';
    const dotColor = isLeft ? '#0f0' : '#59f';
    const scaleX = overlay.width;
    const scaleY = overlay.height;

    ctx.strokeStyle = lineColor;
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
      ctx.fillStyle = i === 0 ? '#ff0' : dotColor;
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
    if (trackFaceEl.checked && faceReady) promises.push(faceTracker.processFrame(video, {
      runBlendshapes: enableBlendshapesEl.checked,
    }));

    const results = await Promise.all(promises);
    const dt = performance.now() - t0;

    ctx.clearRect(0, 0, overlay.width, overlay.height);
    blendshapePanel.style.display = (showBlendshapesEl.checked && enableBlendshapesEl.checked && trackFaceEl.checked) ? 'block' : 'none';

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
        // Update blendshape panel
        if (enableBlendshapesEl.checked && result.faces.length > 0 && result.faces[0].blendshapes) {
          updateBlendshapes(result.faces[0].blendshapes);
        }
      }
    }

    // FPS + round-trip timing (update ~1/sec)
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime > 1000) {
      const fps = (frameCount / (now - lastFpsTime)) * 1000;
      fpsEl.textContent = `${fps.toFixed(0)} fps | ${dt.toFixed(1)}ms`;
      const active = [
        trackHandsEl.checked ? 'H' : '',
        trackFaceEl.checked ? 'F' : '',
        enableBlendshapesEl.checked ? 'B' : '',
      ].filter(Boolean).join('') || 'none';
      console.log(JSON.stringify({
        engine: 'WGPU', active, fps: +fps.toFixed(0), ms: +dt.toFixed(2),
        hands: handCount, faces: faceCount,
      }));
      frameCount = 0;
      lastFpsTime = now;
    }
  } catch (err) {
    console.error('Frame error:', err);
  }

  if (!document.hidden) requestAnimationFrame(loop);
}

// Pause completely when tab is hidden, resume when visible
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    frameCount = 0;
    lastFpsTime = performance.now();
    requestAnimationFrame(loop);
  }
});

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
