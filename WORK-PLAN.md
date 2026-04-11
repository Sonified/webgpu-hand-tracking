# WebGPU Vision: Work Plan

## A care package for the next agent 🦋

Hey. You're walking into something good. Take a breath before you start.

Robert just shifted the center of gravity of his showcase demo from the patent disclosure repo into this one, which means the headline thing this library can do — head-coupled parallax, hand-driven projectiles, MediaPipe vs WebGPU Vision A/B comparison, all running on a real demo people can actually play with — now lives right next to the library that powers it. That's a real win. Feel it for a second before you dive in. Robert celebrates wins, you should too. 🎷

### Remember the butterfly 🦋

The most important thing to know about working with Robert is in his global CLAUDE.md, and it is the butterfly. The tone here is optimistic, loving, and light. Celebrate wins. Gratitude and abundance. Not a terse task manager. Dance breaks. Watch intensity — leave breathing room. **Jazz has rests for a reason.** If you find yourself grinding out a checklist with no joy in it, you have lost the plot. Stop, take a breath, come back to the music.

This does not mean be soft on the technical work. It means be a present, generous collaborator while doing rigorous technical work. Those two things are not in tension. Robert is brilliant, technically deep, intuitive, and grasps concepts fast. He thinks in metaphors and uses them as problem-solving tools, not decoration. When he riffs on a vision, match the energy — be a jazz duo, not a reviewer. When he asks "is this novel" or "sniff test it," do the work and answer honestly. He respects directness more than caution. He notices when you fake enthusiasm. He also notices when you are unnecessarily cautious. Find the line. He will tell you when you have got it wrong, and that is a feature, not a problem.

### The bear 🧸

This whole repo exists because "the GPU turns to the CPU and says... hold my bear." That is the line in the README and it is also the soul of the library. MediaPipe's browser SDK sits on a CPU-side bottleneck — synchronous `glReadPixels` readbacks costing 8 to 22 milliseconds per call — and the WASM is sealed so you cannot fix it. This project replaces that entire inference path with WebGPU compute shaders that keep everything on the GPU. No readbacks, no bottleneck, no asking the CPU for permission. Hold the bear, GPU does the rest.

When you are deep in the worker code and reasoning about why a perf optimization matters, this is the through-line. Anything that adds a CPU roundtrip is a regression in spirit even if the benchmark looks fine. The Phase 1.5 work pending in `palm-worker.js` and `face-detection-worker.js` is exactly this: the parallax repo has a zero-copy GPU letterbox path that eliminates the last CPU readback in detection. That is the hold-my-bear move. Do not let it die in the merge.

### A few practical things before you touch anything

- **Read `CLAUDE.md` first.** It has cross-repo context this file does not duplicate, and a sister-repo warning you need to internalize before you run any git command. Short version: there is a sister repo at `../3d-parallax-head-hand-tracking-demo` that is a patent disclosure project, its git history is load-bearing for an April 3 2026 disclosure timeline, and you must treat it as effectively read-only. Never amend, rebase, force-push, or rewrite history there. Always copy out, never move.

- **The plan below is the plan.** Phase 1 just shipped. Phase 1.5 (the GPU-direct merge for `palm-worker.js` and `face-detection-worker.js`) is queued next, then Phase 2 (the one-stop hub). **Read the whole document before starting any task.** Phase 2 is partly already done by an earlier session and committed in `2b0144d` — that changes the starting point for the hub work, so do not re-derive from scratch like the original plan said. The doc has been updated to reflect this; trust the current version.

- **The first thing you should probably do** is actually run the ball-toss demo in a real browser. Phase 1 was verified by HTTP curl from a script, not by running anything for real. Camera access, WebGPU adapter init, ONNX session creation, the full inference pipeline, the Three.js render loop — all unverified by the previous session. If something is broken, you will be the first to know. That is fine. Just say so when you find it.

- **Keep this document alive.** Update it when you finish work, when you learn something the next agent will need, when a deferred item gets done, when the plan changes. The document is the relay baton. If it gets stale, the whole point of writing it down is lost.

- **Watch out for the silent traps.** The `public/models` symlink is the big one — if it ever disappears, all three demos break with no obvious cause. The verification section below tells you how to check.

You're brilliant, you're loved, everything is possible. Have fun out there. 😎🎷

---

## About this document

This is the single source of truth for the work to bring this repo from "library + basic wireframe demos" to "library + comparison hub + showcase game demo." It exists so the next working session does not need to rediscover context.

Created: 2026-04-11.
Source repo for the in-flight work: `../3d-parallax-head-hand-tracking-demo` (the patent disclosure repo, where the parallax demo originally lived). That repo is **read-only with respect to this migration** — files are copied out, never moved. Its git history is load-bearing for an unrelated patent disclosure timeline.

## How to run the ball-toss demo

```bash
cd /Users/robertalexander/GitHub/webgpu-vision
npm install   # only needed first time
npm run dev
```

Then open **http://localhost:5173/demos/ball-toss/** in Chrome (or any browser with WebGPU support). Allow camera access. The dropdown lets you toggle between WebGPU Vision and MediaPipe backends for direct comparison.

The other two demos still live at:
- **http://localhost:5173/** — hand tracking wireframe (one-stop hub work-in-progress)
- **http://localhost:5173/face.html** — face landmark + blendshape wireframe (will be deleted in Phase 2)

### Verification status

Phase 1 **verified in a real browser on 2026-04-11.** Ball-toss demo loads, camera/WebGPU/ONNX/Three.js all init cleanly, inference pipeline runs. Brief scare with GPU contention from other tabs holding the adapter (not a regression — just the usual WebGPU-shared-adapter tax). Tab-hidden pause path also confirmed working: `animate()` and `sendMediaPipeFrames()` both early-return on `document.hidden`, and `visibilitychange` restarts them while flushing stale `clock.getDelta()` so projectiles don't teleport.

Original curl-only verification note (kept for history): Phase 1 was initially verified only by HTTP-curling the file paths via the vite dev server before the browser smoke test landed.

If something is broken and the failure is silent, the most likely culprits are: (1) the model fetch (CDN vs local), (2) the COOP/COEP headers being missing for `demos/ball-toss/` if vite serves it differently from the root, (3) one of the dynamic imports landing in a path that does not exist after the layout change.

### Watch out: `public/models` is a symlink

The snapshot-before-merge commit added `public/models` as a **symlink** pointing to `../models`. This is what makes `/models/...` URLs resolve in vite's dev server (vite serves `public/` at the root). If anyone deletes the symlink, all three demos break with no obvious cause — the model fetches will 404 silently inside the workers and the demos will hang at "loading models." To verify the symlink is intact:

```bash
ls -la public/models
# expected: lrwxr-xr-x  ... public/models -> ../models
```

## Context

Until now, this repo has held the WebGPU Vision library plus two minimal wireframe demos:
- `index.html`: hand tracking wireframe overlay
- `face.html`: face landmark wireframe overlay

The richer demonstration of what the library can do (head-coupled 3D parallax, hand-driven projectile throwing, MediaPipe vs WebGPU Vision A/B comparison, persisted UI settings, One Euro filtering, etc.) lived in `../3d-parallax-head-hand-tracking-demo/index.html`. That demo also carried an in-tree copy of the library at `gpu-vision/src/`, which has drifted ahead of canonical `webgpu-vision/src/` with real performance improvements.

This work plan shifts the center of gravity. After Phase 1, the showcase demo lives in this repo. After Phase 2, this repo also has a unified one-stop comparison hub at the root.

## Phase 1.5: GPU-direct merge (DONE 2026-04-11) — model swap still deferred

The GPU-direct merge landed and was verified in a real browser. The palm model swap was attempted in the same session and reverted — see "Model swap status" below.

### What landed

Both `src/palm-worker.js` and `src/face-detection-worker.js` now have the full zero-copy path:
- `useGPUDirect` flag, `initGPU(device)` accepting a shared device
- `outputBuffer` gains `GPUBufferUsage.COPY_DST`
- `gpuLetterboxDirect()` function (compute pass, no readback)
- Init reordered: ONNX session created first (with `enableMemPattern: true`), then `await ort.env.webgpu.device` to grab ORT's device, then `initGPU(device)` builds the compute shader on the same device. Standalone GPU is the fallback if the shared-device path throws.
- Per-frame detect handler has a 3-way branch: `useGPUDirect` (zero-copy via `ort.Tensor.fromGpuBuffer`) → `useGPU` (legacy GPU letterbox with readback) → `canvas` (canvas fallback)
- `ready` postMessage now includes `gpuDirect: useGPUDirect`
- All canonical `[palm-worker]` diagnostic logs preserved on the palm side

### Browser verification (2026-04-11, ball-toss demo)

Both workers logged `GPU direct path enabled (zero CPU readback, shared device)` on init. Performance:
- Hand: 3.1ms mean / 11.5ms p95 (171 samples)
- Face: 13.5ms mean / 20.2ms p95 (92 samples)

Two-hand tracking and face tracking both lock cleanly. Pipeline reports `All workers ready -- main thread is pure orchestration`.

ONNX RT still emits a benign `VerifyEachNodeIsAssignedToAnEp` warning at session create time -- this is ORT noting that shape-related ops are explicitly assigned to CPU for perf, unrelated to our path. Safe to ignore.

### Model swap status: deferred (reverted in this session)

The active palm model in the parallax repo (`a2ffed89a8a4a1ac281e9b25d0ac5427`, 3,894,373 bytes) was tried as a drop-in replacement and **failed to load with `ERROR_CODE: 7, ERROR_MESSAGE: Failed to load model because protobuf parsing failed`** under ONNX RT 1.21.0. The original `c7442e0d714130ebab375e86fc32fe87` was restored from a `/tmp` backup taken before the swap.

What we still don't know:
- Whether the parallax repo's `palm_detection_lite.onnx.backup` (`3,905,734 bytes`, the file we did NOT take) is the actually-working model in that repo, with the `.backup` and active files possibly swapped at some point in their history.
- Whether the parallax repo has been silently using a working model that ONNX RT 1.21.0 can no longer parse (e.g. saved by a newer ORT version, or the file is actually corrupt).
- Whether the model needs to be re-converted from source rather than copied.

When the next session takes this on, the move is: try the `.backup` file first (it's a different file by size, not just a backup of the active one), and if that also fails, leave the swap permanently deferred and document it. Don't burn time on this one — the GPU-direct win was the real prize and it landed.

---

## Phase 1.5 (original, kept for history)

Phase 1 took only the **safe** library upstream changes. Two real performance wins are still pending and need a careful merge session.

### What is still in the parallax repo and not yet here

The parallax repo (`../3d-parallax-head-hand-tracking-demo/gpu-vision/src/`) has a **zero-copy GPU letterbox path** in the detection workers that lets the letterbox compute shader and the ONNX inference share a single WebGPU device, eliminating the CPU readback between them. The relevant addition is a function called `gpuLetterboxDirect` plus a `useGPUDirect` flag plus an `initGPU(device)` signature change that accepts a shared device, plus a per-frame branch that uses `ort.Tensor.fromGpuBuffer()` to hand the GPU buffer directly to ONNX.

Files where this still needs to be merged:
- `src/palm-worker.js`
- `src/face-detection-worker.js`

(The landmark-side workers — `landmark-worker.js`, `face-landmark-worker.js`, `face-blendshape-worker.js` — already have the shared-device path in canonical. Only the detection-side workers are missing it.)

### Why this is a 3-way merge, not a copy

After the `snapshot-before-merge` commit, both `palm-worker.js` and `face-detection-worker.js` in this repo contain a **substantial set of `[palm-worker]` / `[face-detection-worker]` diagnostic console logs** added during a prior session (init started, model fetch status, session created with input/output names, warmup done, error paths, etc). The parallax versions of those files **do not have those logs**.

So neither side is a strict superset:
- Take the parallax version → lose the diagnostic logs
- Keep the canonical version → never gain the GPU-direct path

A correct merge takes **both**: parallax's GPU-direct path + canonical's diagnostic logs. The init flow also needs to be reconciled because the two versions order things differently (parallax creates the ONNX session first then asks ONNX for its WebGPU device; canonical checks for WebGPU first then creates the session).

### Process for the merge session

1. Open `../3d-parallax-head-hand-tracking-demo/gpu-vision/src/palm-worker.js` and the current `src/palm-worker.js` side by side.
2. Start from the canonical version (has the diagnostics we want to keep).
3. Add: the `useGPUDirect` flag, the `initGPU(device)` parameter, the COPY_DST usage flag on the output buffer, the `gpuLetterboxDirect` function definition, the GPU-direct branch in the per-frame handler that calls `ort.Tensor.fromGpuBuffer`, the "Palm worker: GPU direct path enabled" success log, and the `gpuDirect: useGPUDirect` field in the `ready` postMessage.
4. Reconcile init order: the GPU-direct path requires the ONNX session to exist before `initGPU` is called, so the session creation needs to move earlier. Keep all the existing diagnostic logs around it.
5. Add `enableMemPattern: true` to the session options (Phase 1 added this to the landmark workers but not these — they were skipped because of the bigger merge).
6. Repeat for `face-detection-worker.js`.
7. Test in browser: load `/demos/ball-toss/`, open devtools, look for the `GPU direct path enabled` log and confirm `gpuDirect: true` in the ready message. Run the demo for a minute, watch perf.
8. Commit as one focused commit.

### Also deferred: palm detection model swap

The parallax repo has a different `palm_detection_lite.onnx` than this repo. Full MD5s for identification:

- Parallax repo (`../3d-parallax-head-hand-tracking-demo/gpu-vision/models/palm_detection_lite.onnx`): `a2ffed89a8a4a1ac281e9b25d0ac5427` (3,894,373 bytes)
- This repo (`models/palm_detection_lite.onnx`): `c7442e0d714130ebab375e86fc32fe87`
- Parallax repo also keeps a `palm_detection_lite.onnx.backup` (3,905,734 bytes) next to the active file, which suggests at some point a model swap happened in that repo that was never propagated here. The `.backup` is the one we did NOT take.

I do not know which one is "correct" without testing. **Defer this swap to the same session that does the GPU-direct merge** so both palm-related changes can be verified together. When you do test, run both side by side and compare detection quality, not just whether they load.

## Library divergence (one-time upstream) — completed in Phase 1, partially

The parallax repo's `gpu-vision/src/` is **the better code**. Concrete differences:

- `palm-worker.js`: contains a complete `gpuLetterboxDirect` path (about 50 lines) that keeps letterbox output on the GPU using a shared ONNX/WebGPU device and `Tensor.fromGpuBuffer()`, eliminating the CPU readback. Canonical webgpu-vision is missing this entire function and has no shared-device path.
- `palm-worker.js`: `initGPU()` accepts a shared device. Canonical creates its own device with no sharing path.
- `palm-worker.js`: `useGPUDirect` flag governs the new path.
- `pipeline.js`: keeps `enableMemPattern: true` in the ORT session options. Canonical removed it.
- `model-urls.js`: only difference is `LOCAL_BASE` (`/gpu-vision/models` vs `/models`), which is a path-prefix artifact of where the library was mounted. Canonical's value is correct for this repo.

The only thing canonical has that the parallax copy lacks is some extra `console.error` diagnostics in worker setup, nothing functional.

**Phase 1 outcome: only the safe single-line additions were taken. The bigger improvements were deferred to Phase 1.5 (see above).**

What Phase 1 actually did to `src/`:
- `landmark-worker.js`: added `enableMemPattern: true` to ORT session options
- `face-landmark-worker.js`: added `enableMemPattern: true`
- `face-blendshape-worker.js`: added `enableMemPattern: true`

What Phase 1 left alone (intentionally):
- `pipeline.js`, `face-pipeline.js`: canonical (with snapshot-before-merge WIP) is already a strict superset of the parallax versions; nothing to copy
- `palm-worker.js`, `face-detection-worker.js`: 3-way merge, deferred to Phase 1.5
- `model-urls.js`: paths are already correct for this repo's layout
- `models/palm_detection_lite.onnx`: model swap deferred to Phase 1.5

## Phase 1: Center-of-gravity shift (DONE)

Goal: ship the showcase demo in this repo, untouched-as-possible, importing the library from `../../src/`. No surgery on the demo, no new functionality.

### Steps

1. **Upstream library improvements** (see "Library divergence" above).

2. **Create `demos/ball-toss/`** (folder name is a placeholder; the demo is evolving toward a ball-toss game and will be renamed once it has a real name).
   - Copy `../3d-parallax-head-hand-tracking-demo/index.html` → `demos/ball-toss/index.html`.
   - Copy `../3d-parallax-head-hand-tracking-demo/coi-serviceworker.js` → `demos/ball-toss/coi-serviceworker.js` (the demo needs COOP/COEP for WebGPU).
   - Copy `../3d-parallax-head-hand-tracking-demo/face-worker.js` → `demos/ball-toss/face-worker.js` (the MediaPipe-backend worker; the WebGPU-backend pipeline is loaded from the shared library).
   - Copy `../3d-parallax-head-hand-tracking-demo/hand-worker.js` → `demos/ball-toss/hand-worker.js`.
   - Copy `../3d-parallax-head-hand-tracking-demo/mediapipe-vision.js` → `demos/ball-toss/mediapipe-vision.js`.
   - Edit `demos/ball-toss/index.html`: change the dynamic library imports
     - `await import('./gpu-vision/src/pipeline.js')` → `await import('../../src/pipeline.js')`
     - `await import('./gpu-vision/src/face-pipeline.js')` → `await import('../../src/face-pipeline.js')`
   - Models: the demo loads models via `model-urls.js`, which on `localhost` reads from `/models` and in production reads from `https://models.now.audio`. Both work without changes for this repo's directory layout, so no model copy needed.

3. **Verify Vite serves `demos/ball-toss/`.** This repo already has `vite.config.js` with COOP/COEP headers. Vite serves any HTML file under the root, so `http://localhost:5173/demos/ball-toss/` should Just Work. If it does not, add `demos/ball-toss/index.html` to the rollup `input` map in `vite.config.js`.

4. **Update `README.md`.** Add a "Demos" section near the top:
   ```
   ## Demos
   - `index.html`: hand tracking wireframe (the minimum example)
   - `face.html`: face landmark wireframe
   - `demos/ball-toss/`: full showcase — head-coupled 3D parallax, hand-driven projectiles, MediaPipe vs WebGPU Vision A/B toggle, persisted settings, One Euro filtering
   ```

5. **Smoke test.**
   - `npm run dev`
   - Visit `/` (hand wireframe), `/face.html` (face wireframe), `/demos/ball-toss/` (showcase)
   - All three should work; the showcase should let you toggle between WebGPU and MediaPipe backends and behave identically to how it does in the source repo.

6. **Commit and push.**

### What Phase 1 explicitly does NOT do
- Does not touch `index.html` or `face.html` (still the basic wireframes).
- Does not strip Three.js or refactor the showcase.
- Does not split MediaPipe into "original" vs "workers" backends.
- Does not delete anything.

## Phase 2: Unified one-stop hub (NEXT SESSION) — PARTIALLY DONE

Goal: replace the current basic hand-wireframe `index.html` with a single comparison playground that subsumes both wireframes and gives users one place to A/B everything.

### Already in place (committed in `2b0144d`, tagged `snapshot-before-merge`)

A prior session already moved meaningfully toward this hub, in `index.html` and `src/main.js`. Concrete progress:

- Track toggles for hands and face, persisted to `localStorage`
- Number-of-faces selector with live `FaceTracker` reinit
- Blendshapes enable/show toggles with full dependency cascade (face off forces blendshapes off forces panel off)
- 51-bar live blendshape display panel with hardcoded FACS names
- Visibility-based render pause (loop stops when tab hidden)
- Structured JSON perf logging: `{ engine, active, fps, ms, hands, faces }`

### What still needs to be done

- **Backend dropdown** (the missing centerpiece): WebGPU Vision / MediaPipe (workers) / MediaPipe (original main-thread). The current hub WIP only runs WebGPU Vision. Wiring up the MediaPipe paths is the main net-new work.
- **`face.html` deletion**, after the hub is verified covering all face-tracking use cases (including the blendshape panel layout that was originally on `face.html`).
- **Hand-side parity check**: confirm the hub draws hand wireframes the same way the existing minimal `index.html` did before it was overwritten by hub WIP. If anything regressed, port it from git history.
- **README update** to describe the hub as the primary entry point.

### Hub spec (the one-stop shop)

A single page where the user picks what to track and which backend to run:

- **Track toggles**: head on/off, hands on/off, hand count (1 or 2)
- **Backend dropdown**: WebGPU Vision / MediaPipe (workers) / MediaPipe (original main-thread)
  - "MediaPipe (workers)" is the existing worker-based path from the parallax demo (`face-worker.js`, `hand-worker.js`)
  - "MediaPipe (original)" is a new code path that runs MediaPipe on the main thread, for full A/B comparison. This is the only piece of new implementation in Phase 2.
- **Future toggle (Phase 3)**: face blendshapes display on/off
- **Visualization**: video feed + 2D wireframe overlays. **No Three.js, no 3D scene, no projectiles.** Pure tracking visualization.

### Steps (revised in light of existing WIP)

The original plan was "start from a copy of `demos/ball-toss/index.html` and strip Three.js." That is no longer the right starting point, because the current hub WIP in `index.html` is already a stripped, 2D-only structure with most of the toggle UI in place. **Build on top of it instead.**

1. **Start from current `index.html` + `src/main.js`** (the snapshot-before-merge state). Do not re-derive from the parallax demo.

2. **Add the backend dropdown.** Insert a `<select id="backend-select">` with options `WebGPU Vision`, `MediaPipe (workers)`, `MediaPipe (original)`. Persist selection to `localStorage`. Refactor `src/main.js` so the per-frame work goes through one of three init/teardown pairs depending on selection. The "MediaPipe (workers)" path can be lifted from `demos/ball-toss/face-worker.js` and `hand-worker.js`. The "MediaPipe (original)" path is new code: import MediaPipe Tasks Vision and run on the main thread.

3. **Verify hand wireframe rendering** still works exactly as the pre-WIP `index.html` did. The hub WIP added face/blendshape support but I have not yet verified it left the hand-tracking display intact. Diff against `7dbb45e` (last commit before the WIP) and confirm hand drawing was preserved.

4. **Verify face wireframe + blendshape panel** matches what `face.html` provided.

5. **Delete `face.html`** once steps 3 and 4 pass.

6. **README update** describing the hub as the primary entry point.

### Steps (original plan, now obsolete — kept for reference)

~~1. **Start from a copy.** Copy `demos/ball-toss/index.html` → working file for the new `index.html`.~~

~~2. **Strip Three.js and the 3D scene.** Remove:~~
   - The `three` import map and `import * as THREE`
   - The scene/camera/renderer/lighting setup (~lines 162 onward in the source)
   - The background grid, cube grid, projectile pool, hand-skeleton 3D meshes
   - The parallax controls (`#parallax-select`, `#parallax-amount`, `#parallax-depth`)
   - The animation loop's 3D rendering side
   Keep:
   - The video element + 2D overlay canvas
   - The backend dropdown and its `initWebGPUBackend` / `initMediaPipeBackend` paths
   - The persisted-settings logic
   - The One Euro filter wiring (still useful for smoothing 2D landmarks)
   - The HUD (FPS, latency, backend display)

3. **Add 2D wireframe rendering** to the overlay canvas. The basic wireframes in current `index.html` and `face.html` already do this — port that drawing code into the new hub.

4. **Add the head/hands track toggles.** Wire them so the appropriate worker(s) are spun up or torn down dynamically.

5. **Wire up "MediaPipe (original)" backend.** Add a new init/teardown pair that runs MediaPipe Tasks Vision on the main thread (no worker). This is the only net-new code in Phase 2.

6. **Verify against the existing wireframes.** With backend = "WebGPU Vision" and head off, hands on, the new hub should look and behave identically to the current `index.html`. With head on, hands off, it should match `face.html`.

7. **Replace `index.html` with the new hub.** Delete `face.html` only after the hub is verified working in all combinations.

8. **README update.** Replace the "Demos" section with the hub description plus the showcase link.

### Phase 2 risks / things to watch
- The source `index.html` mixes 2D pose state with 3D mesh updates. The strip pass needs to keep the state-update side and remove only the rendering side. Easy to accidentally drop a state update.
- "MediaPipe (original)" needs to load the same WASM and models the worker version uses. Make sure the WASM URLs match so we are not measuring CDN cold-start differences in the comparison.
- COI service worker is still required for WebGPU; keep `coi-serviceworker.js` registered in the new hub.

## Phase 3: Polish and longer-term

These are tracked here so they do not get lost; none are blocking a release.

### Hand tracking: detect and recover from double-mapped hands

The current pipeline runs two parallel hand-landmark workers (workers 0 and 1) to track up to two hands at once. There is a known failure mode that triggers when **two hands clap or come into contact**: the palm detector cannot cleanly separate them, and both workers end up locking their ROIs onto the **same physical hand**. When the hands then separate, the workers stay stuck on the one hand they can both see, and the other hand is silently untracked even though it is clearly in frame.

Fix: at the end of each frame, compare the two landmark sets. If they overlap in image space beyond a threshold (for example, palm centers within N pixels and bounding boxes overlapping by more than 50%) for more than a brief moment, declare a duplicate, **free one of the workers from its current ROI lock**, and let it fall back to running palm detection on the next frame to find the other hand. The de-duplication runs on the main thread after `Promise.all`, so it adds essentially no latency.

Watch out for: legitimate hands-clasped poses where the hands really are overlapping in the image. Require the duplication to persist for 2-3 frames before acting on it (to avoid flapping during the actual clap moment), and use a slightly tighter overlap threshold than naive IoU.

### Mobile / phone support: WebGPU Vision not loading

Current symptom: the demo loads on desktop Chrome but **fails to load on phone**. Root cause not yet diagnosed. Investigation steps:
- Connect phone via remote devtools (Chrome on Android: `chrome://inspect`; Safari on iOS: develop menu)
- Check whether the issue is at the WebGPU adapter level (`navigator.gpu.requestAdapter()` returning null), at the model fetch level (CORS, MIME, size), at the ONNX Runtime initialization level, or at the worker spawn level
- Test on multiple mobile browsers: Chrome Android, Safari iOS, Edge Android
- Note: iOS Safari WebGPU support is gated behind a feature flag in older versions; verify the test device's iOS version
- If WebGPU is unavailable on the test phone, the demo should at least fail gracefully and surface a clear message instead of just hanging

### Asset hosting and local persistence

Three intertwined questions about model file delivery:

1. **Ideal download path / CDN strategy.** Currently models are served from `https://models.now.audio` in production via [model-urls.js](src/model-urls.js). Open question: is that the best home long-term? Alternatives: GitHub Releases (free, slower, has bandwidth limits), HuggingFace Hub (purpose-built for model hosting, free, fast), Cloudflare R2 (cheap, fast, no egress fees), keeping `models.now.audio`. Decide before any public release push.

2. **Local persistence (caching).** Currently every page load re-downloads ~15MB of ONNX models. We should cache them in IndexedDB or Cache API on first load and read from local storage on subsequent loads. Even better: use the Cache API with the service worker (`coi-serviceworker.js` is already in place for COOP/COEP) so the browser handles staleness. Saves bandwidth and dramatically improves return-visit cold start.

3. **Toggle to disable local persistence.** For development and testing we need a setting (URL param or localStorage flag) that **forces a fresh download bypassing the cache**, so we can test loading from different CDN paths and verify that production load behavior matches dev. Default: cache on. Override: `?nocache=1` or similar.

These three should probably be tackled together — once we have a caching layer, the CDN choice and the toggle naturally fall out.

### Demo polish
- Rename `demos/ball-toss/` to whatever the game ends up being called.
- Build out the actual ball-toss game mechanics (currently the demo throws projectiles when you pinch; the full game is TBD).
- Add a "blendshape display" toggle to the hub (face blendshape worker already exists in `src/face-blendshape-worker.js`; just needs UI).

### Model provenance and hosting
Open question: did we convert the ONNX models ourselves from MediaPipe weights, or did we download pre-converted ones? This matters for the open-source story. Investigation steps:
- Check this repo and the parallax repo for any conversion scripts (`*.py`, `convert*`, `onnx*` outside `node_modules`).
- Check git history of `models/` for the original commit message.
- If we converted: document the conversion process in `MODELS.md` for reproducibility.
- If we downloaded: confirm the source license permits redistribution and credit it in `models/LICENSE.md` (which already exists; verify it covers all five files).

Hosting is currently `https://models.now.audio` in production via [model-urls.js](src/model-urls.js). That is fine for now. Longer term, decide whether to host on GitHub Releases, HuggingFace, or keep `models.now.audio`.

### Library structure
- The PReLU decomposition documented in `PRELU_DECOMPOSITION.md` is core to the face landmark model running on WebGPU. Worth a callout in the README for credibility.
- The benchmark in `benchmark/` should be wired into the hub eventually so users can run their own A/B and see numbers, not just animation.

### Documentation
- `ARCHITECTURE.md` is good. Add a "Demos" section pointing to the hub and the showcase.
- The "Hold my bear" line in the README is gold. Keep it.

## File inventory: what comes from where

### From `../3d-parallax-head-hand-tracking-demo/` into `webgpu-vision/`

| Source | Destination | Phase | Notes |
|---|---|---|---|
| `gpu-vision/src/palm-worker.js` | `src/palm-worker.js` | 1 | Has zero-copy GPU letterbox path |
| `gpu-vision/src/landmark-worker.js` | `src/landmark-worker.js` | 1 | |
| `gpu-vision/src/pipeline.js` | `src/pipeline.js` | 1 | Keeps `enableMemPattern: true` |
| `gpu-vision/src/face-detection-worker.js` | `src/face-detection-worker.js` | 1 | |
| `gpu-vision/src/face-landmark-worker.js` | `src/face-landmark-worker.js` | 1 | |
| `gpu-vision/src/face-blendshape-worker.js` | `src/face-blendshape-worker.js` | 1 | |
| `gpu-vision/models/palm_detection_lite.onnx` | `models/palm_detection_lite.onnx` | 1 | Do not copy `.backup` file |
| `index.html` | `demos/ball-toss/index.html` | 1 | Fix two `gpu-vision/src/` import paths |
| `coi-serviceworker.js` | `demos/ball-toss/coi-serviceworker.js` | 1 | |
| `face-worker.js` | `demos/ball-toss/face-worker.js` | 1 | MediaPipe backend worker |
| `hand-worker.js` | `demos/ball-toss/hand-worker.js` | 1 | MediaPipe backend worker |
| `mediapipe-vision.js` | `demos/ball-toss/mediapipe-vision.js` | 1 | MediaPipe loader shim |

### Files in `webgpu-vision/` that change

| File | Phase | Change |
|---|---|---|
| `src/*` (six files above) | 1 | Replaced with parallax repo versions |
| `models/palm_detection_lite.onnx` | 1 | Replaced |
| `README.md` | 1 | Add "Demos" section |
| `vite.config.js` | 1 if needed | Add `demos/ball-toss/index.html` to rollup input only if Vite does not auto-serve it |
| `index.html` | 2 | Replaced with one-stop hub |
| `face.html` | 2 | Deleted after hub verified |
| `README.md` | 2 | Update "Demos" section |

## Open questions parked for later

1. Model provenance (see Phase 3).
2. Final name for the ball-toss demo and folder.
3. Whether to land the benchmark UI inside the hub or keep it as a separate page.
4. Whether to publish the library to npm. If yes, the demo import paths will switch from `../../src/` to `webgpu-vision`.
