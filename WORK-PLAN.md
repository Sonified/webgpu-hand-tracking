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

### Investigate: why hand pipeline gets ~10x speedup but face only ~2-3x

Live ball-toss bench on M1 Max, post-Phase-1.5, shows:

| | MediaPipe | WebGPU Vision | Speedup |
|---|---|---|---|
| Hand (2 hands) | 40-48ms mean, 56-60ms p95 | 0.7-6ms mean, 11-16ms p95 | ~8-10x |
| Face Detector (BlazeFace) | 18-21ms mean, 27-30ms p95 | 7-8ms mean, 11ms p95 | ~2.5-3x |
| Face Landmark (FaceMesh) | 27-31ms mean, 37-38ms p95 | 13.5ms mean, 20ms p95 | ~2x |

The hand delta is much bigger than the face delta. **Why** is worth understanding before we publish numbers, because the answer shapes the pitch.

Hypotheses to test (in rough order of likely contribution):

1. **Two-hand parallelism on our side, sequential cascade on MediaPipe's side.** Our `HandTracker` runs the two landmark workers under `Promise.all` so they execute on separate worker threads in true parallel. MediaPipe's `HandLandmarker` runs the per-hand landmark cascade sequentially within a single graph. So at 2 hands, our advantage is roughly (cascade-overhead-removal) x (2x parallelism). Face has no equivalent — only one face, no parallelism axis. To test: rerun both backends at `numHands=1`. If WGPU hand speedup at 1 hand drops to ~3-5x, parallelism is the dominant factor.

2. **Cascade depth = readback count.** MediaPipe pays a `glReadPixels` between every model in a cascade. Hand cascade for 2 hands = 1 palm detect + 2 landmark = 3 model calls = 3 readbacks. Face landmark cascade = 1 detect + 1 landmark = 2 readbacks. More readbacks per frame = more pain we save by removing them.

3. **Model size & inference time.** Hand landmark is a smaller model than face landmark (3.9 MB vs 4.8 MB, but face mesh runs at 256x256 which is more pixels). Smaller/cheaper inference = readback is a bigger fraction of total time = bigger relative win when readback is removed. Face mesh inference itself is heavier so removing readback is a smaller fraction of the total.

4. **PReLU decomposition tax (face-only headwind).** Our face landmark model has 69 PReLU ops decomposed into Relu+Neg+Mul+Add for WebGPU compatibility. MediaPipe's TFLite has PReLU as a native op. So the face landmark inference has a small structural disadvantage on our side that the hand landmark does not. See [PRELU_DECOMPOSITION.md](PRELU_DECOMPOSITION.md).

How to investigate: rerun the ball-toss bench with `numHands=1` (eliminates parallelism), then with the hand model running through a single landmark worker (eliminates per-hand parallelism while keeping the same cascade), and compare deltas. The numbers should tell us which hypothesis carries the most weight. Worth 30 minutes of experiment time before any public benchmark publication.



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
- **Proper hand Z-axis depth:** Currently the hand wireframe Z position is a rough offset from camera (`camera.z - 6 + lm.z * -3`). The hand landmark model outputs a real Z coordinate per joint, but we're only using it as a small perturbation. The fix: map the landmark Z values to actual world-space depth so that (a) the hand moves along the Z axis as you push it toward/away from the camera, (b) the wireframe size stays fixed in world units so it gets smaller with distance (perspective projection handles this naturally once Z is correct), and (c) the held ball and throw velocity incorporate real depth. This requires calibrating the landmark model's Z output range to the Three.js scene's coordinate system. The parallax compensation in `landmarkToScene` (the `pFrac = handZ / camera.z` factor) already handles the camera offset correctly — just the Z mapping needs work.
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

## Phase 4: Drop ONNX Runtime entirely — custom WGSL inference engine

**Status:** designed and parked. This is no longer just a performance optimization — it is the path to iOS support and the only way to make the demo work on phones.

### Why this just became urgent (2026-04-12)

ONNX Runtime's WebGPU backend **does not work on iOS Safari**. At all. Not our code, not our hosting, not our headers — Microsoft's ORT simply doesn't support iOS WebGPU ([issue #22776](https://github.com/microsoft/onnxruntime/issues/22776)). Even on Safari 26 desktop it has severe CPU/memory issues ([issue #26827](https://github.com/microsoft/onnxruntime/issues/26827)). This means the demo Robert built — the thing that proves WebGPU Vision works — cannot run on any iPhone. MediaPipe is NOT a fallback (see memory: this repo demos WebGPU Vision, period). So either we wait for Microsoft to fix ORT on Safari (timeline unknown), or we drop ORT entirely and run the models ourselves.

### What dropping ORT means

We have the ONNX model files. They contain the weights AND the computation graph. A neural network inference is:

1. Read the graph: "input → Conv2D → Relu → Conv2D → Relu → ... → output"
2. For each op, dispatch a WebGPU compute shader that runs the math using the stored weights
3. Feed the output buffer to the next op

Our four models use maybe **15-20 unique operation types** (Conv2D, DepthwiseConv2D, MatMul, Relu, PReLU, Add, Mul, Reshape, Transpose, Sigmoid, Pad, Resize, Concat, Softmax, BatchNorm). Each op is a WGSL compute shader, ~50-100 lines each. The "inference engine" is a loop that walks the graph and dispatches shaders in order.

**Estimated scope:**
- ~15-20 WGSL compute shaders for the ops our models use
- A graph parser (read ONNX protobuf or pre-convert to JSON at build time)
- A graph walker that allocates GPU buffers and dispatches shaders in order
- ~2000-3000 lines of JS/WGSL total
- Download: ~50KB instead of 23MB
- Zero WASM, zero threads, zero SharedArrayBuffer, zero crossOriginIsolated requirement
- Works everywhere WebGPU works — including iOS Safari

**What we lose:** ONNX Runtime's graph optimizer, operator fusion, memory planner, and the ability to run arbitrary ONNX models. We don't need any of that — we run exactly four small, known, fixed models.

**What we gain:** iOS support, ~450x smaller download, no WASM dependency, no COOP/COEP header requirement, no service worker hacks, works on GitHub Pages natively, total control over the inference pipeline.

### Confirmed: WebGPU compute works on iOS Safari (2026-04-12)

Deployed a bare-bones sanity check (`webgpu-test.html`) to Cloudflare Pages. On an iPhone running iOS 26.4.1, Safari:
- `navigator.gpu`: true
- `requestAdapter()`: success
- `requestDevice()`: success
- Compute shader (multiply array by 2): correct output
- **Result: SUCCESS**

WebGPU compute is fully functional on iOS Safari. The ONLY thing blocking our demo was ONNX Runtime's WASM backend. Pure WGSL compute shaders run fine. This confirms Phase 4 is not speculative — the target platform works.

### No more COOP/COEP headers

The entire reason we needed `Cross-Origin-Embedder-Policy` and `Cross-Origin-Opener-Policy` headers was `SharedArrayBuffer`, which ONNX Runtime's multi-threaded WASM requires. Pure WebGPU compute shaders don't use SharedArrayBuffer. No WASM threads, no special headers needed.

This means:
- **GitHub Pages works natively** (no service worker shim)
- **Any static host works** (S3, Netlify, Vercel, Cloudflare, anything)
- **No `crossOriginIsolated` requirement**
- The `coi-serviceworker.js` hack can be deleted entirely
- Just HTML + JS + WGSL. Open the page, it runs.

### Fused mega-shaders: why we'll be FASTER than ORT

ORT runs each neural network operation as a separate GPU shader dispatch. Conv2D, then BatchNorm, then Relu — three dispatches, three intermediate buffers, three round-trips through the GPU command queue. It has to do this because it's general-purpose: it doesn't know at compile time which ops will be adjacent.

We know exactly which ops are adjacent. Our models are fixed. So we fuse entire blocks:

| ORT approach (6 dispatches) | Fused approach (1 dispatch) |
|---|---|
| Conv2D → buffer → BatchNorm → buffer → Relu → buffer → Conv2D → buffer → BatchNorm → buffer → Relu | Conv+BN+Relu+Conv+BN+Relu as ONE shader |

Specific fusions available in our models:
- **Pre-bake BatchNorm into Conv2D weights at build time.** BN is just `(x - mean) / sqrt(var) * scale + bias` — fold scale/bias into the conv weights offline. Zero runtime cost, one fewer op.
- **Fuse Conv+Relu** — clamp output at the end of the convolution loop. Trivial.
- **Fuse entire residual blocks** — Conv+BN+Relu+Conv+BN+Add+Relu as one monolithic shader. One dispatch instead of seven.
- **Inline PReLU** — we decomposed PReLU into 4 ops (`Relu(x) + slope * -Relu(-x)`) for ORT compatibility. A custom shader is just `x > 0 ? x : slope * x`. One instruction instead of four dispatches.

Result: maybe **5-8 fused mega-shaders per model** instead of 60-80 individual dispatches. Fewer dispatches = less GPU idle time = potentially faster than ORT even on desktop.

### Verification: zero secret sauce

ONNX is an open format. Every operation, weight, shape, and connection in our models is fully readable:
```python
import onnx
model = onnx.load("palm_detection_lite.onnx")
for node in model.graph.node:
    print(node.op_type, [i for i in node.input], [o for o in node.output])
```

We can verify correctness by running ORT and our custom engine side-by-side, comparing output tensors at every stage. `GPUCommandEncoder` timestamps give nanosecond-precision per-dispatch benchmarks. Every step is measurable, every result is verifiable.

### Incremental build plan

Each step produces a working, testable artifact. We never go more than a day without something we can benchmark against ORT.

**Step 1: Model surgery (no code, pure analysis)**
- Dump the computation graph of all 4 models (palm, hand landmark, face detector, face landmark)
- List every unique op type and count occurrences
- Identify fusion opportunities (consecutive Conv+BN+Relu blocks, residual connections)
- Pre-bake BatchNorm into Conv weights (offline Python script, produces new weight files)
- Output: a JSON "fused graph" for each model describing the mega-shader sequence

**Step 2: One model, one shader at a time (palm detector first)**
- Palm detection is the simplest model (smallest graph, 192x192 input)
- Write the first fused mega-shader (probably the initial Conv+BN+Relu block)
- Hardcode buffer sizes (we know the exact tensor shapes)
- Run it, compare output against ORT's output for the same layer
- Iterate until the full palm detector runs end-to-end on pure WGSL
- Benchmark against ORT's palm detector on the same hardware

**Step 3: Hand landmark model**
- Same process. Slightly different ops (the landmark model has fully-connected layers at the end)
- Once both palm + hand landmark work, wire them into the existing pipeline in place of ORT
- The ball-toss demo runs on pure WGSL for hand tracking. First real proof of life.

**Step 4: Face detector + face landmark + blendshapes**
- Face detector is almost identical to palm (same architecture, different weights/anchors)
- Face landmark has PReLU — fuse it back to inline (the decomposition was an ORT workaround)
- Face blendshape model (52 expression coefficients from 146 landmarks)
- Full face tracking + blendshapes on pure WGSL. Ball-toss demo is now 100% ORT-free.

**Step 5: Delete ONNX Runtime**
- Remove `vendor/onnxruntime-web/` (23MB)
- Remove all worker code that references ORT
- The library is now pure WebGPU. ~50KB. Works on iOS Safari.
- Benchmark the full pipeline against the ORT version
- Ship it. Write the blog post. Tell Google.

### The original Phase 4 plan (still relevant for context)

### The realization

Phase 1.5 killed CPU readbacks **inside** the inference cascade. Camera frame -> letterbox -> detect -> warp -> landmark all stays on the GPU, all stages share an ONNX RT WebGPU device, every model-to-model handoff is via `Tensor.fromGpuBuffer()`. That is the headline win and it shipped.

But the workers' final output is **a JS `Float32Array` returned across the worker boundary**, which means the landmark coordinates get **read back from GPU to CPU at the very end** even when the consumer is going to turn around and draw them on screen — the most GPU-native operation imaginable. Worse, the existing demos then draw the wireframes via **Canvas2D** (`ctx.lineTo`), which is the slowest, most legacy 2D API the browser ships, in a project literally called WebGPU Vision. The whole identity of the library says "GPU never asks CPU for permission" and the very last step of the pipeline asks the CPU to draw lines.

In absolute perf terms on M1 Max this is microseconds, not milliseconds. It is not a bottleneck on current hardware. **But it is thematically incoherent**, it stops being free on weaker hardware (Chromebooks, phones), and it blocks the spec-pure version of the library from existing. Worth fixing as its own focused phase, after the user-facing work in Phases 2 and 3 is done.

### What "pure GPU" actually means

End to end, with no readbacks anywhere unless the consumer explicitly opts in:

```
camera frame (GPU texture)
  -> letterbox (compute, GPU buffer)
  -> detect inference (GPU tensor in/out)
  -> anchor decode + NMS (still CPU today, see option below)
  -> warp (compute, GPU buffer)
  -> landmark inference (GPU tensor in/out)
  -> landmark coords land in a GPU buffer
  -> bound directly as a vertex buffer to a render pipeline
  -> drawn via WebGPU render pass to the canvas
  -> swapchain present
```

Zero `mapAsync`, zero `Float32Array`, zero `ctx.lineTo`, zero `glReadPixels` anywhere.

### The worker contract change (the architectural piece)

Today every landmark worker postMessages a `Float32Array` of coordinates back to the main thread. That is the readback we need to make optional.

New contract: `init({ outputMode: 'cpu' | 'gpu' })`.

- **`'cpu'` (default, current behavior):** workers read landmarks back and post a `Float32Array`. Existing apps keep working with no changes. Apps that need landmarks in JS for gesture detection, parallax math, One Euro filtering, etc all stay on this path.
- **`'gpu'` (new):** workers do NOT read back. Instead they expose a `GPUBuffer` handle that the main thread (or another worker on the same shared device) can bind directly. The postMessage carries metadata (frame id, count, layout) but no coordinate data. The consumer uses the buffer in a render pipeline.

The hard part is that **`GPUBuffer` is not transferable across worker boundaries**. So `'gpu'` mode requires either:
- (a) all workers AND the renderer to share a single ONNX-RT-owned WebGPU device that lives in the main thread, with the landmark workers writing into a buffer the main thread also has a handle to, or
- (b) the workers expose a way for the main thread to import a buffer reference (some browsers allow this via `device.importExternalBuffer`-like APIs, but support is uneven).

Option (a) is the more portable path. It implies a refactor where the main thread owns the WebGPU device and the workers receive it via a structured-clone-able handle (or are kicked out entirely in favor of main-thread orchestration with off-thread inference via `OffscreenCanvas`). Either way, this is a real change to how `pipeline.js` and `face-pipeline.js` are wired, not a one-line patch.

### Three implementation options for the drawing side

Once landmarks live in a `GPUBuffer`, drawing becomes a pure WebGPU render pipeline. Three ways to wire it:

1. **Three.js `WebGPURenderer` for ball-toss specifically.** Three.js ships a WebGPU backend at `three/addons/renderers/webgpu/WebGPURenderer.js`. The ball-toss demo currently uses `WebGLRenderer` (the default). Switching to `WebGPURenderer` is a small change at the renderer instantiation, but the materials, lights, and post-processing the demo uses need to be checked for parity. Some features still lag the WebGL renderer. Once switched, the wireframe overlays become `THREE.LineSegments` / `THREE.Points` with a `BufferGeometry` whose vertex buffer **is the landmark `GPUBuffer` from the worker**. Zero copy. The whole visual stack is then WebGPU end-to-end. Lowest effort for the highest visible payoff in the showcase demo.

2. **Raw WebGPU render pipeline written by us.** ~100 lines of WGSL + JS. Vertex shader takes a uniform buffer of landmark positions, fragment shader does line + point rasterization. Used by the minimal demos (`/index.html`, `/face.html`) and exposed as a library helper `drawWireframeWebGPU(canvas, gpuBuffer, schema)`. Zero deps, zero Three.js, total control. The library-correct answer because it does not make webgpu-vision secretly require Three.js.

3. **Both — Three.js `WebGPURenderer` for ball-toss, raw WebGPU helper for the minimal demos and as a public library export.** Best of both. Ball-toss gets the full Three.js scene graph on WebGPU because it has a real 3D scene with lighting, shadows, projectiles. Library users who just want a drop-in wireframe overlay get the lightweight helper. This is the recommended path.

### Anchor decode + NMS: should they move to compute too?

Currently both the hand and face pipelines decode anchors and run weighted NMS on the **CPU**, after reading the detection model's output back as a `Float32Array`. That is a separate readback we have not addressed.

For "pure GPU" to be honest end-to-end, anchor decode and NMS need to be compute shaders too. Anchor decode is trivially parallel (one workgroup per anchor, multiply-add math). NMS is harder because it is order-dependent and weighted NMS specifically needs to average overlapping detections by score — there is published work on GPU NMS approaches (parallel reduction, sweep-and-prune, etc) but it is not free to implement.

Defer this until after the worker contract change lands. It is a meaningful piece of work on its own and it is downstream of the contract decision.

### Order of operations within Phase 4

1. Design and prototype the `outputMode: 'gpu'` worker contract on **one** worker (suggest: `landmark-worker.js`, since it is the simplest cascade endpoint). Confirm the cross-worker buffer sharing approach actually works on M1 Max Chrome, then on a couple of secondary platforms.
2. Land option 2 first (raw WebGPU wireframe helper), wired to that one worker, with one demo using it end-to-end. Smallest possible vertical slice. Validate that the GPU-only path is actually faster or at least at parity with the readback path (it should be, especially on weaker hardware).
3. Roll the contract change out to the other workers (palm, face detection, face landmark, face blendshape) once the pattern is proven.
4. Add option 1 (Three.js `WebGPURenderer` swap) for ball-toss.
5. Move anchor decode + NMS to compute shaders, removing the last detection-side readback.
6. Update the README and ARCHITECTURE.md to describe the pure-GPU path as the default and CPU readback as opt-in.

### Why this is Phase 4 and not "do it now"

- It is not a perf bottleneck on current hardware. Phase 2 (the unified comparison hub) and the Phase 3 polish items move the library from "works" to "people can actually use it." Phase 4 moves it from "works in spirit" to "works in spec." That ordering matches user impact.
- The worker contract change deserves careful design and a clean session, not a rushed implementation jammed between two other in-flight things.
- Doing Phase 4 before Phase 2 means Phase 2 (the hub) would have to be ported twice — once as Canvas2D, once as WebGPU. Doing Phase 2 first, then Phase 4, lets us do the WebGPU port once across all the demos at the same time.
- The per-mode settings work that the user originally asked for (and that we got happily distracted from when MediaPipe FaceLandmarker came up) needs to land first. That is the more pressing user-facing item.

When the next session takes this on: read this whole section, look at the **investigate why hand >> face** entry above (it informs which workers benefit most from going pure-GPU), and prototype on one worker before touching all of them.

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
