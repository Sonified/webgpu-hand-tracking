# WebGPU Vision

Hand and face tracking running entirely on WebGPU compute shaders in the browser. No sealed WASM binary, no WebGL, no `glReadPixels` bottleneck.

The GPU turns to the CPU and says... "Hold my bear" 🧸

## Why This Exists

MediaPipe's browser SDK uses WebGL internally for inference with synchronous readbacks. This project replaces the inference path with WebGPU compute shaders via ONNX Runtime Web -- zero CPU readback during inference, true parallel workers, and full pipeline visibility.

## Quick Start

```bash
npm install
npm run dev
```

Open http://localhost:5173 for hand tracking, or http://localhost:5173/face.html for face tracking. Chrome 113+ (or any browser with WebGPU support). Allow camera access.

Models are included in the repo (Apache 2.0). No separate downloads needed.

## Demos

- **`/`** (`index.html`): hand tracking wireframe overlay (hub work-in-progress, see [WORK-PLAN.md](WORK-PLAN.md))
- **`/face.html`**: face landmark + blendshape wireframe overlay
- **`/demos/ball-toss/`**: full showcase. Head-coupled 3D parallax with Three.js, hand-driven projectile throwing, MediaPipe vs WebGPU Vision A/B toggle, persisted UI settings, One Euro filtering. The demo that proves the library is production-ready.

## Architecture

### Hand Tracking

Three Web Workers, all GPU-accelerated. Main thread is pure orchestration.

- **Palm Worker**: WebGPU compute shader letterbox + BlazePalm inference + anchor decode + weighted NMS. Fire-and-forget, never blocks tracking.
- **Landmark Worker 0 + 1**: WebGPU compute shader affine warp + Hand Landmark inference. `Tensor.fromGpuBuffer()` passes warp output directly to inference with zero CPU readback. Both run in parallel via `Promise.all`.

### Face Tracking

Two Web Workers, same GPU architecture.

- **Face Detection Worker**: WebGPU compute shader letterbox + BlazeFace inference (128x128, 896 anchors) + anchor decode + weighted NMS. Fire-and-forget.
- **Face Landmark Worker**: WebGPU compute shader affine warp + Face Mesh inference (256x256, 478 landmarks). Same zero-copy `Tensor.fromGpuBuffer()` path.

The face landmark model required a PReLU decomposition to run on WebGPU -- see [PRELU_DECOMPOSITION.md](PRELU_DECOMPOSITION.md).

## Pipeline

```
Camera Frame (640x480)
    |
    v
createImageBitmap (main thread, fast GPU op)
    |
    |  HAND TRACKING                         FACE TRACKING
    |                                        
    ├──> Palm Worker (fire-and-forget)       ├──> Face Detection Worker (fire-and-forget)
    |      GPU letterbox -> BlazePalm        |      GPU letterbox -> BlazeFace
    |      -> anchor decode -> weighted NMS  |      -> anchor decode -> weighted NMS
    |                                        |
    ├──> Landmark Worker 0 ──┐               └──> Face Landmark Worker
    └──> Landmark Worker 1 ──┤ Promise.all         GPU warp -> Tensor.fromGpuBuffer
                             |                     -> 478-landmark inference
    GPU warp -> Tensor.fromGpuBuffer               -> 1434 floats return to CPU
    -> 21-landmark inference                 
    -> 63 floats return to CPU               
    |                                        
    v                                        
Main thread: landmarksToRect, draw overlay, tracking loop
```

Data never leaves GPU until the final landmark coordinates return (252 bytes per hand, 5.7KB per face).

## Performance

Tested on MacBook Pro M1 Max (32-core GPU, 64GB), Chrome 146, macOS 26.2, 640x480 camera. MediaPipe uses its official `@mediapipe/tasks-vision` with GPU delegate. See [benchmark/](benchmark/) to reproduce.

### Hand Tracking (two hands)

| Metric | MediaPipe | webgpu-vision |
|--------|-----------|---------------|
| FPS | ~47fps | ~72fps |
| Per-frame latency | ~19ms | ~13ms |
| CPU readback | Full frame via WebGL | 252 bytes (landmarks only) |
| Parallel two-hand | Serial (same WebGL context) | True parallel (separate workers) |

### Face Tracking (single face, 478 landmarks)

| Metric | MediaPipe | webgpu-vision |
|--------|-----------|---------------|
| FPS | ~55fps | ~77fps |
| Per-frame latency | ~16ms | ~10ms |
| CPU readback | Full frame via WebGL | 5.7 KB (landmarks only) |

## Key Technical Decisions

- **Zero CPU readback**: Workers get ONNX RT's device via `ort.env.webgpu.device`, build compute shaders on the same device, and use `Tensor.fromGpuBuffer()` to pass warp output directly to inference.
- **Web Workers for true parallelism**: ONNX RT's WASM backend shares memory within a thread, so concurrent `.run()` calls deadlock. Separate workers = separate WASM instances = true parallel inference.
- **Weighted NMS** (not standard suppress and discard): overlapping detections averaged by score, matching MediaPipe's internal approach.
- **PReLU decomposition**: The face landmark model's 69 PReLU ops aren't supported by ONNX RT's WebGPU backend. Decomposing `PReLU(x, slope)` into `Relu(x) + slope * (-Relu(-x))` keeps everything on GPU. Same math, zero accuracy loss, 12x speedup (9fps to 77fps). See [PRELU_DECOMPOSITION.md](PRELU_DECOMPOSITION.md).
- **Model URL auto-switching**: `model-urls.js` serves from local `/models/` on localhost, from `https://models.now.audio/` in production. Zero config.

## Project Structure

```
src/
  main.js                  Hand tracking entry point, webcam, render loop
  pipeline.js              HandTracker class, orchestration, worker management
  palm-worker.js           Palm detection worker (GPU letterbox + inference + decode + NMS)
  landmark-worker.js       Hand landmark worker (GPU affine warp + inference + projection)
  anchors.js               Anchor generation + decoding for BlazePalm
  nms.js                   Weighted NMS + detectionToRect for hands
  preprocessing.js         Palm preprocessing (canvas fallback)
  face-main.js             Face tracking entry point
  face-pipeline.js         FaceTracker class, orchestration
  face-detection-worker.js Face detection worker (GPU letterbox + BlazeFace inference)
  face-landmark-worker.js  Face landmark worker (GPU warp + 478-point inference)
  face-anchors.js          Anchor generation + decoding for BlazeFace
  face-nms.js              Weighted NMS + faceDetectionToRect
  model-urls.js            Auto-switching model URLs (local dev / CDN production)
models/                    ONNX model files (Apache 2.0, see models/LICENSE.md)
```

## Models

All models are Apache 2.0 licensed. Hand models from [OpenCV Zoo](https://github.com/opencv/opencv_zoo). Face models converted from Google's [MediaPipe .task bundle](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) via tf2onnx, with PReLU decomposition for WebGPU compatibility.

| Model | Input | Output | Size |
|-------|-------|--------|------|
| Palm detection | (1, 192, 192, 3) float32 [0,1] | 2016 anchor boxes + scores | 3.7 MB |
| Hand landmark | (1, 224, 224, 3) float32 [0,1] | 21 keypoints (x,y,z) + hand flag + handedness | 3.9 MB |
| Face detection | (1, 128, 128, 3) float32 [0,1] | 896 anchor boxes + scores | 409 KB |
| Face landmark | (1, 256, 256, 3) float32 [0,1] | 478 landmarks (x,y,z) + face flag | 4.8 MB |

## Requirements

- Chrome 113+, Edge 113+, or Safari 18+ (WebGPU support)
- A device with a camera
- Node.js 18+ (for the dev server)

## Acknowledgments

- Google MediaPipe team for the trained models and published research (Apache 2.0)
- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) for hand model ONNX conversions
- [PINTO0309](https://github.com/PINTO0309) for the reference Python implementation
- [geaxgx](https://github.com/geaxgx/depthai_hand_tracker) for the clearest reference glue code
- Microsoft ONNX Runtime team for the WebGPU execution provider

## License

MIT
