# WebGPU Vision

A port of Google's MediaPipe hand tracking pipeline to run on WebGPU compute shaders in the browser. No sealed WASM binary, no WebGL, no `glReadPixels` bottleneck.

## Why This Exists

MediaPipe's browser SDK uses WebGL internally for inference with synchronous `glReadPixels` readbacks costing 8-22ms per call. Two-hand tracking drops to ~15fps. The WASM binary is sealed; you can't optimize it. This project replaces the inference path with WebGPU compute shaders via ONNX Runtime Web, achieving 120fps two-hand tracking with zero CPU readback.

The GPU to the CPU: "Hold my bear." 🧸

## Quick Start

```bash
npm install

# Download the ONNX models
mkdir -p models
curl -L "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx" \
  -o models/palm_detection_lite.onnx
curl -L "https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb.onnx" \
  -o models/hand_landmark_full.onnx

npm run dev
```

Open http://localhost:5173 in Chrome 113+ (or any browser with WebGPU support). Allow camera access.

## Architecture

Three Web Workers, all GPU-accelerated. Main thread is pure orchestration.

- **Palm Worker**: WebGPU compute shader letterbox preprocessing + BlazePalm ONNX inference + anchor decode + weighted NMS. Runs async, fire-and-forget, never blocks tracking.
- **Landmark Worker 0**: WebGPU compute shader affine warp + Hand Landmark ONNX inference (hand 0). Shares ONNX RT's WebGPU device via `ort.env.webgpu.device`. `Tensor.fromGpuBuffer()` passes the warp output directly to inference with zero CPU readback.
- **Landmark Worker 1**: Same as above for hand 1. Both run in parallel via `Promise.all`.

Main thread only does: `createImageBitmap(video)`, `postMessage` to workers, `landmarksToRect` math (12 points), and canvas overlay drawing. No ONNX imports, no preprocessing, no inference.

## Pipeline

```
Camera Frame (640x480)
    |
    v
createImageBitmap (main thread, fast GPU op)
    |
    ├──> Palm Worker (when empty slots exist, fire-and-forget)
    |      GPU letterbox compute shader -> BlazePalm inference -> anchor decode -> weighted NMS
    |      Returns detections with keypoints
    |
    ├──> Landmark Worker 0 (Promise.all, parallel)
    |      GPU affine warp compute shader -> Tensor.fromGpuBuffer -> ONNX inference
    |      Data never leaves GPU until 63 landmark floats (252 bytes) return
    |
    └──> Landmark Worker 1 (Promise.all, parallel)
           Same as above for second hand
    |
    v
Main thread receives landmarks, computes next-frame ROI from stable
palm/MCP landmarks, draws 21-point skeleton overlay.
Tracking loop skips palm detection when hands are found.
```

## Key Technical Decisions

- **OpenCV Zoo ONNX models** (Apache 2.0): `palm_detection_mediapipe_2023feb.onnx` (3.7MB, 192x192 input, [0,1] normalization) and `handpose_estimation_mediapipe_2023feb.onnx` (3.9MB, 224x224 input, [0,1] normalization). NOT the PINTO post-processed models which have NMS baked in.
- **Anchor generation**: 2016 anchors, strides [8,16,16,16], 2 anchors per grid cell, fixed anchor size. Scores are raw logits (sigmoid applied in decode).
- **Weighted NMS** (not standard suppress and discard): overlapping detections averaged by score.
- **Detection-to-rotated-rect**: ported from geaxgx reference. Works in pixel space, square_long=true (longer side * 2.9), shift_y=-0.5 in rotated direction.
- **Landmarks-to-rect**: uses 12 stable palm/MCP landmarks (IDs 0,1,2,3,5,6,9,10,13,14,17,18), no fingertips. Rotation from wrist to weighted average of 3 MCPs. Bounds computed in rotated coordinate space.
- **Affine warp**: 3-point affine transform matching mediapipe_utils.py `warp_rect_img()`. Corner ordering: p0(BL), p1(TL), p2(TR), p3(BR). Warp uses pts[1:] = [TL, TR, BR] mapped to [(0,0), (S,0), (S,S)].
- **Zero CPU readback**: Each landmark worker gets ONNX RT's device via `await ort.env.webgpu.device` (after session creation), builds compute shader on that device, uses `Tensor.fromGpuBuffer()` to pass warp output directly to inference.
- **Web Workers for true parallelism**: ONNX RT's WASM backend shares memory within a thread, so concurrent `.run()` calls on the same thread deadlock. Separate workers = separate WASM instances = true parallel inference.

## Performance

| Metric | MediaPipe WebGL | webgpu-vision |
|--------|----------------|---------------|
| Two-hand tracking | ~15fps | 120fps |
| CPU readback per hand | 8-22ms (glReadPixels) | 252 bytes (landmarks only) |
| Pipeline visibility | Sealed WASM binary | Full source |
| Parallel two-hand | Serial (same WebGL context) | True parallel (separate workers) |

## Project Structure

```
src/
  main.js              Entry point, webcam, render loop, overlay drawing
  pipeline.js           HandTracker class, orchestration, worker management, landmarksToRect
  palm-worker.js        Palm detection worker (GPU letterbox + inference + decode + NMS)
  landmark-worker.js    Landmark inference worker (GPU affine warp + inference + projection)
  anchors.js            Anchor generation + detection decoding for BlazePalm
  nms.js                Weighted NMS + detectionToRect
  preprocessing.js      Palm preprocessing (canvas fallback for palm worker)
models/                 ONNX files (gitignored, see Quick Start)
```

## Requirements

- Chrome 113+, Edge 113+, or Safari 18+ (WebGPU support)
- A device with a camera
- Node.js 18+ (for the dev server)

## Models

Uses Google's MediaPipe hand tracking models converted to ONNX format by the OpenCV Zoo team (Apache 2.0 license):

| Model | Input | Output | Size |
|-------|-------|--------|------|
| Palm detection | (1, 192, 192, 3) float32 [0,1] | 2016 anchor boxes + scores | 3.7 MB |
| Hand landmark | (1, 224, 224, 3) float32 [0,1] | 21 keypoints (x,y,z) + hand flag + handedness | 3.9 MB |

## Acknowledgments

- Google MediaPipe team for the trained models and published research
- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) for ONNX model conversions
- [PINTO0309](https://github.com/PINTO0309) for the reference Python implementation
- [geaxgx](https://github.com/geaxgx/depthai_hand_tracker) for the clearest reference glue code
- Microsoft ONNX Runtime team for the WebGPU execution provider

## License

MIT
