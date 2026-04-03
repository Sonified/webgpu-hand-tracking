# WebGPU Hand Tracking

Real-time hand tracking in the browser via WebGPU compute shaders. No WebGL, no WASM black box, no glReadPixels bottleneck.

Runs Google's BlazePalm + Hand Landmark neural networks through ONNX Runtime Web with the WebGPU backend. The full MediaPipe hand tracking pipeline, rebuilt for the GPU-native web.

## Status

Under construction. The models are validated, the architecture is designed, the build is underway.

## Why

MediaPipe's browser SDK uses WebGL internally for "GPU" inference, but synchronous `glReadPixels` readbacks cost 8-22ms per call. Two-hand tracking drops to ~15fps even on modern hardware. The WASM binary is sealed -- you can't optimize what you can't see.

This project replaces the WebGL inference path with WebGPU compute shaders via ONNX Runtime Web. No synchronous readbacks. Full pipeline visibility. True parallel two-hand inference.

## Architecture

```
Camera Frame
    |
    v
Palm Detection (BlazePalm, 192x192, 1.76M params)
    |
    v
Anchor decode + Weighted NMS + Rotation
    |
    v
Crop + Warp (per detected hand)
    |
    v
Hand Landmark (224x224, 2M params) -- parallel per hand
    |
    v
21 3D keypoints per hand
```

Both neural networks run as WebGPU compute shaders. The tracking loop skips palm detection when hands are already found, running only the lightweight landmark model frame-to-frame.

## Models

Uses Google's published hand tracking models converted to ONNX format:

- **BlazePalm** (palm detection): 1.76M parameters, 192x192 input
- **Hand Landmark Full**: 2.0M parameters, 224x224 input

Models are openly available via [PINTO0309's model zoo](https://github.com/PINTO0309/PINTO_model_zoo), [Qualcomm AI Hub](https://huggingface.co/qualcomm/MediaPipe-Hand-Detection), and [OpenCV Zoo](https://github.com/opencv/opencv_zoo).

## Requirements

- Chrome 113+, Edge 113+, or Safari 18+ (WebGPU support)
- Any device with a camera

## Acknowledgments

- Google MediaPipe team for the trained models and published research
- [PINTO0309](https://github.com/PINTO0309) for ONNX model conversions and the reference Python implementation that validated this approach
- Microsoft ONNX Runtime team for the WebGPU execution provider

## License

MIT
