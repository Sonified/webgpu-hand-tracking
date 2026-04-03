# WebGPU Hand Tracking Pipeline

Rebuild MediaPipe's hand tracking inference on ONNX Runtime Web + WebGPU backend. Eliminate the glReadPixels bottleneck, own the full pipeline, run two-hand tracking at 60fps+ in-browser.

Nobody has done this publicly.

## Why

MediaPipe's WASM uses WebGL internally. Even with "GPU delegate," it does synchronous `glReadPixels` readbacks that cost 8-22ms per call. With two hands, landmark regression runs twice serially inside a sealed WASM binary. We can't optimize what we can't see.

ONNX Runtime Web with WebGPU backend runs neural nets via compute shaders. No glReadPixels. IO Binding keeps tensors on GPU between operations. We get full control over the pipeline and can parallelize the two-hand landmark passes.

## The Two Models

### 1. BlazePalm (Palm Detection)

- Parameters: 1.76M
- Input: `(1, 192, 192, 3)` float32, normalized to [-1, 1]
- Output:
  - Regressors: `(1, 2016, 18)` -- 2016 anchor boxes, each with bbox (4) + 7 keypoints x 2 coords (14)
  - Classificators: `(1, 2016, 1)` -- confidence score per anchor
- Purpose: find WHERE hands are in the frame
- Speed: ~2-5ms on mobile GPU natively. The browser overhead is the problem, not the model.

### 2. Hand Landmark

- Parameters: 2.0M (Full), 1.0M (Lite)
- Input: `(1, 224, 224, 3)` float32, normalized to [0, 1]
- Output:
  - landmarks: `(1, 63)` -- 21 keypoints x 3 (x, y, z), normalized to 224x224 space
  - hand_flag: `(1, 1)` -- probability hand is present (sigmoid)
  - handedness: `(1, 1)` -- left/right classification
  - world_landmarks: `(1, 63)` -- 21 keypoints x 3 in meters
- Purpose: given a cropped hand region, extract the 21 3D joint positions

## The Full Pipeline

```
Camera Frame (320x240)
        |
        v
[Letterbox pad to square]
[Resize to 192x192]
[Normalize: 2 * (pixel/255) - 1]
        |
        v
===== PALM DETECTION (BlazePalm) =====
        |
        v
[Sigmoid on scores]
[Decode anchors: raw / 192 + anchor_center]
[Score threshold: 0.5]
[Weighted NMS: IoU 0.3, average overlapping boxes by score]
        |
        v
For each detected palm:
  [Compute rotation: atan2 from wrist to middle finger keypoint]
  [Expand bbox 2.9x, shift -0.5y]
  [Affine warp: crop rotated rectangle from original frame]
  [Resize to 224x224]
  [Normalize: pixel / 255]
        |
        v
===== HAND LANDMARK (per hand) =====
        |
        v
[Reshape 63 -> 21 x 3]
[Project landmarks back through inverse affine transform]
[If hand_flag > threshold: skip palm detection next frame, use landmarks for next ROI]
[If hand_flag < threshold: re-run palm detection]
```

## Anchor Generation

Palm detection uses 2016 pre-computed anchor positions. Parameters:

```
num_layers: 4
strides: [8, 16, 16, 16]
input_size: 192
anchor_offset: 0.5
aspect_ratios: [1.0]
fixed_anchor_size: true
```

Generate at startup, store as Float32Array. Each anchor is (x_center, y_center). With fixed_anchor_size, width and height are always 1.0.

Pseudocode:
```
anchors = []
for each layer (stride in strides):
  grid_size = ceil(192 / stride)
  for y in 0..grid_size:
    for x in 0..grid_size:
      // 2 anchors per grid cell for first layer, 6 for others
      num_anchors = 2 if stride == 8 else 6
      for n in 0..num_anchors:
        cx = (x + 0.5) / grid_size
        cy = (y + 0.5) / grid_size
        anchors.push(cx, cy)
```

Total: 2016 anchors. Verify count matches model output.

## Anchor Decoding

```javascript
function decodeDetections(regressors, scores, anchors) {
  const detections = [];
  for (let i = 0; i < 2016; i++) {
    const score = sigmoid(scores[i]);
    if (score < 0.5) continue;

    const ax = anchors[i * 2];
    const ay = anchors[i * 2 + 1];

    // Decode bbox
    const cx = regressors[i * 18 + 0] / 192 + ax;
    const cy = regressors[i * 18 + 1] / 192 + ay;
    const w  = regressors[i * 18 + 2] / 192;
    const h  = regressors[i * 18 + 3] / 192;

    // Decode 7 keypoints
    const keypoints = [];
    for (let k = 0; k < 7; k++) {
      keypoints.push({
        x: regressors[i * 18 + 4 + k * 2] / 192 + ax,
        y: regressors[i * 18 + 5 + k * 2] / 192 + ay,
      });
    }

    detections.push({ cx, cy, w, h, score, keypoints });
  }
  return detections;
}
```

## Weighted NMS

MediaPipe uses weighted NMS, not standard suppress-and-discard. Overlapping detections are averaged by score.

```javascript
function weightedNMS(detections, iouThreshold = 0.3) {
  detections.sort((a, b) => b.score - a.score);
  const kept = [];

  while (detections.length > 0) {
    const best = detections.shift();
    const cluster = [best];

    detections = detections.filter(d => {
      if (computeIoU(best, d) > iouThreshold) {
        cluster.push(d);
        return false;
      }
      return true;
    });

    // Weighted average of cluster
    let totalW = 0;
    let cx = 0, cy = 0, w = 0, h = 0;
    for (const d of cluster) {
      cx += d.cx * d.score;
      cy += d.cy * d.score;
      w += d.w * d.score;
      h += d.h * d.score;
      totalW += d.score;
    }
    kept.push({
      cx: cx / totalW, cy: cy / totalW,
      w: w / totalW, h: h / totalW,
      score: best.score,
      keypoints: best.keypoints, // use highest-score keypoints
    });
  }
  return kept;
}
```

## Detection to Rotated Rectangle

The hand landmark model expects an upright hand. Use the palm keypoints to compute rotation.

```javascript
function detectionToRect(detection) {
  // Keypoint 0 = wrist, keypoint 2 = middle finger MCP
  const wrist = detection.keypoints[0];
  const middle = detection.keypoints[2];

  // Rotation angle to align hand vertically
  const angle = Math.atan2(-(middle.y - wrist.y), middle.x - wrist.x) - Math.PI / 2;

  // Expand bbox by 2.9x, shift center by -0.5 * height in rotated direction
  const scale = 2.9;
  const shiftY = -0.5;

  const w = detection.w * scale;
  const h = detection.h * scale;
  const cx = detection.cx + shiftY * h * Math.sin(angle);
  const cy = detection.cy - shiftY * h * Math.cos(angle);

  return { cx, cy, w, h, angle };
}
```

## Affine Warp (Crop Rotated Region)

Extract the rotated rectangle from the original frame and resize to 224x224.

Canvas approach (CPU):
```javascript
function cropRotatedRect(video, rect, outSize) {
  const canvas = new OffscreenCanvas(outSize, outSize);
  const ctx = canvas.getContext('2d');

  ctx.translate(outSize / 2, outSize / 2);
  ctx.rotate(rect.angle);
  ctx.scale(outSize / (rect.w * videoWidth), outSize / (rect.h * videoHeight));
  ctx.translate(-rect.cx * videoWidth, -rect.cy * videoHeight);
  ctx.drawImage(video, 0, 0);

  return ctx.getImageData(0, 0, outSize, outSize);
}
```

WebGPU compute approach (GPU, future optimization):
A compute shader that reads from the source texture with a 2x3 affine matrix and writes to a 224x224 output texture. Keeps everything on GPU between palm detection output and landmark input.

## Tracking Loop (Skip Palm Detection)

The key optimization: palm detection is expensive, landmark regression is cheaper. Once a hand is found, use the landmark output to compute the next frame's crop region. Only re-run palm detection when the hand_flag drops below threshold.

```javascript
const handSlots = [
  { active: false, rect: null, landmarks: null },
  { active: false, rect: null, landmarks: null },
];
let framesSincePalmDetect = 0;
const REDETECT_INTERVAL = 30; // full palm detection every N frames

async function processFrame(video) {
  const needPalmDetect = !handSlots.some(s => s.active)
    || framesSincePalmDetect >= REDETECT_INTERVAL;

  if (needPalmDetect) {
    const palms = await runPalmDetection(video);
    assignPalmsToSlots(palms);
    framesSincePalmDetect = 0;
  }

  // Run landmark model on each active hand slot (can be parallel!)
  const promises = handSlots.map(async (slot, i) => {
    if (!slot.active) return;
    const crop = cropRotatedRect(video, slot.rect, 224);
    const result = await runLandmarkModel(crop);

    if (result.hand_flag > 0.5) {
      slot.landmarks = projectLandmarks(result.landmarks, slot.rect);
      slot.rect = landmarksToRect(slot.landmarks); // next frame's crop
    } else {
      slot.active = false;
      slot.landmarks = null;
    }
  });

  await Promise.all(promises); // both hands in parallel!
  framesSincePalmDetect++;
}
```

## ONNX Runtime Web Setup

```javascript
import * as ort from 'onnxruntime-web/webgpu';

// Load both models
const palmSession = await ort.InferenceSession.create('palm_detection.onnx', {
  executionProviders: ['webgpu'],
  graphOptimizationLevel: 'all',
});

const landmarkSession = await ort.InferenceSession.create('hand_landmark_full.onnx', {
  executionProviders: ['webgpu'],
  graphOptimizationLevel: 'all',
});

// Run palm detection
async function runPalmDetection(imageData) {
  const input = new ort.Tensor('float32', preprocessPalm(imageData), [1, 192, 192, 3]);
  const { regressors, classificators } = await palmSession.run({ input });
  return decodeDetections(regressors.data, classificators.data, anchors);
}

// Run landmark model
async function runLandmarkModel(cropData) {
  const input = new ort.Tensor('float32', preprocessLandmark(cropData), [1, 224, 224, 3]);
  const results = await landmarkSession.run({ input });
  return {
    landmarks: results.landmarks.data,      // Float32Array(63)
    hand_flag: results.hand_flag.data[0],    // number
    handedness: results.handedness.data[0],  // number
  };
}
```

## Model Sources

| Model | Source | Size |
|-------|--------|------|
| palm_detection_lite.onnx | [PINTO model zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/033_Hand_Detection_and_Tracking) | ~3.7 MB |
| hand_landmark_full.onnx | [PINTO hand_landmark](https://github.com/PINTO0309/hand_landmark) | ~7.7 MB |
| hand_landmark_lite.onnx | Same | ~3.9 MB |

Alternative source: [Qualcomm AI Hub](https://huggingface.co/qualcomm/MediaPipe-Hand-Detection) has official ONNX exports.

## Reference Implementations (for porting)

- [PINTO0309/hand-gesture-recognition-using-onnx](https://github.com/PINTO0309/hand-gesture-recognition-using-onnx) -- full pipeline in Python
- [geaxgx/depthai_hand_tracker](https://github.com/geaxgx/depthai_hand_tracker) -- cleanest glue code reference
- [SBoulanger/blazepalm](https://github.com/SBoulanger/blazepalm) -- clean anchor decoding and NMS

## Build Order

### Phase 1: Palm detection working (day 1)
1. Download palm_detection ONNX model
2. Set up ONNX Runtime Web with WebGPU
3. Port anchor generation (static, compute once)
4. Port preprocessing (letterbox, resize, normalize)
5. Run model, decode outputs
6. Port weighted NMS
7. Verify: draw detected palm bboxes on canvas overlay

### Phase 2: Landmark model working (day 1-2)
1. Download hand_landmark ONNX model
2. Port detection-to-rotated-rect conversion
3. Implement affine warp via OffscreenCanvas
4. Run landmark model on crop
5. Project landmarks back to image space
6. Verify: draw 21 keypoints on canvas overlay

### Phase 3: Tracking loop (day 2)
1. Implement hand_flag-based palm detection skip
2. Use landmarks to compute next-frame ROI
3. Handle hand loss and re-detection
4. Two-hand slot assignment

### Phase 4: Integration (day 2-3)
1. Wire into parallax-demo.html replacing MediaPipe workers
2. Run both landmark passes via Promise.all (true parallel)
3. Benchmark against MediaPipe baseline
4. Tune confidence thresholds

### Phase 5: Full GPU path (stretch)
1. WebGPU compute shader for affine warp (keep crop on GPU)
2. IO Binding to avoid CPU roundtrip between models
3. Potentially run postprocessing (anchor decode, NMS) as compute shaders

## Expected Performance

| Stage | MediaPipe WebGL | ONNX + WebGPU (estimated) |
|-------|----------------|--------------------------|
| Palm detection | 10-15ms + glReadPixels | 5-10ms, no sync |
| Landmark (per hand) | 10-15ms + glReadPixels | 5-10ms, no sync |
| Two hands total | 40-55ms (~20fps) | 15-25ms (~40-60fps) |
| With tracking (skip palm) | 25-35ms (~30fps) | 10-20ms (~50-100fps) |

Conservative estimate: 2x improvement. Optimistic: 3-4x with IO Binding and GPU crop.

## Why This Matters

- Nobody has done this publicly
- Eliminates the fundamental WebGL/glReadPixels bottleneck
- Full pipeline visibility and control
- True parallel two-hand inference via Promise.all
- Foundation for custom gesture models in the same WebGPU pipeline
- Potential to keep data on GPU from camera frame through to particle system rendering
