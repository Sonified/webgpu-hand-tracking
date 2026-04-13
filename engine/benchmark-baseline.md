# Benchmark Baseline — Before Optimization

Date: 2026-04-12
Hardware: M1 Max (headless Chrome, --enable-unsafe-webgpu)
Iterations: 50, warmup excluded

## Results (naive per-dispatch approach)

| Model | WGSL (ms) | ORT-GPU (ms) | ORT-CPU (ms) | vs ORT-GPU |
|---|---|---|---|---|
| Palm Detector (192x192, 124 nodes → 64 dispatches) | 18.61 | 33.41 | 29.59 | **1.8x faster** |
| Hand Landmark (224x224, 99 nodes → 53 dispatches) | 12.67 | 7.02 | 18.32 | 0.6x (slower) |
| Face Detector (128x128, 95 nodes → 53 dispatches) | 12.70 | 3.73 | 3.07 | 0.3x (slower) |

## Analysis

Palm detector wins because it's the largest model — actual GPU compute time dominates dispatch overhead.

Hand landmark and face detector LOSE because dispatch overhead dominates actual compute:
- Each dispatch = create uniform buffer + writeBuffer + beginComputePass + setBindGroup + dispatchWorkgroups + end
- 53 dispatches × ~0.15ms overhead each ≈ 8ms of pure overhead before any math happens
- Face detector only has 0.39MB of weights — the model is so small that ORT's batched execution crushes our per-dispatch approach

## What to fix

Mega-shader fusion: instead of 53 separate dispatches, loop through blocks inside a single shader.
- DW Conv → 1x1 Conv → Add → Relu as ONE dispatch (not 4)
- Entire backbone stages as 1-2 dispatches
- Target: 5-8 total dispatches per model instead of 50-60
- Expected speedup: 3-5x from overhead reduction alone, before any shader optimization

Pre-allocate uniform buffers: currently creating a new GPUBuffer for params on every dispatch.
Pool and reuse instead.

This baseline is the "before" for comparison after mega-shader fusion ships.
