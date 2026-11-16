# Experiments

Everything tried in project 3, including what did not work. Final numbers are on `val` (3769 frames), F1 at the
KITTI 3D IoU thresholds 0.7 / 0.5 / 0.5, Moderate difficulty, same protocol as projects 1–2.

Data hygiene: Faster R-CNN is fine-tuned on `train[:-300]`; every fusion and C2-s choice (weights, thresholds,
offsets, headings, feature sets) is made on `train[-300:]`, unseen by both detectors; `val` is used only to report.
The one exception is noted in §4.3.

## 1. Camera detector C2: Faster R-CNN fine-tuning

Faster R-CNN ResNet-50 FPN with COCO weights, new 4-output box predictor (background + Car / Pedestrian /
Cyclist), SGD lr 0.005, momentum 0.9, weight decay 5e-4, StepLR ×0.1, horizontal flip as augmentation.
Trained on 3212 frames, monitored with COCO-style 2D AP on 200 frames held out of `train[:-300]`.

| Run | Input scale | Epochs (LR drop) | Batch | Time / epoch (GPU) | AP @[.5:.95] | AP @.5 | AP @.75 | AP small | AP medium | AP large |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | default (×1.07, 1333 × 402) | 5 (after 3) | 16 | 2:30 | 0.503 | 0.811 | 0.544 | 0.395 | 0.523 | 0.594 |
| 2 (kept) | ×1.5 (`min_size=560, max_size=1860`) | 10 (after 8) | 16 | 4:55 | **0.579** | **0.850** | **0.669** | **0.459** | **0.617** | 0.598 |

- Upscaling helps small and medium (distant) objects most; large objects do not change.
- AP @.75 grows more than AP @.5: boxes get tighter, which matters for the fusion's 2D IoU feature.
- Run 1 had stopped improving; run 2's longer high-LR phase kept AP rising until epoch 6.
- On the Mac (16 GB unified memory) batch 32 did not fit (20 GB footprint, swap thrashing): training was moved
  to the CUDA machine (RTX 4090, 24 GB; run 2 peaks at 18.9 GB allocated).

## 2. Learned late fusion FL-n (simplified CLOCs)

L2 candidates before threshold × C2 2D detections (score ≥ 0.05). Each candidate is projected into the image;
one row per same-class 2D box with IoU > 0: `[2D IoU, 2D score, 3D score, distance / 70 m]`, or a single row
`[-1, -1, 3D score, distance]` if none. Shared MLP 4-16-32-16-1, max over the candidate's rows. Target: 3D IoU
≥ KITTI threshold with a same-class GT. 5 contiguous folds on the 300 holdout frames → out-of-fold scores →
per-class F1-best threshold; final net on all 300.

### 2.1 Holdout (out-of-fold F1, 300 frames)

| | Car | Pedestrian | Cyclist |
|---|---|---|---|
| Positives in the 300 frames | 433 | 38 | 24 |
| L2 alone (best threshold) | 0.559 | 0.395 | 0.667 |
| FL-n with C2 boxes | **0.592** | 0.390 | **0.677** |
| FL-n with un-tuned COCO boxes | 0.580 | 0.400 | 0.667 |
| Ceiling: perfect rescoring of L2's candidates | 0.683 | 0.476 | 0.772 |

- Rescoring cannot move boxes, so the ceiling is set by L2's candidate boxes. FL-n recovers about a quarter of
  the Car gap to it.
- Fine-tuning C2 adds about 0.01 Car F1 over plain COCO detections.
- Pedestrian and Cyclist have too few positives to learn from or to measure: ±0.01 is one or two objects.

### 2.2 Val

| | Car | Pedestrian | Cyclist |
|---|---|---|---|
| L2 | 0.571 | 0.358 | 0.423 |
| FL-n | **0.587** | 0.356 | **0.463** |

Car: false positives 1736 → 1250 for 49 fewer true positives (precision 0.689 → 0.752). Cyclist: true positives
164 → 188 for 12 more false positives. BEV F1 for Car drops slightly (0.678 → 0.671): the gain is from removing
L2's badly placed boxes, which only the strict 3D threshold punishes.

## 3. Attempts to improve FL-n (negative results)

All on the holdout, out-of-fold F1, 3 seeds (mean ± spread across seeds).

### 3.1 Deeper network

| MLP | Parameters | Car | Pedestrian | Cyclist | Train loss | Time (5 folds) |
|---|---|---|---|---|---|---|
| 4-16-32-16-1 (kept) | 1,169 | 0.594 | 0.389 | 0.674 | 0.049 | ~1.5 min |
| 4-16-32-64-128-64-32-16-1 | 21,937 | 0.591 | 0.394 | 0.670 | 0.047 | ~7 min |

Same F1 within noise, lower training loss only: 4 input features are a simple function that the small net already
learns, and 300 frames do not support more capacity.

### 3.2 Extra pair features

| Extra features | Car | Pedestrian | Cyclist |
|---|---|---|---|
| none (kept) | 0.593 ± 0.001 | 0.389 ± 0.001 | 0.674 ± 0.005 |
| + candidate class one-hot | 0.588 ± 0.002 | 0.394 ± 0.002 | 0.660 ± 0.013 |
| + all L2 class probabilities | 0.589 ± 0.004 | 0.399 ± 0.003 | 0.653 ± 0.010 |
| + log LiDAR point count | 0.596 ± 0.003 | 0.398 ± 0.004 | 0.674 ± 0.005 |
| + 2D box height, projected / 2D height ratio | 0.589 ± 0.005 | 0.397 ± 0.006 | 0.678 ± 0.000 |
| + cross-class pairs (with same-class flag) | 0.594 ± 0.002 | 0.390 ± 0.002 | 0.670 ± 0.005 |
| all except cross-class | 0.599 ± 0.001 | 0.392 ± 0.003 | 0.674 ± 0.019 |
| all | 0.599 ± 0.001 | 0.394 ± 0.003 | 0.656 ± 0.008 |
| class + points + height + ratio | 0.600 ± 0.001 | 0.396 ± 0.001 | 0.667 ± 0.009 |

Best gain about +0.007 Car, no set better on all three classes. Point count was best by mean F1 over classes
(0.556 vs 0.552), within seed noise.

### 3.3 Point-count model on val

| FL-n on val | Car | Pedestrian | Cyclist |
|---|---|---|---|
| base (kept) | 0.587 | 0.356 | 0.463 |
| + log point count | 0.585 | 0.372 | 0.430 |

No net gain: Pedestrian +0.016, Cyclist −0.033, following its threshold (0.05 → 0.28) chosen on 24 positives.
The base model was kept. Choosing between the two after seeing `val` is a mild look at the test set; the
selection rule should have required a holdout gain larger than the seed spread, which would have kept the base
model anyway.

**Conclusion of §3**: the fusion is limited by information in the features and by the 300 training frames, not
by network size. The feature code stays available (`train_clocs.py --features ...`).

## 4. Camera-only 3D baseline C2-s

Each C2 box lifted to 3D with SGBM stereo (project 2's settings, flying pixels removed): median depth of the
inner half of the box → box pixels within a class margin of that depth → surface points in LiDAR coordinates →
centre = median surface point pushed `offset` metres along the viewing ray, bottom from the 2D box's lower edge,
l / w / h = class means of `train[:-300]`. Lifting all 4069 frames: 49 s on 10 processes.

### 4.1 Choices on the holdout (F1, grid over offset 0–3 m × heading × threshold)

| Heading rule | Car | Pedestrian | Cyclist |
|---|---|---|---|
| parallel to our lane ("zero") | **0.085** | 0.251 | **0.188** |
| along the viewing ray | 0.057 | 0.251 | 0.111 |
| PCA of the surface points | 0.040 | **0.320** | 0.167 |

One heading for all classes picked PCA (Car 0.040); choosing per class doubled Car. Kept: Car 1.0 m / zero / 0.99,
Pedestrian 0 m / PCA / 0.99, Cyclist 0 m / zero / 0.98.

### 4.2 Car depth error (holdout, matched cars, error along the viewing ray)

| Range | Cars | Median error | Median abs. error | 90th percentile abs. error |
|---|---|---|---|---|
| 0–15 m | 212 | +0.15 m | 0.40 m | 1.04 m |
| 15–30 m | 367 | −0.31 m | 0.42 m | 1.40 m |
| 30–80 m | 296 | −1.55 m | 1.63 m | 3.26 m |

Stereo depth error grows with distance (it scales with Z²), and its bias changes sign with range, so no constant
offset fixes it. At IoU 0.7 this, plus guessed heading and average size, keeps Car near zero.

### 4.3 Val

| | Car 3D (BEV) | Pedestrian 3D (BEV) | Cyclist 3D (BEV) |
|---|---|---|---|
| C1 (classical stereo, project 2) | 0.109 (0.175) | 0.003 (0.005) | 0.012 (0.015) |
| C2-s | 0.080 (0.205) | **0.366 (0.466)** | 0.300 (0.350) |

A learned detector fixes the camera's detection problem (Pedestrian 0.003 → 0.366); depth remains the
bottleneck for cars. C2-s is the best of all methods on Pedestrian, while FL-n gains nothing there: the limit
is the fusion's 38 Pedestrian training examples, not the camera.

## 5. Final comparison on val

| Method | Car 3D (BEV) | Pedestrian 3D (BEV) | Cyclist 3D (BEV) |
|---|---|---|---|
| L1 | 0.301 (0.417) | 0.014 (0.027) | 0.053 (0.069) |
| C1 | 0.109 (0.175) | 0.003 (0.005) | 0.012 (0.015) |
| FE | 0.303 (0.381) | 0.015 (0.028) | 0.058 (0.073) |
| FM | 0.307 (0.419) | 0.012 (0.022) | 0.048 (0.061) |
| FL-c | 0.394 (0.521) | 0.030 (0.059) | 0.123 (0.157) |
| L2 | 0.571 (**0.678**) | 0.358 (0.389) | 0.423 (0.464) |
| C2-s | 0.080 (0.205) | **0.366 (0.466)** | 0.300 (0.350) |
| **FL-n** | **0.587** (0.671) | 0.356 (0.390) | **0.463 (0.503)** |

Runtime: do not use the `s/frame` column of `results/val/comparison.md`. It shows 0.00 for C2-s and FL-n because
their inputs are cached, and the project 2 values were measured with 10 frames running in parallel, which inflates
them. The real latencies are in §6.

## 6. Inference latency on the M4 MacBook

`scripts/benchmark_latency.py`: 100 random val frames (seed 0), one frame at a time, each pipeline alone in its
own process, 3 warm-up frames, files already in the OS cache, calibration loaded beforehand. Time runs from
reading the frame's files to the final detections. MacBook with Apple M4 (10 cores, 16 GB); the machine was not
fully idle (load average about 3 at the start). C2-s and FL-n are sums of separately measured parts, frame by frame.

| Pipeline | Mean (ms) | Std | Median | p95 | Max | FPS | CPU cores used | Peak RAM (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 407 | 79 | 397 | 537 | 638 | 2.46 | 1.0 | 289 |
| C1 | 495 | 100 | 509 | 627 | 910 | 2.02 | 1.1 | 384 |
| FE | 429 | 77 | 410 | 563 | 696 | 2.33 | 1.1 | 407 |
| FM | 452 | 77 | 442 | 580 | 662 | 2.21 | 1.1 | 404 |
| FL-c | 897 | 135 | 923 | 1090 | 1213 | 1.12 | 1.0 | 385 |
| L2 | 428 | 84 | 424 | 565 | 666 | 2.33 | 1.1 | 334 |
| C2, 2D only (MPS) | 289 | 5 | 289 | 293 | 334 | 3.45 | 0.1 | 674 + 2198 GPU |
| C2, 2D only (CPU) | 563 | 72 | 551 | 582 | 1135 | 1.78 | 2.3 | 2048 |
| C2-s (MPS) | 315 | 6 | 314 | 321 | 365 | 3.18 | 0.2 | as C2 + 438 |
| C2-s (CPU) | 588 | 72 | 576 | 610 | 1164 | 1.70 | 2.3 | as C2 + 438 |
| FL-n (MPS) | 719 | 85 | 713 | 856 | 961 | 1.39 | 0.7 | L2 + C2 + 253 |
| FL-n (CPU) | 992 | 107 | 979 | 1161 | 1583 | 1.01 | 1.8 | L2 + C2 + 253 |
| **FL-n ‖, L2 on CPU + C2 on GPU** | **445** | 94 | 434 | 616 | 769 | 2.25 | 1.2 | 960 + GPU |
| FL-n ‖, both on CPU | 726 | 43 | 715 | 804 | 894 | 1.38 | 2.7 | 2285 |
| FL-c ‖, L1 + C1 on two cores | 678 | 101 | 683 | 824 | 934 | 1.47 | 1.7 | 563 |
| FM ‖, floor + stereo cloud on two cores | 422 | 75 | 411 | 544 | 627 | 2.37 | 1.2 | 802 |

Rows marked ‖ are measured end to end with the two sensor branches running at the same time, in two processes:
the main one sends the frame to the side process, computes its own branch, waits for the other, then fuses. The
side process's L2 candidates equal the cached ones on 25 frames, and the live fused detections match the saved
run. FE has no ‖ row: its LiDAR branch before the merge is a 1 ms file read, so there is nothing to overlap.

Stage means (ms):

| Stage | Time | Used by |
|---|---:|---|
| LiDAR file read | 1 | L1, FE, FM, FL-c, L2, FL-n |
| Image read + PNG decode | 7 per image | C1, FE, FM, FL-c (two images), C2, C2-s (two), FL-n (one) |
| Classical proposals (RANSAC floor, BEV grid, clustering, boxes) | 406 | L1, L2, FL-n; similar in C1, FE, FM |
| PointNet on the proposals | 21 | L2, FL-n |
| SGBM disparity | 15 | C1, FE, FM, FL-c, C2-s |
| Faster R-CNN inference at ×1.5 scale | 281 MPS / 555 CPU | C2, C2-s, FL-n |
| Stereo lifting of the C2 boxes | 5 | C2-s |
| Fusion step (projection, pairs, MLP, thresholds) | 1 | FL-n |

Cost against gain (3D F1 on val):

| Comparison | Extra time | Time × | Δ Car | Δ Pedestrian | Δ Cyclist |
|---|---:|---:|---:|---:|---:|
| L2 vs L1 (learned decision on the same proposals) | +21 ms | 1.05 | +0.270 | +0.344 | +0.369 |
| FL-n (MPS) vs L2 | +290 ms | 1.68 | +0.016 | −0.002 | +0.040 |
| FL-n (CPU) vs L2 | +564 ms | 2.32 | +0.016 | −0.002 | +0.040 |
| **FL-n ‖ (L2 CPU, C2 GPU) vs L2** | **+16 ms** | **1.04** | +0.016 | −0.002 | +0.040 |
| FL-n ‖ (both on CPU) vs L2 | +298 ms | 1.70 | +0.016 | −0.002 | +0.040 |
| FL-c vs L1 | +489 ms | 2.20 | +0.093 | +0.016 | +0.070 |
| FL-c ‖ vs L1 | +271 ms | 1.67 | +0.093 | +0.016 | +0.070 |
| FE vs L1 | +21 ms | 1.05 | +0.002 | +0.001 | +0.005 |
| FM vs L1 | +45 ms | 1.11 | +0.006 | −0.001 | −0.005 |
| FM ‖ vs L1 | +14 ms | 1.04 | +0.006 | −0.001 | −0.005 |
| C2-s (MPS) vs C1 | −180 ms | 0.64 | −0.029 | +0.363 | +0.289 |

- The fusion logic itself is free (1 ms). What fusion costs is running a second detector: run one after the
  other, +290 ms on the GPU, i.e. 68% more time than L2, for +0.016 Car and +0.040 Cyclist.
- Run concurrently on different hardware, that cost almost disappears: with L2 on one CPU core and C2 on the
  GPU, FL-n takes 445 ms against L2's 428 ms (+16 ms, +4%). C2 finishes first (298 ms) and the pipeline waits
  145 ms for L2, so the LiDAR branch sets the latency. The price is hardware and memory, not time: about 1 GB of
  RAM plus 2.2 GB of GPU memory, against 0.3 GB for L2.
- Concurrency on the CPU alone helps much less. With C2 on the CPU, FL-n ‖ takes 726 ms: C2 slows from 563 to
  725 ms and L2 from 428 to 481 ms when they share the CPU. FL-c ‖ gains 219 ms (897 → 678) but both its branches
  slow down too (L1 407 → 555 ms, C1 495 → 591 ms). FM ‖ gains 30 ms.
- The best value by far is L2 over L1: 21 ms of PointNet for +0.27 to +0.37 F1.
- 95% of L2's time, and 56% of FL-n's, is the classical proposal stage inherited from L1 (single core, Python).
  It is the first thing to optimise, before either network.
- The camera branch is the steadier one: C2 on MPS varies by 5 ms, the LiDAR pipelines by about 80 ms with the
  scene (number of points and clusters).
- No pipeline is real-time for KITTI's 10 Hz sensors (100 ms per frame): the fastest, C2-s, runs at 3.2 FPS.

## 7. Power and energy on the M4 MacBook

`scripts/benchmark_latency.py --power`: 20 frames per pipeline (the first 20 of the latency sample), sensor
branches concurrent wherever possible, on battery. CPU and GPU power come from Apple's `powermetrics`
(`sudo powermetrics --samplers cpu_power,gpu_power -i 200`), averaged over each pipeline's time window; a 15 s
idle window gives the baseline (CPU 0.60 W, GPU 0.06 W). Energy per frame = average CPU + GPU power × time per
frame. These are the SoC's CPU and GPU only: memory, display and the rest of the laptop are not included. The GPU
figures agree with Apple's unprivileged energy counter within 5%.

| Pipeline | ms / frame | CPU (W) | GPU (W) | CPU + GPU (W) | Energy / frame (J) | Above idle (J) |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 411 | 6.67 | 0.05 | 6.72 | 2.76 | 2.49 |
| C1 | 512 | 6.96 | 0.06 | 7.02 | 3.60 | 3.26 |
| FE | 407 | 6.92 | 0.06 | 6.98 | 2.85 | 2.58 |
| FM ‖ (two CPU cores) | 426 | 6.91 | 0.05 | 6.97 | 2.97 | 2.69 |
| FL-c ‖ (two CPU cores) | 689 | 7.84 | 0.05 | 7.89 | 5.43 | 4.98 |
| L2 | 429 | 6.55 | 0.05 | 6.60 | 2.83 | 2.55 |
| C2, 2D only, on GPU | 313 | 1.68 | 14.12 | 15.80 | 4.95 | 4.74 |
| C2, 2D only, on CPU | 536 | 9.71 | 0.01 | 9.71 | 5.21 | 4.86 |
| C2-s ‖ (stereo on CPU, C2 on GPU) | 289 | 1.59 | 14.59 | 16.18 | 4.68 | 4.49 |
| **FL-n ‖ (L2 on CPU, C2 on GPU)** | 427 | 6.23 | 9.47 | 15.70 | **6.70** | 6.42 |

Energy against gain (3D F1 on val):

| Comparison | Extra energy / frame | Energy × | Δ Car | Δ Pedestrian | Δ Cyclist |
|---|---:|---:|---:|---:|---:|
| L2 vs L1 | +0.07 J | 1.03 | +0.270 | +0.344 | +0.369 |
| **FL-n ‖ vs L2** | **+3.87 J** | **2.37** | +0.016 | −0.002 | +0.040 |
| FL-c ‖ vs L1 | +2.67 J | 1.97 | +0.093 | +0.016 | +0.070 |
| FE vs L1 | +0.08 J | 1.03 | +0.002 | +0.001 | +0.005 |
| FM ‖ vs L1 | +0.21 J | 1.08 | +0.006 | −0.001 | −0.005 |
| C2-s ‖ vs C1 | +1.08 J | 1.30 | −0.029 | +0.363 | +0.289 |

- Concurrency hides the fusion's cost in time, not in energy. FL-n ‖ has L2's latency (427 vs 429 ms) but draws
  15.7 W against 6.6 W and uses 2.4 times the energy per frame, for +0.016 Car and +0.040 Cyclist.
- The camera network is the expensive part: one C2 inference costs about 5 J, more than a whole L2 frame (2.8 J),
  on the GPU or on the CPU. The GPU is faster (313 vs 536 ms) but draws more (15.8 vs 9.7 W), so the energy per
  frame is nearly the same.
- L2 over L1 is almost free in energy too (+3%) for the largest F1 gain of all.
- In FL-n ‖ the GPU averages 9.5 W, not 14 W, because it sits idle about 130 ms per frame waiting for L2.
- C2-s ‖ is the cheapest way to detect pedestrians well: 4.7 J per frame for the best Pedestrian F1, against
  6.7 J for FL-n.
- The single-core classical pipelines all draw about 6.7–7 W: one performance core at full speed.

## Reproduce

```bash
.venv/bin/python src/methods/fine_tuning.py            # C2, on the CUDA machine
.venv/bin/python src/methods/cache_c2.py               # C2 detections on val + train[-300:]
.venv/bin/python src/methods/train_clocs.py            # FL-n: cross-validated thresholds + final net
.venv/bin/python src/methods/camera_stereo.py          # C2-s: stereo lifting
.venv/bin/python src/methods/fit_c2s.py                # C2-s: offset, heading, thresholds
.venv/bin/python src/run.py FL-n
.venv/bin/python src/run.py C2-s
.venv/bin/python src/compare.py
.venv/bin/python scripts/benchmark_latency.py          # latency table, needs ../kitticlassicfusion
.venv/bin/python scripts/benchmark_latency.py --power  # power table, with sudo powermetrics running (see --help)
```
