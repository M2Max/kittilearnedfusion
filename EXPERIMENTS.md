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

Runtime: the `s/frame` column of `results/val/comparison.md` shows 0.00 for C2-s and FL-n because their inputs are
cached; the real cost is L2 (~1.05 s per frame on the Mac) plus C2 (~0.33 s on the GPU), plus SGBM for C2-s.

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
```
