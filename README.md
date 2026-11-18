# 3rd Project Report

**KITTI LiDAR and Camera learned late fusion**

Maximiliano Mamone

## Project Objective

Moving from classical to learned fusion on the KITTI 3D Object Detection benchmark: fine-tuning a camera
detector, fusing its decisions with those of the LiDAR network of the previous project through a small learned
model, and comparing the result with the previous methods on the same frames and with the same metrics, adding
the cost in time and energy.

## Project Overview

There are two decision makers: C2, a Faster R-CNN fine-tuned on the KITTI 2D boxes, and L2, the PointNet on the
L1 clusters of the previous project, reused as is. A simplified CLOCs network (FL-n) then re-scores the 3D
candidates of L2 using the 2D boxes of C2.

The last 300 frames of train were never seen by either detector, so they are used to train the fusion network and
to choose every threshold. C2 is fine-tuned on the other 3412 train frames (200 of them only to monitor the
training) and val (3769 frames) is used only for the final numbers.

## Camera detector: fine-tuned Faster R-CNN (C2)

I started from the torchvision Faster R-CNN (ResNet-50 FPN, COCO weights) and replaced its box predictor with a
new one with 4 outputs (background, Car, Pedestrian, Cyclist); Van and DontCare boxes are dropped. Training
used SGD (learning rate 0.005, momentum 0.9, weight decay 5·10<sup>−4</sup>) with horizontal flips, on an
RTX 4090.

A first run of 5 epochs at the default input size reached AP@0.5 = 0.811 on the monitoring frames, but only 0.395
AP on small objects: the model resizes a KITTI image to 1333 × 402 px and its smallest anchors are 32 px, larger
than a distant car. Upscaling the input by 1.5 (1860 × 560 px) and training for 10 epochs (learning rate drop after
8, about 5 minutes per epoch) gave AP@0.5 = 0.850 and AP@[0.5:0.95] from 0.503 to 0.579, with small objects at
0.459 and tighter boxes (AP@0.75 from 0.544 to 0.669).

## Learned late fusion (FL-n)

The candidates of L2 are taken before any threshold (about 25 per frame). Each one is projected into image 2 (its
8 corners through P2, then the enclosing rectangle) and paired with every C2 box of the same class that overlaps
it. A pair is described by four numbers: the 2D IoU, the 2D score, the 3D score and the distance; a candidate with
no overlapping box gets a single row with IoU and 2D score set to −1. The same small MLP (4→16→32→16→1)
scores every pair and the candidate keeps its best one:

<p align="center"><i>s<sub>i</sub> = σ( max<sub>j</sub> MLP( IoU<sub>ij</sub>, s<sup>2D</sup><sub>j</sub>, s<sup>3D</sup><sub>i</sub>, d<sub>i</sub> ) )</i></p>

The target is 1 when the candidate reaches the KITTI 3D IoU threshold with a ground truth box of its class (495
positives out of 8835 candidates in the 300 frames). The per-class thresholds come from a 5-fold cross-validation
on contiguous folds, so that every frame is scored by a network that has not seen it; the final network is then
trained on all 300 frames.

## Camera only in 3D (C2-s)

To have a camera-only term of comparison with a modern detector, each C2 box is lifted to 3D with the stereo
depth of the previous project: median depth of the inner half of the box, centre pushed back along the viewing
ray, bottom from the lower edge of the 2D box and size equal to the class mean. Offset, heading rule and
threshold are chosen per class on the 300 frames.

## Evaluation

Same as the previous projects (greedy matching by 3D IoU, thresholds 0.7 Car and 0.5 Pedestrian/Cyclist,
Moderate subset, Van and Person_sitting ignored). In addition I measured, on a MacBook with an M4, the latency
of every pipeline on 100 random val frames and the CPU + GPU power on 20 of them, one frame at a time.

## Results

| Method | Car TP | Car FP | Car P | Car R | Car F1 | Ped F1 | Cyc F1 | ms/frame | J/frame |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 LiDAR | 3375 | 11179 | 0.232 | 0.429 | 0.301 | 0.014 | 0.053 | 407 | 2.76 |
| FL-c late | 3099 | 4747 | 0.395 | 0.394 | 0.394 | 0.030 | 0.123 | 678 | 5.43 |
| L2 PointNet | 3841 | 1736 | 0.689 | 0.488 | 0.571 | 0.358 | 0.423 | 428 | 2.83 |
| C2-s camera | 769 | 10625 | 0.067 | 0.098 | 0.080 | 0.366 | 0.300 | 315 | 4.68 |
| FL-n learned | 3792 | 1250 | 0.752 | 0.482 | 0.587 | 0.356 | 0.463 | 445 | 6.70 |

<p align="center"><sub>Run on the 3769 val frames, 3D IoU, Moderate difficulty. Time and energy on the M4;<br>FL-c and FL-n with their two branches running together (C2-s too, for the energy).</sub></p>

The learned fusion is the best method on cars and cyclists. For cars the camera mostly removes false positives
(28% fewer, precision from 0.689 to 0.752); for cyclists it recovers objects (24 more true positives). The gain is
small because re-scoring cannot move a box: even a perfect re-scoring of the L2 candidates would stop at Car F1
0.68 on the 300 frames (L2 0.56, fusion 0.59). Pedestrians do not improve, with only 38 positive examples to
learn from; still C2-s is the best pedestrian method, so the information is in the camera. A deeper MLP (22k
parameters instead of 1.2k) and nine sets of extra pair features gave the same F1 within the noise between seeds:
the limit is the data and the candidates, not the network.

The fusion step itself costs 1 ms: what fusion costs is the second detector. One after the other the two take
719 ms against 428 ms of L2 alone; with L2 on the CPU and C2 on the GPU at the same time the latency is 445 ms
(+4%), but the energy per frame is 2.4 times that of L2. In comparison, the PointNet of L2 adds 21 ms and 3% of
energy to L1 for by far the largest gain, and 95% of the time of L2 is still the classical proposal stage.

Limits: one operating point per method, Pedestrian and Cyclist thresholds chosen on 38 and 24 positives, fusion
bounded by the candidates of L2, and power measured only for the CPU and GPU of the chip.

## AI Use Disclosure

An AI assistant was used above all as a tutor, to understand in depth the concepts and intuitions behind
the methods (fine-tuning a detector pre-trained on COCO, anchors and input scale, the AP metrics, projecting
3D boxes into the image, cross-validation) and the CLOCs architecture. The code is a mix of components carried
over from the second project and additional parts developed by myself with AI assistance, mainly for debugging
and performance-improvement purposes.
