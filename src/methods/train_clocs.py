import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import numpy as np
import torch

from clocs import (DET2D, FEATURES, MODEL_PATH, Clocs, decode_candidates, frame_pairs, fused_scores, image_size,
                   load_candidates, load_det2d, pad)
from config import CLASSES, IOU_THRESH, NEIGHBOR
from evaluation.iou import iou_3d
from evaluation.result_analysis import class_indices, iou_table, prf, score_frame
from kitti.data import load_calib
from kitti.ground_truth import gt_boxes

N_FOLDS = 5
EPOCHS = 1000                                               # full batch: ogni epoca è un passo
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.01), 2)


# ================================================================ dati di training

def frame_data(i, cand, cand_classes, source, features=()):
    calib = load_calib(i)
    dec = decode_candidates(cand, cand_classes)
    boxes, classes, s3d = dec["boxes"], dec["classes"], dec["s3d"]
    gt_b, gt_c, gt_ign = gt_boxes(i, calib)
    ious = iou_table(boxes, gt_b, iou_3d)                   # candidati x GT, una volta sola

    target = np.zeros(len(boxes), dtype=np.float32)
    weight = np.ones(len(boxes), dtype=np.float32)
    rec = {"det_classes": classes, "gt_classes": gt_c, "gt_ignored": gt_ign}
    per_class = {}
    for c in CLASSES:
        det, valid, ignored = class_indices(rec, c)
        same = [j for j, k in enumerate(gt_c) if k == c]
        near = [j for j, k in enumerate(gt_c) if k == NEIGHBOR[c]]
        for d in det:
            if len(same) and ious[d, same].max() >= IOU_THRESH[c]:
                target[d] = 1                               # vero oggetto della sua classe, anche se fuori Moderate
            elif len(near) and ious[d, near].max() >= IOU_THRESH[c]:
                weight[d] = 0                               # su un Van o un Person_sitting: né premio né errore
        per_class[c] = {"det": det, "valid": valid,
                        "iou_v": ious[np.ix_(det, valid)], "iou_i": ious[np.ix_(det, ignored)]}

    pairs = frame_pairs(dec, calib, image_size(i), load_det2d(i, source), features)
    return {"frame": i, "pairs": pairs, "target": target, "weight": weight, "s3d": s3d, "per_class": per_class}


# ================================================================ training e soglie

def train_model(frames, seed=0):
    torch.manual_seed(seed)
    pairs = [p for f in frames for p in f["pairs"]]
    x, mask = pad(pairs)
    y = torch.from_numpy(np.concatenate([f["target"] for f in frames]))
    w = torch.from_numpy(np.concatenate([f["weight"] for f in frames]))
    model = Clocs(x.shape[2])                               # tante entrate quante colonne
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    for epoch in range(EPOCHS):
        opt.zero_grad()
        loss = (loss_fn(model(x, mask), y) * w).sum() / w.sum()
        loss.backward()
        opt.step()
    return model.eval(), loss.item()

def f1_at(frames, scores, c, t):
    # F1 della classe c tenendo i candidati con score >= t, con le regole di matching della valutazione
    tp = fp = fn = 0
    for f, s in zip(frames, scores):
        pc = f["per_class"][c]
        rows = np.flatnonzero(s[pc["det"]] >= t)
        pairs, fps, fns = score_frame({"dets": rows, "valid": pc["valid"], "iou_v": pc["iou_v"][rows],
                                       "iou_i": pc["iou_i"][rows]}, IOU_THRESH[c])
        tp, fp, fn = tp + len(pairs), fp + len(fps), fn + len(fns)
    return prf(tp, fp, fn)

def best_thresholds(frames, scores):
    out = {}
    for c in CLASSES:
        f1, t = max((f1_at(frames, scores, c, t)[2], float(t)) for t in THRESHOLDS)
        out[c] = (t, f1)
    return out

def cross_val_scores(frames):
    # score fusi di ogni frame da una rete che non l'ha visto; fold contigui, per non spezzare le sequenze
    scores = [None] * len(frames)
    for fold in np.array_split(np.arange(len(frames)), N_FOLDS):
        held = set(fold.tolist())
        model, loss = train_model([f for k, f in enumerate(frames) if k not in held])
        for k in fold:
            scores[k] = fused_scores(model, frames[k]["pairs"])
        print(f"  fold {fold[0]}-{fold[-1]}: loss {loss:.4f}", flush=True)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det2d", choices=sorted(DET2D), default="c2", help="cache dei box 2D (coco = fallback)")
    ap.add_argument("--features", nargs="*", choices=FEATURES, default=[], help="feature oltre alle 4 di base")
    args = ap.parse_args()

    cands, cand_classes = load_candidates("train_holdout")
    frames = [frame_data(i, cands[i], cand_classes, args.det2d, args.features) for i in sorted(cands)]
    n = sum(len(f["target"]) for f in frames)
    pos = sum(f["target"].sum() for f in frames)
    print(f"{len(frames)} frame, {n} candidati, {int(pos)} positivi, 2D da {args.det2d}, feature extra {args.features}")

    oof = cross_val_scores(frames)
    fused = best_thresholds(frames, oof)
    lidar = best_thresholds(frames, [f["s3d"] for f in frames])   # L2 da solo, stessa scelta di soglia
    print("\nF1 sui 300 frame (fusione: score fuori fold)")
    for c in CLASSES:
        print(f"  {c:10s}  L2 {lidar[c][1]:.3f} (soglia {lidar[c][0]:.2f})   "
              f"fusione {fused[c][1]:.3f} (soglia {fused[c][0]:.2f})")

    model, loss = train_model(frames)                       # rete finale su tutti i 300 frame
    torch.save({"weights": model.state_dict(), "thresholds": {c: fused[c][0] for c in CLASSES},
                "det2d": args.det2d, "features": args.features}, MODEL_PATH)
    print(f"\nrete finale: loss {loss:.4f}, salvata in {MODEL_PATH}")


if __name__ == "__main__":
    main()
