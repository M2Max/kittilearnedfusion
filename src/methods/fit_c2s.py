import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import numpy as np

from camera_stereo import PARAMS_PATH, load_lifted, make_boxes
from config import CLASSES, IOU_THRESH
from evaluation.iou import iou_3d
from evaluation.result_analysis import class_indices, iou_table, prf, score_frame
from kitti.data import frame_ids, load_calib, load_labels
from kitti.ground_truth import gt_boxes

OFFSETS = np.round(np.arange(0.0, 3.01, 0.25), 2)           # metri dalla superficie visibile al centro
HEADINGS = ["zero", "ray", "pca"]
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.01), 2)


def mean_sizes():
    # l, w, h medi per classe dalle etichette di train[:-300] (le stesse che ha visto C2)
    sizes = {c: [] for c in CLASSES}
    for i in frame_ids("train")[:-300]:
        for o in load_labels(i):
            if o["type"] in sizes:
                h, w, l = o["hwl"]
                sizes[o["type"]].append([l, w, h])
    return {c: np.mean(v, axis=0).round(3).tolist() for c, v in sizes.items()}

def holdout_frame(i):
    gt_b, gt_c, gt_ign = gt_boxes(i, load_calib(i))
    return {"frame": i, "lifted": load_lifted(i), "gt_boxes": gt_b, "gt_classes": gt_c, "gt_ignored": gt_ign}

def best_f1(args):
    # per una classe e una combinazione (offset, orientamento): F1 migliore sulle soglie, sui frame di controllo
    frames, c, offset, mode, size = args
    params = {"offset": {k: offset for k in CLASSES}, "size": size, "heading": {k: mode for k in CLASSES}}
    tables = []
    for f in frames:
        boxes = make_boxes(f["lifted"], params)
        rec = {"det_classes": list(f["lifted"]["labels"]), "gt_classes": f["gt_classes"], "gt_ignored": f["gt_ignored"]}
        det, valid, ignored = class_indices(rec, c)
        tables.append((f["lifted"]["scores"][det], valid, iou_table(boxes[det], f["gt_boxes"][valid], iou_3d),
                       iou_table(boxes[det], f["gt_boxes"][ignored], iou_3d)))
    best = (-1.0, None)
    for t in THRESHOLDS:
        tp = fp = fn = 0
        for s, valid, iou_v, iou_i in tables:
            rows = np.flatnonzero(s >= t)
            pairs, fps, fns = score_frame({"dets": rows, "valid": valid, "iou_v": iou_v[rows], "iou_i": iou_i[rows]},
                                          IOU_THRESH[c])
            tp, fp, fn = tp + len(pairs), fp + len(fps), fn + len(fns)
        f1 = prf(tp, fp, fn)[2]
        if f1 > best[0]:
            best = (f1, float(t))
    return c, float(offset), mode, best[0], best[1]


def main():
    size = mean_sizes()
    print("misure medie l, w, h:", size)
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        frames = list(pool.map(holdout_frame, frame_ids("train")[-300:], chunksize=16))
        jobs = [(frames, c, o, m, size) for c, o, m in itertools.product(CLASSES, OFFSETS, HEADINGS)]
        results = list(pool.map(best_f1, jobs))

    params = {"size": size, "offset": {}, "heading": {}, "threshold": {}}
    for c in CLASSES:                                       # offset, orientamento e soglia insieme, per classe
        _, offset, mode, f1, t = max((r for r in results if r[0] == c), key=lambda r: r[3])
        params["offset"][c], params["heading"][c], params["threshold"][c] = offset, mode, t
        print(f"  {c:10s} offset {offset:.2f} m, orientamento {mode}, soglia {t:.2f} -> F1 {f1:.3f} "
              f"sui 300 frame di controllo")
    PARAMS_PATH.write_text(json.dumps(params, indent=2))
    print(f"salvati in {PARAMS_PATH}")


if __name__ == "__main__":
    main()
