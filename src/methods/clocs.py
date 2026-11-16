import pickle
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]                # src/ (kitti, config, evaluation)

import numpy as np
import torch
from torch import nn

from config import CLASSES, DATA, RESULTS
from evaluation.iou import box_corners_bev
from kitti.transforms import lidar_to_rectcam

CANDIDATES = RESULTS / "cache" / "l2_candidates"            # candidati di L2 prima della soglia
DET2D = {                                                   # cartella della cache 2D, etichetta -> classe KITTI
    "c2": (RESULTS / "cache" / "c2_detector2d", {c: c for c in CLASSES}),
    "coco": (RESULTS / "cache" / "coco_detector2d", {"car": "Car", "person": "Pedestrian", "bicycle": "Cyclist"}),
}
MODEL_PATH = RESULTS / "cache" / "clocs.pt"
MAX_DIST = 70.0                                             # normalizza la distanza a circa 0-1
EMPTY = -1.0                                                # IoU e score 2D di un candidato senza box 2D sovrapposti
MIN_DEPTH = 0.1                                             # angoli più vicini di così alla camera non si proiettano


# ================================================================ dati

def load_candidates(split):
    # split: "val" o "train_holdout"; frame -> {boxes, probs, n_points}
    with open(CANDIDATES / f"{split}.pkl", "rb") as f:
        data = pickle.load(f)
    return {fr["frame"]: fr for fr in data["frames"]}, data["classes"]

def image_size(i):
    # larghezza e altezza dall'intestazione del PNG, senza decodificare l'immagine
    with open(DATA / "image_2" / f"{i:06d}.png", "rb") as f:
        return struct.unpack(">II", f.read(24)[16:24])

def load_det2d(i, source):
    folder, names = DET2D[source]
    z = np.load(folder / f"{i:06d}.npz")
    keep = np.array([l in names for l in z["labels"]], dtype=bool)   # COCO ha anche classi che non servono
    return z["boxes"][keep].reshape(-1, 4), np.array([names[l] for l in z["labels"][keep]]), z["scores"][keep]


# ================================================================ geometria

def corners_3d(box):
    bev = box_corners_bev(box)
    z0, z1 = box[2], box[2] + box[5]
    return np.vstack([np.c_[bev, np.full(4, z0)], np.c_[bev, np.full(4, z1)]])   # 4 sotto, poi 4 sopra

def project_boxes(boxes, calib, size):
    # rettangolo che contiene gli 8 angoli proiettati in image_2, tagliato ai bordi; NaN se fuori dalla foto
    w, h = size
    P2 = calib["P2"].reshape(3, 4)
    out = np.full((len(boxes), 4), np.nan)
    for k, box in enumerate(boxes):
        cam = lidar_to_rectcam(calib, corners_3d(box))
        cam = cam[cam[:, 2] > MIN_DEPTH]                    # solo gli angoli davanti alla camera
        if len(cam) == 0:
            continue
        uvw = np.c_[cam, np.ones(len(cam))] @ P2.T
        uv = uvw[:, :2] / uvw[:, 2:]
        u0, v0 = np.clip(uv.min(axis=0), 0, [w, h])
        u1, v1 = np.clip(uv.max(axis=0), 0, [w, h])
        if u1 > u0 and v1 > v0:
            out[k] = [u0, v0, u1, v1]
    return out

def iou_2d(a, b):
    # tabella (len(a), len(b)) di IoU fra rettangoli [u0, v0, u1, v1]; righe NaN -> 0
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(hi - lo, 0, None), axis=2)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.nan_to_num(inter / np.where(union > 0, union, 1))


# ================================================================ feature delle coppie

FEATURES = ["class", "probs", "points", "height", "ratio", "cross"]   # opzionali, oltre alle 4 di base
N_COLUMNS = {"class": 3, "probs": 4, "points": 1, "height": 1, "ratio": 1, "cross": 1}

def decode_candidates(cand, cand_classes):
    # classe = oggetto più probabile (0 è lo sfondo), score 3D = sua probabilità, come in L2
    finite = np.isfinite(cand["boxes"]).all(axis=1)         # 2 candidati di val hanno box non finiti
    boxes, probs = cand["boxes"][finite], cand["probs"][finite]
    best = 1 + np.argmax(probs[:, 1:], axis=1)
    return {"boxes": boxes, "classes": [cand_classes[k] for k in best], "s3d": probs[np.arange(len(best)), best],
            "probs": probs, "n_points": cand["n_points"][finite]}

def n_inputs(features):
    return 4 + sum(N_COLUMNS[f] for f in features)

def frame_pairs(cand, calib, size, det2d, features=()):
    # per ogni candidato le righe [IoU 2D, score 2D, score 3D, distanza, opzionali]: una per box 2D che si sovrappone
    # al candidato proiettato (della stessa classe, o di ogni classe con "cross"), altrimenti una sola riga con
    # IoU e score 2D a -1
    boxes, classes, s3d = cand["boxes"], np.array(cand["classes"]), cand["s3d"]
    d_boxes, d_classes, d_scores = det2d
    proj = project_boxes(boxes, calib, size)
    ious = iou_2d(proj, d_boxes)
    same = classes[:, None] == d_classes[None, :]
    if "cross" not in features:
        ious *= same                                        # coppie solo fra classi uguali
    dist = np.hypot(boxes[:, 0], boxes[:, 1]) / MAX_DIST
    d_height = d_boxes[:, 3] - d_boxes[:, 1]
    p_height = proj[:, 3] - proj[:, 1]
    pairs = []
    for k in range(len(boxes)):
        j = np.flatnonzero(ious[k] > 0)
        n = max(len(j), 1)
        cols = [np.c_[ious[k, j], d_scores[j]] if len(j) else np.full((1, 2), EMPTY),
                np.full((n, 1), s3d[k]), np.full((n, 1), dist[k])]
        if "class" in features:
            cols.append(np.tile([classes[k] == c for c in CLASSES], (n, 1)))   # stessa rete, regole diverse per classe
        if "probs" in features:
            cols.append(np.tile(cand["probs"][k], (n, 1)))  # tutte le probabilità di L2, sfondo compreso
        if "points" in features:
            cols.append(np.full((n, 1), np.log1p(cand["n_points"][k]) / 7))   # pochi punti = lontano o coperto
        if "height" in features:
            cols.append((d_height[j] / 100)[:, None] if len(j) else np.zeros((1, 1)))   # altezza del box 2D
        if "ratio" in features:                             # proiezione molto più grande o piccola = profondità sbagliata
            r = np.log(p_height[k] / d_height[j]).clip(-2, 2)[:, None] if len(j) else np.zeros((1, 1))
            cols.append(r)
        if "cross" in features:
            cols.append(same[k, j][:, None] if len(j) else np.zeros((1, 1)))   # il box 2D è della stessa classe?
        pairs.append(np.hstack(cols).astype(np.float32))
    return pairs

def pad(pairs):
    # lista di (p_k, d) -> tensore (N, P, d) e maschera (N, P) delle righe vere
    n, p, d = len(pairs), max(len(x) for x in pairs), pairs[0].shape[1]
    x = np.zeros((n, p, d), dtype=np.float32)
    mask = np.zeros((n, p), dtype=bool)
    for k, rows in enumerate(pairs):
        x[k, :len(rows)] = rows
        mask[k, :len(rows)] = True
    return torch.from_numpy(x), torch.from_numpy(mask)


# ================================================================ rete

class Clocs(nn.Module):
    def __init__(self, n_in=4):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))   # stessa MLP per ogni coppia

    def forward(self, x, mask):
        logits = self.mlp(x).squeeze(-1)                    # (N, P): un logit per coppia
        return logits.masked_fill(~mask, float("-inf")).max(dim=1).values   # max sulle coppie del candidato

def fused_scores(model, pairs):
    if not pairs:
        return np.zeros(0)
    with torch.inference_mode():
        return torch.sigmoid(model(*pad(pairs))).numpy()


# ================================================================ detector per run.py

STATE = None

def get_state():
    global STATE
    if STATE is None:
        torch.set_num_threads(1)                            # run.py lancia già un processo per core
        saved = torch.load(MODEL_PATH, map_location="cpu")
        features = saved.get("features", [])
        model = Clocs(n_inputs(features))
        model.load_state_dict(saved["weights"])
        cands, cand_classes = load_candidates("val")
        STATE = model.eval(), saved["thresholds"], saved["det2d"], features, cands, cand_classes
    return STATE

def detect(i, calib):
    model, thresholds, source, features, cands, cand_classes = get_state()
    cand = decode_candidates(cands[i], cand_classes)
    boxes, classes = cand["boxes"], cand["classes"]
    if len(boxes) == 0:
        return np.zeros((0, 7)), [], np.zeros(0)
    scores = fused_scores(model, frame_pairs(cand, calib, image_size(i), load_det2d(i, source), features))
    keep = [k for k in range(len(boxes)) if scores[k] >= thresholds[classes[k]]]
    return boxes[keep].reshape(-1, 7), [classes[k] for k in keep], scores[keep]
