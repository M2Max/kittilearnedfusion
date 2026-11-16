import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]                # src/ (kitti, config, stereo)

import cv2
import numpy as np

from config import CLASSES, RESULTS
from kitti.data import frame_ids, load_calib, load_stereo_pair
from kitti.transforms import rectcam_to_lidar
from stereo.disparity import compute_disparity
from stereo.pseudo_lidar import camera_offset, disparity_to_depth, remove_flying_pixels, stereo_params

C2_CACHE = RESULTS / "cache" / "c2_detector2d"
LIFTED = RESULTS / "cache" / "c2s_lifted"                   # per frame: box 2D di C2 con la loro misura stereo
PARAMS_PATH = RESULTS / "cache" / "c2s_params.json"         # misure medie, offset, orientamento, soglie
MAX_JUMP = 0.05                                             # pixel volanti, come C1
INNER = 0.25                                                # profondità solo dal centro del box: tolgo un quarto per lato
MIN_PIXELS = 10
SURFACE_MARGIN = {"Car": 2.5, "Pedestrian": 1.0, "Cyclist": 1.5}   # pixel del box tenuti attorno alla mediana (m)
MIN_PCA_POINTS = 20


# ================================================================ misura stereo di ogni box 2D

def backproject(u, v, z, calib):
    # pixel di image_2 alla profondità z -> punti in camera 0 rettificata (pinhole al contrario)
    P2 = calib["P2"].reshape(3, 4)
    f, cu, cv = P2[0, 0], P2[0, 2], P2[1, 2]
    return np.stack([(u - cu) * z / f, (v - cv) * z / f, z], axis=-1) - camera_offset(P2)

def measure_box(box, depth, calib, cls):
    # profondità della superficie visibile, punto mediano della superficie e asse principale in BEV (in LiDAR)
    u0, v0, u1, v1 = np.round(box).astype(int)
    du, dv = int((u1 - u0) * INNER), int((v1 - v0) * INNER)
    inner = depth[v0 + dv:v1 - dv, u0 + du:u1 - du]
    inner = inner[np.isfinite(inner)]
    if len(inner) < MIN_PIXELS:
        return None
    z = float(np.median(inner))

    patch = depth[v0:v1, u0:u1]
    vv, uu = np.nonzero(np.abs(patch - z) < SURFACE_MARGIN[cls])   # niente sfondo né primo piano
    if len(vv) < MIN_PIXELS:
        return None
    pts = rectcam_to_lidar(calib, backproject(uu + u0, vv + v0, patch[vv, uu], calib))
    surface = np.median(pts, axis=0)
    bottom = rectcam_to_lidar(calib, backproject(np.array([(u0 + u1) / 2]), np.array([v1]), np.array([z]), calib))[0]   # fondo del box 2D

    psi_pca = np.nan
    if len(pts) >= MIN_PCA_POINTS:
        xy = pts[:, :2] - pts[:, :2].mean(axis=0)
        _, vecs = np.linalg.eigh(xy.T @ xy)
        psi_pca = float(np.arctan2(vecs[1, -1], vecs[0, -1]))   # direzione di massima estensione
    return z, surface, bottom[2], psi_pca, len(pts)

def lift_boxes(det2d, left, right, calib, depth=None):
    # box 2D di C2 (boxes, labels, scores) + coppia stereo (o profondità già pronta) -> misure stereo dei box
    boxes2d, labels2d, scores2d = det2d
    out = {"labels": [], "scores": [], "boxes2d": [], "depth": [], "surface": [], "z_bottom": [], "psi_pca": [],
           "n_points": []}
    if len(boxes2d):
        if depth is None:
            f, _, _, baseline = stereo_params(calib)
            depth = remove_flying_pixels(disparity_to_depth(compute_disparity(left, right), f, baseline), MAX_JUMP)
        h, w = depth.shape
        for box, cls, s in zip(boxes2d, labels2d, scores2d):
            box = np.clip(box, 0, [w, h, w, h])
            m = measure_box(box, depth, calib, str(cls))
            if m is None:
                continue                                    # niente stereo affidabile: il box non diventa 3D
            for k, v in zip(["labels", "scores", "boxes2d", "depth", "surface", "z_bottom", "psi_pca", "n_points"],
                            [str(cls), s, box, *m]):
                out[k].append(v)
    return {"labels": np.array(out["labels"], dtype="<U10"), "scores": np.array(out["scores"], dtype=np.float32),
            "boxes2d": np.array(out["boxes2d"], dtype=np.float32).reshape(-1, 4), "depth": np.array(out["depth"]),
            "surface": np.array(out["surface"]).reshape(-1, 3), "z_bottom": np.array(out["z_bottom"]),
            "psi_pca": np.array(out["psi_pca"]), "n_points": np.array(out["n_points"])}

def lift_frame(i):
    z2d = np.load(C2_CACHE / f"{i:06d}.npz")
    left, right = load_stereo_pair(i) if len(z2d["boxes"]) else (None, None)
    np.savez(LIFTED / f"{i:06d}.npz", **lift_boxes((z2d["boxes"], z2d["labels"], z2d["scores"]), left, right,
                                                    load_calib(i)))
    return i

def single_thread():
    cv2.setNumThreads(1)                                    # un processo per core: niente thread annidati


# ================================================================ dai box misurati ai box 3D

def load_lifted(i):
    z = np.load(LIFTED / f"{i:06d}.npz")
    return {k: z[k] for k in z.files}

def heading(lifted, centers, mode):
    if mode == "zero":
        return np.zeros(len(centers))                       # parallelo alla nostra corsia
    if mode == "ray":
        return np.arctan2(centers[:, 1], centers[:, 0])     # lungo il raggio di vista
    psi = lifted["psi_pca"].copy()                          # "pca", con il raggio dove i punti sono pochi
    bad = ~np.isfinite(psi)
    psi[bad] = np.arctan2(centers[bad, 1], centers[bad, 0])
    return psi

def make_boxes(lifted, params):
    # superficie visibile -> centro spostato lungo il raggio di offset[c]; misure medie della classe
    n = len(lifted["labels"])
    if n == 0:
        return np.zeros((0, 7))
    cls = lifted["labels"]
    offset = np.array([params["offset"][c] for c in cls])
    ray = lifted["surface"][:, :2] / np.linalg.norm(lifted["surface"][:, :2], axis=1, keepdims=True)
    centers = lifted["surface"][:, :2] + offset[:, None] * ray
    lwh = np.array([params["size"][c] for c in cls])
    psi = np.zeros(n)
    for mode in set(params["heading"].values()):            # orientamento scelto per classe
        rows = np.array([params["heading"][c] == mode for c in cls])
        psi[rows] = heading(lifted, centers, mode)[rows]
    return np.c_[centers, lifted["z_bottom"], lwh, psi]


# ================================================================ detector per run.py

PARAMS = None

def detect(i, calib):
    global PARAMS
    if PARAMS is None:
        PARAMS = json.loads(PARAMS_PATH.read_text())
    lifted = load_lifted(i)
    boxes = make_boxes(lifted, PARAMS)
    keep = [k for k, c in enumerate(lifted["labels"]) if lifted["scores"][k] >= PARAMS["threshold"][c]]
    return boxes[keep].reshape(-1, 7), [str(lifted["labels"][k]) for k in keep], lifted["scores"][keep]


def main():
    frames = frame_ids("val") + frame_ids("train")[-300:]   # come la cache di C2
    LIFTED.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(os.cpu_count(), initializer=single_thread) as pool:
        for k, _ in enumerate(pool.map(lift_frame, frames, chunksize=16), 1):
            if k % 500 == 0 or k == len(frames):
                print(f"  {k}/{len(frames)} frame, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
