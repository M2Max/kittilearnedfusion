import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]                # src/ (kitti, config) e src/methods/ (utils, fine_tuning)

import numpy as np
import torch

import utils
from config import CLASSES, RESULTS
from dataset2d import Kitti2D
from fine_tuning import DEVICE, N_HOLDOUT, NUM_WORKERS, WEIGHTS, load_model
from kitti.data import frame_ids

CACHE = RESULTS / "cache" / "c2_detector2d"                 # stesso formato di coco_detector2d, un .npz per frame
BATCH_SIZE = 8


def frames_to_cache():
    return frame_ids("val") + frame_ids("train")[-N_HOLDOUT:]   # val per i numeri finali, holdout per la fusione


@torch.inference_mode()
def cache_detections(frames):
    model = load_model()                                    # stessa scala del training (min_size 560, max_size 1860)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.to(DEVICE).eval()                                 # soglia score 0.05 di default: la fusione vuole anche le deboli

    loader = torch.utils.data.DataLoader(
        Kitti2D(frames, train=False),                       # niente flip
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=utils.collate_fn
    )

    CACHE.mkdir(parents=True, exist_ok=True)
    done = 0
    for images, targets in loader:
        outputs = model([img.to(DEVICE) for img in images])
        for target, out in zip(targets, outputs):
            np.savez(CACHE / f"{target['image_id']:06d}.npz",
                     boxes=out["boxes"].cpu().numpy().astype(np.float32),   # [u0, v0, u1, v1] pixel immagine originale
                     labels=np.array([CLASSES[k - 1] for k in out["labels"].tolist()], dtype="<U10"),
                     scores=out["scores"].cpu().numpy().astype(np.float32))
        done += len(images)
        if done % 400 < BATCH_SIZE or done == len(frames):
            print(f"  c2: {done}/{len(frames)} frame", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="solo i primi N frame, per provare")
    parser.add_argument("--recompute", action="store_true", help="riscrive anche i frame già in cache")
    args = parser.parse_args()

    frames = frames_to_cache()[:args.limit]
    if not args.recompute:
        frames = [i for i in frames if not (CACHE / f"{i:06d}.npz").exists()]   # riprende da dove si era fermato
    print(f"{len(frames)} frame da calcolare su {DEVICE}")
    if frames:
        cache_detections(frames)


if __name__ == "__main__":                                  # i worker del DataLoader reimportano il modulo
    main()
