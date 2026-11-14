import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

from config import CLASSES
from kitti.data import load_image, load_labels

LABEL_ID = {name: k + 1 for k, name in enumerate(CLASSES)}  # Car 1, Pedestrian 2, Cyclist 3; 0 = sfondo


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))      # specchia immagine e box insieme, solo in training
    transforms.append(T.ToDtype(torch.float, scale=True))   # uint8 0-255 -> float 0-1
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


class Kitti2D(Dataset):
    def __init__(self, frames, train):
        self.frames = list(frames)                          # id dei frame KITTI, non indici 0..N
        self.transforms = get_transform(train)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        i = self.frames[idx]
        img = load_image(i)                                 # RGB uint8 (H, W, 3)
        h, w = img.shape[:2]

        objs = [o for o in load_labels(i) if o["type"] in LABEL_ID]   # Van, DontCare e il resto scartati
        boxes = torch.as_tensor(np.array([o["bbox"] for o in objs]), dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([LABEL_ID[o["type"]] for o in objs], dtype=torch.int64)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])   # torchvision rifiuta box degeneri
        boxes, labels = boxes[keep], labels[keep]

        img = tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1))    # (3, H, W)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w)),
            "labels": labels,
            "image_id": i,                                  # int: chiave del valutatore COCO
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return self.transforms(img, target)
