import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"                                  # results/<split>/<metodo>/: un'uscita per run
KITTI_ROOT = Path(os.environ.get("KITTI_ROOT", REPO / "data/kitti"))   # radice KITTI, configurabile da ambiente
DATA = KITTI_ROOT / "training"
SPLITS = KITTI_ROOT / "ImageSets"                           # train.txt, val.txt: elenco dei frame per split
CLASSES = ["Car", "Pedestrian", "Cyclist"]

IOU_THRESH = {"Car": 0.70, "Pedestrian": 0.50, "Cyclist": 0.50}   # soglie ufficiali KITTI
NEIGHBOR = {"Car": "Van", "Pedestrian": "Person_sitting", "Cyclist": None}   # classi "vicine": né premio né errore
