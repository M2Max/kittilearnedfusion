import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]                # src/ (kitti, config) e src/methods/ (engine, utils)

import utils
from engine import train_one_epoch, evaluate
from dataset2d import Kitti2D
from kitti.data import frame_ids
from config import RESULTS

import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# train with mps or CUDA based on what machine I run it, fallback to cpu
DEVICE = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
LEARNING_RATE=0.005
MOMENTUM=0.9
WEIGHT_DECAY=0.0005
NUM_EPOCHS = 5
STEP_SIZE=3
GAMMA=0.1
BATCH_SIZE = 8
NUM_WORKERS = 8                                             # processi che decodificano i PNG
N_HOLDOUT = 300                                             # ultimi frame di train: riservati alla fusione
N_MONITOR = 200                                             # coda di train[:-300] usata solo per evaluate
WEIGHTS = RESULTS / "cache" / "c2_fasterrcnn.pt"            # pesi del modello fine-tuned

def load_model():
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    num_classes = 4  # 3 classes (car, pedestrian, cyclist) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def prepare_data():
    ids = frame_ids("train")[:-N_HOLDOUT]                   # gli ultimi 300 non entrano mai qui
    dataset = Kitti2D(ids[:-N_MONITOR], train=True)
    dataset_test = Kitti2D(ids[-N_MONITOR:], train=False)   # stessi dati, senza flip

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=utils.collate_fn
    )

    return data_loader, data_loader_test

def train_config():
    # get the model using our helper function
    model = load_model()
    # move model to the right device
    model.to(DEVICE)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=STEP_SIZE,
        gamma=GAMMA
    )

    return model, optimizer, lr_scheduler

def train():
    model, optimizer, lr_scheduler = train_config()
    data_loader, data_loader_test = prepare_data()
    for epoch in range(NUM_EPOCHS):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, DEVICE, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=DEVICE)
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), WEIGHTS)             # sovrascritto a ogni epoca: un crash non perde tutto

if __name__ == "__main__":                                  # i worker del DataLoader reimportano il modulo
    train()
