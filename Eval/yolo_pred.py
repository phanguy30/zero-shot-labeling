from datetime import datetime
import csv

from ultralytics import YOLO
from pathlib import Path
import os
import time
import torch

DATA_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = DATA_ROOT / "compare" / "pred_images"
PROJECT_DIR = str((DATA_ROOT / 'yolo_train_out').resolve())
PROJECT_DIR = Path(PROJECT_DIR) 
RUN_NAME    = 'yolo_pt_manual_label'

best_weights = PROJECT_DIR / RUN_NAME / 'weights' / 'best.pt'


best_weights = PROJECT_DIR / RUN_NAME / 'weights' / 'best.pt'
yolo_eval    = YOLO(str(best_weights))

results = yolo_eval.predict(
    source=IMAGE_DIR,
    imgsz=640,
    conf=0.35,
    save=True,
    show_labels=False,   
    show_conf=True,     
    project=PROJECT_DIR,
    name=RUN_NAME + "_pred",
    exist_ok=True,
)
