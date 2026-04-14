from ultralytics import YOLO
from pathlib import Path
import os
import time
import torch

DATA_ROOT = Path(__file__).resolve().parent.parent
YOLO_DATA_DIR = DATA_ROOT / "dataset" / "yolo_data_set"

YAML_PATH   = str((YOLO_DATA_DIR / 'data.yaml').resolve())
PROJECT_DIR = (DATA_ROOT / 'yolo_train_out').resolve()
RUN_NAME    = 'yolo_pt'



best_weights = PROJECT_DIR / RUN_NAME / 'weights' / 'best.pt'
yolo_eval    = YOLO(str(best_weights))

yolo_eval.predict(
    source = str((DATA_ROOT / "dataset" / "cherry-detection-1" / "valid" / "images").resolve()),
    project = str((DATA_ROOT / "dataset" / "yolo_predictions").resolve()),
    name = RUN_NAME + "_pred",
    exist_ok = True,
    conf = 0.03,
    iou = 0.30,
    device = 'mps' if torch.backends.mps.is_available() else 'cpu',
    save = True ,
    save_txt = True,
    save_conf = True
    
)