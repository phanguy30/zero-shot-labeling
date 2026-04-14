from datetime import datetime
import csv

from ultralytics import YOLO
from pathlib import Path
import os
import time
import torch

DATA_ROOT = Path(__file__).resolve().parent.parent
YOLO_DATA_DIR = DATA_ROOT / "dataset" / "cherry-detection-1"

YAML_PATH   = str((YOLO_DATA_DIR / 'bench.yaml').resolve())
PROJECT_DIR = str((DATA_ROOT / 'yolo_train_out').resolve())
PROJECT_DIR = Path(PROJECT_DIR) 
RUN_NAME    = 'yolo_pt_manual_label'


best_weights = PROJECT_DIR / RUN_NAME / 'weights' / 'best.pt'


best_weights = PROJECT_DIR / RUN_NAME / 'weights' / 'best.pt'
yolo_eval    = YOLO(str(best_weights))

val_metrics = yolo_eval.val(
    data     = YAML_PATH,
    imgsz    = 640, 
    project  = PROJECT_DIR,
    name     = RUN_NAME + '_bench_val',
    exist_ok = True,
)


print('\n── Validation Metrics ──────────────────────────────')
print(f'  mAP50      : {val_metrics.box.map50:.4f}')
print(f'  mAP50-95   : {val_metrics.box.map:.4f}')
print(f'  Precision  : {val_metrics.box.mp:.4f}')
print(f'  Recall     : {val_metrics.box.mr:.4f}')



save_path = PROJECT_DIR / "metrics_log.csv"

row = [
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    RUN_NAME,
    val_metrics.box.map50,
    val_metrics.box.map,
    val_metrics.box.mp,
    val_metrics.box.mr,
]

header = [ "timestamp", "run", "mAP50", "mAP50-95", "precision", "recall"]

file_exists = save_path.exists()

with open(save_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(header)
    writer.writerow(row)


