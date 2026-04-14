from ultralytics import YOLO
from pathlib import Path
import os
import time
import torch

DATA_ROOT = Path(__file__).resolve().parent.parent
YOLO_DATA_DIR = DATA_ROOT / "dataset" / "cherry-detection-1"

YAML_PATH   = str((YOLO_DATA_DIR / 'data.yaml').resolve())
PROJECT_DIR = str((DATA_ROOT / 'yolo_train_out').resolve())
RUN_NAME    = 'yolo_pt_manual_label'

weights_path = f"{PROJECT_DIR}/{RUN_NAME}/weights/last.pt"

MAX_RETRIES = 10
retry_count = 0

while retry_count < MAX_RETRIES:
    try:
        if os.path.exists(weights_path):
            print(f"Resuming from {weights_path}")
            yolo_model = YOLO(weights_path)
            resume_flag = True
        else:
            print("Starting fresh from yolov8n.pt")
            yolo_model = YOLO("yolov8n.pt")
            resume_flag = False
            

        yolo_model.train(
            data=YAML_PATH,
            epochs=50,
            imgsz=640,
            batch=32,
            project=PROJECT_DIR,
            name=RUN_NAME,
            device='mps' if torch.backends.mps.is_available() else 'cpu',
            exist_ok=True,
            resume=resume_flag,
            workers=8
        )

        print("Training finished successfully")
        break

    except Exception as e:
        retry_count += 1
        print(f"Training crashed (attempt {retry_count}/{MAX_RETRIES})")
        print(e)
        time.sleep(10)

else:
    print("Max retries reached. Training failed permanently")
    



