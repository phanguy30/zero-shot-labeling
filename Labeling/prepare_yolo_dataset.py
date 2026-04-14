import os
import time
from pathlib import Path
import shutil
import random


DATA_ROOT = Path(__file__).resolve().parent.parent
LABEL_DIR = DATA_ROOT / "dataset" / "autolabels" / "merged_labels"
YOLO_DATA_DIR = DATA_ROOT / "dataset" / "yolo_data_set"


IMAGES_DIR = DATA_ROOT / "dataset" / "cherry-detection-1" / "valid" / "images"


images_train_dir = YOLO_DATA_DIR / "train" / "images"
images_val_dir   = YOLO_DATA_DIR / "valid" / "images"

labels_train_dir = YOLO_DATA_DIR / "train" / "labels"
labels_val_dir   = YOLO_DATA_DIR / "valid" / "labels"

from pathlib import Path

def create_yolo_yaml(yolo_data_dir: Path, output_path: Path, class_names: list):
    names_dict = {i: name for i, name in enumerate(class_names)}

    yaml_content = f"""path: "{yolo_data_dir.resolve()}"

train: train/images
val: valid/images

names:
"""

    for k, v in names_dict.items():
        yaml_content += f"  {k}: {v}\n"

    with open(output_path, "w") as f:
        f.write(yaml_content)

    print(f"✅ YAML saved to: {output_path}")


# Create all folders
for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Create YOLO YAML
class_names = ["cherry"]
yaml_output_path = YOLO_DATA_DIR / "data.yaml"
create_yolo_yaml(YOLO_DATA_DIR, yaml_output_path, class_names)

    
#copy dino_out to a folder for yolo training

for img_path in IMAGES_DIR.glob("*.jpg"):
    r = random.random()

    if r < 0.8:
        img_target_dir = images_train_dir
        label_target_dir = labels_train_dir
    else:
        img_target_dir = images_val_dir
        label_target_dir = labels_val_dir

    # copy image
    shutil.copy(img_path, img_target_dir / img_path.name)

    # label paths
    label_path = LABEL_DIR / (img_path.stem + ".txt")
    target_label_path = label_target_dir / (img_path.stem + ".txt")

    
    shutil.copy(label_path, target_label_path)
    






