import sys
from pathlib import Path

import cv2
import torch
import numpy as np
import pandas as pd

from torchvision.ops import box_convert
from torchvision.ops import nms


# ---- paths to submodule ----
REPO_ROOT = Path(__file__).resolve().parent.parent
print(f"Repo root: {REPO_ROOT}")
MODEL_ROOT = REPO_ROOT / "Grounded-SAM-2"
sys.path.insert(0, str(MODEL_ROOT))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict


# =========================
# CONFIG
# =========================
TEXT_PROMPT = "small red fruit"

IMAGE_DIR = REPO_ROOT / "dataset" / "cherry-detection-1" / "valid" / "images"
LABEL_DIR = REPO_ROOT / "dataset" / "cherry-detection-1" / "valid" / "labels"
OUTPUT_DIR = REPO_ROOT / "dataset" / "dino_out"

SAM2_CHECKPOINT = MODEL_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = MODEL_ROOT / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = MODEL_ROOT / "gdino_checkpoints" / "groundingdino_swint_ogc.pth"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
CONFIDENCE_THRESHOLD = 0.0
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MULTIMASK_OUTPUT = False

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "viz").mkdir(exist_ok=True)
(OUTPUT_DIR/ "labels").mkdir(exist_ok=True)
(OUTPUT_DIR / "viz_with_gt").mkdir(exist_ok=True)


# =========================
# LOAD MODELS ONCE
# =========================
sam2_model = build_sam2(
    SAM2_MODEL_CONFIG,
    str(SAM2_CHECKPOINT),
    device=DEVICE
)
sam2_predictor = SAM2ImagePredictor(sam2_model)

grounding_model = load_model(
    model_config_path=str(GROUNDING_DINO_CONFIG),
    model_checkpoint_path=str(GROUNDING_DINO_CHECKPOINT),
    device=DEVICE
)


def run_grounding_dino(image_path: Path):
    """
    Run GroundingDINO on one image.

    Returns
    -------
    image_source : np.ndarray   (H, W, 3) BGR
    input_boxes  : np.ndarray   (N, 4) xyxy pixel coordinates
    conf_np      : np.ndarray   (N,) confidence scores
    labels       : list[str]
    """
    image_source, image = load_image(str(image_path))
    h, w, _ = image_source.shape

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE,
    )

    if boxes.shape[0] == 0:
        return image_source, np.empty((0, 4)), np.array([]), []

    boxes = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
    conf_np = confidences.cpu().numpy() if hasattr(confidences, "cpu") else np.asarray(confidences)

    return image_source, input_boxes, conf_np, labels


def remove_big_containers(boxes, scores, area_ratio_thresh=1.5):
    keep = []
    
    for i, boxA in enumerate(boxes):
        x1A, y1A, x2A, y2A = boxA
        areaA = (x2A - x1A) * (y2A - y1A)
        
        remove = False
        
        for j, boxB in enumerate(boxes):
            if i == j:
                continue
                
            x1B, y1B, x2B, y2B = boxB
            areaB = (x2B - x1B) * (y2B - y1B)
            
            tol = 10  # pixels
            
            # Check if B is inside A
            inside = (
                x1B >= x1A-tol and y1B >= y1A-tol and
                x2B <= x2A+tol and y2B <= y2A+tol
            )
            
            # Only remove if A is significantly bigger
            if inside and areaA > areaB * area_ratio_thresh:
                remove = True
                break
        
        if not remove:
            keep.append(i)
    
    return keep



if __name__ == "__main__":
    grounding_model.eval()
    for image_path in IMAGE_DIR.glob("*.jpg"):
        image_source, input_boxes, conf_np, labels = run_grounding_dino(image_path)
        
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        # Filter boxes by confidence threshold
        for box, conf, label in zip(input_boxes, conf_np, labels):
            if conf < CONFIDENCE_THRESHOLD:
                continue

            filtered_boxes.append(box)
            filtered_scores.append(conf)
            filtered_labels.append(label)
            
            
  
            
        # Apply NMS
        if len(filtered_boxes) > 0:
            filtered_boxes_tensor = torch.tensor(filtered_boxes, dtype=torch.float32)
            filtered_scores_tensor = torch.tensor(filtered_scores, dtype=torch.float32)
            
            keep_indices = nms(filtered_boxes_tensor, filtered_scores_tensor, iou_threshold=0.5)
        

            filtered_boxes  = filtered_boxes_tensor[keep_indices].cpu().numpy()
            filtered_scores = filtered_scores_tensor[keep_indices].cpu().numpy()
            filtered_labels = [filtered_labels[i] for i in keep_indices.cpu().numpy()]
        
        # Remove big containers
        if len(filtered_boxes) > 0:
            keep_indices = remove_big_containers(filtered_boxes, filtered_scores, area_ratio_thresh=1.5)
            filtered_boxes  = filtered_boxes[keep_indices]
            filtered_scores = filtered_scores[keep_indices]
            filtered_labels = [filtered_labels[i] for i in keep_indices]

        # Save results
        output_image_path = OUTPUT_DIR / "viz" / image_path.name
        output_label_path = OUTPUT_DIR / "labels" / (image_path.stem + ".txt")

        # Visualization
        
        viz_image_rgb = cv2.cvtColor(image_source.copy(), cv2.COLOR_BGR2RGB)
        
        for box, conf, label in zip(filtered_boxes, filtered_scores, filtered_labels):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(viz_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(viz_image_rgb, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(output_image_path), viz_image_rgb)

        # Save labels in YOLO format
        with open(output_label_path, "w") as f:
            for box, conf, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2 / image_source.shape[1]
                cy = (y1 + y2) / 2 / image_source.shape[0]
                w = (x2 - x1) / image_source.shape[1]
                h = (y2 - y1) / image_source.shape[0]
                f.write(f"0 {cx} {cy} {w} {h}\n")

