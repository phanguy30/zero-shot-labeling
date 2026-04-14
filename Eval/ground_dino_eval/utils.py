import sys
from pathlib import Path

import cv2
import torch
import numpy as np
import pandas as pd

from torchvision.ops import box_convert

# ---- paths to submodule ----
LABELING_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = LABELING_DIR.parent / "Grounded-SAM-2"
sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict


# =========================
# CONFIG
# =========================
TEXT_PROMPT = "gate."

IMAGE_DIR = LABELING_DIR / "dataset" / "gate_dataset" / "images" / "train"
LABEL_DIR = LABELING_DIR / "dataset" / "gate_dataset" / "labels" / "train"
OUTPUT_DIR = LABELING_DIR / "outputs" / "batch_compare"

SAM2_CHECKPOINT = REPO_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = REPO_ROOT / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = REPO_ROOT / "gdino_checkpoints" / "groundingdino_swint_ogc.pth"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MULTIMASK_OUTPUT = False

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "viz").mkdir(exist_ok=True)
(OUTPUT_DIR / "pred_masks").mkdir(exist_ok=True)


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


# =========================
# LABEL PARSING
# =========================

def yolo_bbox_line_to_xyxy(label_line: str, img_w: int, img_h: int):
    """
    Parse one YOLO bbox line:
    class_id x_center y_center width height  (all normalized to [0, 1])
    """
    parts = label_line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO bbox line: {label_line}")

    class_id = int(parts[0])
    xc, yc, bw, bh = map(float, parts[1:5])

    xc *= img_w
    yc *= img_h
    bw *= img_w
    bh *= img_h

    x1 = xc - bw / 2
    y1 = yc - bh / 2
    x2 = xc + bw / 2
    y2 = yc + bh / 2

    return class_id, [x1, y1, x2, y2]


def yolo_bbox_file_to_boxes(label_file: Path, img_w: int, img_h: int):
    """Parse a YOLO bbox txt file into a list of xyxy boxes."""
    boxes = []
    class_ids = []

    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id, box = yolo_bbox_line_to_xyxy(line, img_w, img_h)
            class_ids.append(class_id)
            boxes.append(box)

    return boxes, class_ids


def yolo_seg_to_mask(label_file, img_h=480, img_w=640, normalize=True):
    """
    Convert YOLOv8 segmentation label file to a multi-class mask (H x W).

    Returns:
        mask: np.ndarray (H, W), dtype=np.uint8
        labels: list of (class_id, polygon)
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    labels = []

    with open(label_file, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        cls = int(parts[0])
        coords = np.array(list(map(float, parts[5:])), dtype=np.float32)
        pts = coords.reshape(-1, 2)

        if normalize:
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h

        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], cls + 1)  # 0 = background, classes start at 1
        labels.append((cls, pts))

    return mask, labels


# =========================
# METRICS
# =========================

def compute_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))


def compute_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    inter = np.logical_and(pred, gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def compute_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((pred == gt).sum() / gt.size)


def compute_precision(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    return float((tp + eps) / (tp + fp + eps))


def compute_recall(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return float((tp + eps) / (tp + fn + eps))


def get_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: pred={pred_mask.shape}, gt={gt_mask.shape}")

    return {
        "IoU": compute_iou(pred_mask, gt_mask),
        "Dice": compute_dice(pred_mask, gt_mask),
        "Accuracy": compute_accuracy(pred_mask, gt_mask),
        "Precision": compute_precision(pred_mask, gt_mask),
        "Recall": compute_recall(pred_mask, gt_mask),
    }


def compute_box_iou(box1, box2) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def get_bbox_metrics(
    model_out: list,
    gt: list,
    iou_threshold: float = 0.5
) -> dict:
    """
    Model output should be a list of (box, confidence, label) tuples
    GT should be an list of (box, label) tuples.
    Evaluate bounding box predictions using greedy matching.

    Returns dict with IoU, Precision, Recall, F1, TP, FP, FN.
    """
    
    
    
    
    
    filtered = sorted(model_out, key=lambda x: x[1], reverse=True)  # Sort by confidence

    matched_gt = set()
    matched_pred = set()
    ious = []

    for i in range(len(filtered)):
        pbox, _ , plabel = filtered[i]

        best_j, best_iou = -1, 0.0

        for j, (gtbox, gtlabel) in enumerate(gt):
            if plabel != gtlabel:  # Match class labels first
                continue
            if j in matched_gt:
                continue
            iou = compute_box_iou(pbox, gtbox)
            if iou > best_iou:  # Match class labels
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j != -1:
            matched_pred.add(i)
            matched_gt.add(best_j)
            ious.append(best_iou)

    tp = len(matched_pred)
    fp = len(filtered) - tp
    fn = len(gt) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    return {
        "IoU": mean_iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


# =========================
# INFERENCE
# =========================

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


# =========================
# VISUALIZATION
# =========================

def make_mask_overlay(
    image_bgr: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> np.ndarray:
    """
    Overlay mask comparison on image.
      Green  = GT only
      Red    = Pred only
      Yellow = Overlap
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    overlay = image_bgr.copy()
    overlay[np.logical_and(gt_mask, ~pred_mask)] = [0, 255, 0]
    overlay[np.logical_and(pred_mask, ~gt_mask)] = [0, 0, 255]
    overlay[np.logical_and(pred_mask, gt_mask)] = [0, 255, 255]

    return cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)


def make_bbox_overlay(
    image_bgr: np.ndarray,
    pred_boxes: list,
    gt_boxes: list,
) -> np.ndarray:
    """
    Draw GT boxes (green) and predicted boxes (red) on image.
    """
    overlay = image_bgr.copy()

    for box in gt_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)


# =========================
# SHARED BATCH RUNNER
# =========================

def get_matched_stems(image_folder: Path, label_folder: Path):
    """Return sorted list of stems present in both folders, with warnings for mismatches."""
    image_files = {
        p.stem: p for p in image_folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    }
    label_files = {
        p.stem: p for p in label_folder.iterdir()
        if p.suffix.lower() == ".txt"
    }

    common_stems = sorted(image_files.keys() & label_files.keys())

    if not common_stems:
        raise ValueError("No matching image/label pairs found.")

    for stem in sorted(image_files.keys() - label_files.keys()):
        print(f"[WARN] Image without label, skipping: {stem}")
    for stem in sorted(label_files.keys() - image_files.keys()):
        print(f"[WARN] Label without image, skipping: {stem}")

    return common_stems, image_files, label_files