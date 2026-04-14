"""
run_bbox.py
-----------
Evaluate GroundingDINO bounding box predictions against YOLO bbox ground truth.

Usage:
    python run_bbox.py
"""

import json

import cv2
import numpy as np
import pandas as pd

from utils import (
    IMAGE_DIR,
    LABEL_DIR,
    OUTPUT_DIR,
    get_matched_stems,
    run_grounding_dino,
    yolo_bbox_file_to_boxes,
    get_bbox_metrics,
    make_bbox_overlay,
)


def run_bbox_evaluation(
    image_folder=IMAGE_DIR,
    label_folder=LABEL_DIR,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate bounding box predictions on all matching image/label pairs.

    Parameters
    ----------
    image_folder : Path
        Folder containing images.
    label_folder : Path
        Folder containing YOLO bbox .txt labels
        (format: class_id x_center y_center width height, normalized).
    iou_threshold : float
        IoU threshold for counting a detection as a true positive.

    Returns
    -------
    avg_metrics : dict
        Mean IoU, Precision, Recall, F1 across all images.
    df : pd.DataFrame
        Per-image metrics.
    """
    common_stems, image_files, label_files = get_matched_stems(image_folder, label_folder)

    rows = []

    for stem in common_stems:
        image_path = image_files[stem]
        label_path = label_files[stem]

        print(f"Processing {stem}...")

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[WARN] Failed to read image: {image_path}")
            continue

        img_h, img_w = image_bgr.shape[:2]

        # ── Inference ──────────────────────────────────────────────
        image_source, pred_boxes, conf_np, labels = run_grounding_dino(image_path)
        
        model_out = list(zip(pred_boxes, conf_np, labels))
        

        # ── Ground truth ───────────────────────────────────────────
        gt_boxes, gt_label = yolo_bbox_file_to_boxes(label_path, img_w, img_h)
        gt = list(zip(gt_boxes, gt_label))

        # ── Metrics ────────────────────────────────────────────────

        metrics = get_bbox_metrics(model_out, gt, iou_threshold=iou_threshold)

        # ── Visualization ──────────────────────────────────────────
        viz_boxes = [box for box, conf, label in model_out if conf >= confidence_threshold]
        
        overlay = make_bbox_overlay(image_bgr, viz_boxes, gt_boxes)
        cv2.imwrite(str(OUTPUT_DIR / "viz" / f"{stem}_bbox_overlay.png"), overlay)

        rows.append({
            "stem": stem,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "num_pred_boxes": len(model_out),
            "num_viz_boxes": len(viz_boxes),
            "num_gt_boxes": len(gt_boxes),
            "avg_confidence": float(np.mean(conf_np)) if len(conf_np) else 0.0,
            **metrics,
        })

    df = pd.DataFrame(rows)

    metric_cols = ["IoU", "Precision", "Recall", "F1"]
    avg_metrics = {
        col: float(df[col].mean()) if (len(df) and col in df.columns) else None
        for col in metric_cols
    }

    return avg_metrics, df


if __name__ == "__main__":
    avg_metrics, df = run_bbox_evaluation()

    df.to_csv(OUTPUT_DIR / "bbox_metrics.csv", index=False)

    with open(OUTPUT_DIR / "bbox_summary.json", "w") as f:
        json.dump(avg_metrics, f, indent=4)

    print("\nPer-image metrics:")
    print(df.to_string(index=False))

    print("\nAverage metrics:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")