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
from torchvision.ops import box_convert
from torchvision.ops import nms
import torch


from utils import (
    IMAGE_DIR,
    LABEL_DIR,
    LABEL_MAPPING,
    OUTPUT_DIR,
    CONFIDENCE_THRESHOLD,
    get_matched_stems,
    run_grounding_dino,
    yolo_bbox_file_to_boxes,
    get_bbox_metrics,
    make_bbox_overlay,
    remove_big_containers
)




def run_bbox_evaluation(
    image_folder=IMAGE_DIR,
    label_folder=LABEL_DIR,
    iou_threshold: float = 0.5,
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
        image_source, input_boxes , conf_np, labels = run_grounding_dino(image_path)
        
        
        
        
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

        # Visualization with only predicted boxes (no GT)
        
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
        
        

        # ── Ground truth ───────────────────────────────────────────
        gt_boxes, gt_label = yolo_bbox_file_to_boxes(label_path, img_w, img_h)
        gt = list(zip(gt_boxes, gt_label))

        # ── Metrics ────────────────────────────────────────────────
        filtered_labels = [LABEL_MAPPING.get(lbl, -1) for lbl in filtered_labels]
        model_out = list(zip(filtered_boxes, filtered_scores, filtered_labels))

        metrics = get_bbox_metrics(model_out, gt, iou_threshold=iou_threshold)

        # ── Visualization ──────────────────────────────────────────
        
        
        overlay = make_bbox_overlay(image_bgr, filtered_boxes, gt_boxes)
        cv2.imwrite(str(OUTPUT_DIR / "viz_with_gt" / f"{stem}_bbox_overlay.png"), overlay)

        rows.append({
            "stem": stem,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "num_pred_boxes": len(model_out),
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