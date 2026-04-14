"""
run_mask.py
-----------
Evaluate GroundingDINO + SAM2 segmentation masks against YOLOv8 segmentation ground truth.

Usage:
    python run_mask.py
"""

import json

import cv2
import numpy as np
import pandas as pd

from utils import (
    IMAGE_DIR,
    LABEL_DIR,
    OUTPUT_DIR,
    MULTIMASK_OUTPUT,
    sam2_predictor,
    get_matched_stems,
    run_grounding_dino,
    yolo_seg_to_mask,
    get_mask_metrics,
    make_mask_overlay,
)


def run_mask_evaluation(
    image_folder=IMAGE_DIR,
    label_folder=LABEL_DIR,
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate segmentation mask predictions on all matching image/label pairs.

    Parameters
    ----------
    image_folder : Path
        Folder containing images.
    label_folder : Path
        Folder containing YOLOv8 segmentation .txt labels
        (format: class_id x1 y1 x2 y2 px1 py1 ..., normalized polygon coords).

    Returns
    -------
    avg_metrics : dict
        Mean IoU, Dice, Accuracy, Precision, Recall across all images.
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
        image_source, input_boxes, conf_np, labels = run_grounding_dino(image_path)

        # Build merged predicted mask from SAM2
        pred_mask = np.zeros((img_h, img_w), dtype=bool)

        if input_boxes.shape[0] > 0:
            sam2_predictor.set_image(image_source)

            for box in input_boxes:
                masks, scores, _ = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box[None, :],
                    multimask_output=MULTIMASK_OUTPUT,
                )

                if MULTIMASK_OUTPUT:
                    best_mask = masks[int(np.argmax(scores))]
                else:
                    best_mask = masks[0]

                pred_mask |= best_mask.astype(bool)

        # ── Ground truth ───────────────────────────────────────────
        gt_mask_raw, _ = yolo_seg_to_mask(label_path, img_h, img_w)
        gt_mask = gt_mask_raw.astype(bool)

        # ── Metrics ────────────────────────────────────────────────
        metrics = get_mask_metrics(pred_mask, gt_mask)

        # ── Save predicted mask ────────────────────────────────────
        pred_u8 = pred_mask.astype(np.uint8) * 255
        cv2.imwrite(str(OUTPUT_DIR / "pred_masks" / f"{stem}_pred.png"), pred_u8)

        # ── Visualization ──────────────────────────────────────────
        overlay = make_mask_overlay(image_bgr, pred_mask, gt_mask)
        cv2.imwrite(str(OUTPUT_DIR / "viz" / f"{stem}_mask_overlay.png"), overlay)

        rows.append({
            "stem": stem,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "num_boxes": input_boxes.shape[0],
            "avg_confidence": float(np.mean(conf_np)) if len(conf_np) else 0.0,
            "pred_pixels": int(pred_mask.sum()),
            "gt_pixels": int(gt_mask.sum()),
            **metrics,
        })

    df = pd.DataFrame(rows)

    metric_cols = ["IoU", "Dice", "Accuracy", "Precision", "Recall"]
    avg_metrics = {
        col: float(df[col].mean()) if (len(df) and col in df.columns) else None
        for col in metric_cols
    }

    return avg_metrics, df


if __name__ == "__main__":
    avg_metrics, df = run_mask_evaluation()

    df.to_csv(OUTPUT_DIR / "mask_metrics.csv", index=False)

    with open(OUTPUT_DIR / "mask_summary.json", "w") as f:
        json.dump(avg_metrics, f, indent=4)

    print("\nPer-image metrics:")
    print(df.to_string(index=False))

    print("\nAverage metrics:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")