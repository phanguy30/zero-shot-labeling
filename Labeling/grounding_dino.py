from __future__ import annotations

from pathlib import Path
import sys

from Labeling.common import ensure_directories, iter_image_paths, write_yolo_labels
from Labeling.config import REPO_ROOT


GROUNDED_SAM2_ROOT = REPO_ROOT / "Grounded-SAM-2"


def _filter_and_dedupe_boxes(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[str],
    confidence_threshold: float,
    nms_iou_threshold: float,
) -> tuple[list[list[float]], list[float], list[str]]:
    """Filter boxes by confidence and apply non-maximum suppression.

    Args:
        boxes: List of boxes in xyxy or cxcywh format as lists of floats.
        scores: Corresponding confidence scores.
        labels: Corresponding class/phrase labels.
        confidence_threshold: Minimum score to keep a box.
        nms_iou_threshold: IoU threshold used for NMS.

    Returns:
        Tuple of (filtered_boxes, filtered_scores, filtered_labels) after NMS.
    """
    import torch
    from torchvision.ops import nms

    filtered_boxes: list[list[float]] = []
    filtered_scores: list[float] = []
    filtered_labels: list[str] = []

    for box, score, label in zip(boxes, scores, labels):
        if score < confidence_threshold:
            continue
        filtered_boxes.append(box)
        filtered_scores.append(score)
        filtered_labels.append(label)

    if not filtered_boxes:
        return filtered_boxes, filtered_scores, filtered_labels

    boxes_tensor = torch.tensor(filtered_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(filtered_scores, dtype=torch.float32)
    keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=nms_iou_threshold)

    return (
        boxes_tensor[keep_indices].cpu().numpy().tolist(),
        scores_tensor[keep_indices].cpu().numpy().tolist(),
        [filtered_labels[idx] for idx in keep_indices.cpu().numpy().tolist()],
    )


def run_grounding_dino_directory(
    image_dir: Path,
    output_dir: Path,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    confidence_threshold: float = 0.0,
    nms_iou_threshold: float = 0.5,
    device: str | None = None,
    grounding_dino_config: Path | None = None,
    grounding_dino_checkpoint: Path | None = None,
    ) -> Path:
    """Run Grounding DINO on all images in ``image_dir`` and write YOLO labels.

    This function loads the bundled Grounded-SAM-2 Grounding DINO model (from
    the repository submodule), runs inference for each image, writes YOLO-style
    label files to ``output_dir/labels``, and saves visualization images to
    ``output_dir/viz``.

    Args:
        image_dir: Directory containing input images.
        output_dir: Directory to write labels and visualizations.
        text_prompt: Text prompt used by Grounding DINO.
        box_threshold: Box presence probability threshold.
        text_threshold: Text matching threshold.
        confidence_threshold: Minimum confidence to keep a box after filtering.
        nms_iou_threshold: IoU threshold for NMS.
        device: Optional device string (e.g., 'cpu', 'cuda', 'mps').
        grounding_dino_config: Optional config file path override.
        grounding_dino_checkpoint: Optional checkpoint path override.

    Returns:
        The ``output_dir`` path passed in.
    """
    import cv2
    import numpy as np
    import torch
    from torchvision.ops import box_convert

    if str(GROUNDED_SAM2_ROOT) not in sys.path:
        sys.path.insert(0, str(GROUNDED_SAM2_ROOT))

    from grounding_dino.groundingdino.util.inference import load_image, load_model, predict

    resolved_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    config_path = grounding_dino_config or (REPO_ROOT / "Grounded-SAM-2" / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
    checkpoint_path = grounding_dino_checkpoint or (REPO_ROOT / "Grounded-SAM-2" / "gdino_checkpoints" / "groundingdino_swint_ogc.pth")

    labels_dir = output_dir / "labels"
    viz_dir = output_dir / "viz"
    ensure_directories(output_dir, labels_dir, viz_dir, output_dir / "viz_with_gt")

    model = load_model(
        model_config_path=str(config_path),
        model_checkpoint_path=str(checkpoint_path),
        device=resolved_device,
    )
    model.eval()

    for image_path in iter_image_paths(image_dir):
        image_source, image = load_image(str(image_path))
        height, width, _ = image_source.shape

        boxes, confidences, labels = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=resolved_device,
        )

        if boxes.shape[0] == 0:
            write_yolo_labels([], labels_dir / f"{image_path.stem}.txt")
            continue

        boxes = boxes * torch.tensor([width, height, width, height], dtype=boxes.dtype, device=boxes.device)
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy().tolist()
        confidences_np = confidences.cpu().numpy().tolist() if hasattr(confidences, "cpu") else np.asarray(confidences).tolist()

        filtered_boxes, filtered_scores, filtered_labels = _filter_and_dedupe_boxes(
            boxes=input_boxes,
            scores=confidences_np,
            labels=labels,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )

        viz_image_rgb = cv2.cvtColor(image_source.copy(), cv2.COLOR_BGR2RGB)
        for box, score in zip(filtered_boxes, filtered_scores):
            x1, y1, x2, y2 = [int(value) for value in box]
            cv2.rectangle(viz_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(viz_image_rgb, f"{score:.2f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(viz_dir / image_path.name), viz_image_rgb)

        yolo_boxes: list[dict[str, float | int | None]] = []
        for box in filtered_boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            yolo_boxes.append({"class": 0, "x": cx, "y": cy, "w": w, "h": h, "conf": None})

        write_yolo_labels(yolo_boxes, labels_dir / f"{image_path.stem}.txt")

    return output_dir
