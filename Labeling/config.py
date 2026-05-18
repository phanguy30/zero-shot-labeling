from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BootstrapWorkflowConfig:
    image_dir: Path = REPO_ROOT / "dataset" / "cherry-detection-1" / "valid" / "images"
    dino_output_dir: Path = REPO_ROOT / "dataset" / "dino_out"
    yolo_dataset_dir: Path = REPO_ROOT / "dataset" / "yolo_data_set"
    yolo_project_dir: Path = REPO_ROOT / "yolo_train_out"
    yolo_prediction_dir: Path = REPO_ROOT / "dataset" / "yolo_predictions"
    merged_label_dir: Path = REPO_ROOT / "dataset" / "autolabels" / "merged_labels"
    text_prompt: str = "small red fruit"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    confidence_threshold: float = 0.0
    train_run_name: str = "yolo_pt_bootstrap"
    prediction_run_name: str = "yolo_pt_bootstrap_pred"
    class_names: tuple[str, ...] = ("cherry",)
    train_ratio: float = 0.8
    random_seed: int = 42
    nms_iou_threshold: float = 0.5
    merge_iou_threshold: float = 0.5
    merge_mode: str = "auto"
    train_epochs: int = 50
    train_batch_size: int = 32
    train_imgsz: int = 640
    train_workers: int = 8
    train_device: str | None = None
    inference_conf: float = 0.03
    inference_iou: float = 0.30
    grounding_dino_config: Path = REPO_ROOT / "Grounded-SAM-2" / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint: Path = REPO_ROOT / "Grounded-SAM-2" / "gdino_checkpoints" / "groundingdino_swint_ogc.pth"
    device: str | None = None
