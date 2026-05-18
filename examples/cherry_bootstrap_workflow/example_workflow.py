"""Run the cherry bootstrap labeling workflow step by step.

This script mirrors the notebook in the same folder, but keeps the
workflow executable from the command line. Each step is isolated in a
small function so the pipeline stays readable and easy to modify.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Labeling.config import BootstrapWorkflowConfig
from Labeling.manual_review import ManualReviewConfig, ManualReviewSession
from Labeling.grounding_dino import run_grounding_dino_directory
from Labeling.merge import merge_label_sets
from Labeling.yolo import prepare_yolo_dataset, run_yolo_inference, train_yolo


WORKFLOW_CONFIG = BootstrapWorkflowConfig(
    image_dir=REPO_ROOT / "dataset" / "cherry-detection-1" / "valid" / "images",
    dino_output_dir=REPO_ROOT / "dataset" / "dino_out",
    yolo_dataset_dir=REPO_ROOT / "dataset" / "yolo_data_set",
    yolo_project_dir=REPO_ROOT / "yolo_train_out",
    yolo_prediction_dir=REPO_ROOT / "dataset" / "yolo_predictions",
    merged_label_dir=REPO_ROOT / "dataset" / "autolabels" / "merged_labels",
    text_prompt="small red fruit",
    box_threshold=0.35,
    text_threshold=0.25,
    confidence_threshold=0.0,
    train_run_name="yolo_pt_bootstrap",
    prediction_run_name="yolo_pt_bootstrap_pred",
    class_names=("cherry",),
    train_ratio=0.8,
    random_seed=42,
    nms_iou_threshold=0.5,
    merge_iou_threshold=0.5,
    merge_mode="auto",
    train_epochs=50,
    train_batch_size=32,
    train_imgsz=640,
    train_workers=8,
    train_device=None,
    inference_conf=0.03,
    inference_iou=0.30,
    grounding_dino_config=REPO_ROOT
    / "Grounded-SAM-2"
    / "grounding_dino"
    / "groundingdino"
    / "config"
    / "GroundingDINO_SwinT_OGC.py",
    grounding_dino_checkpoint=REPO_ROOT
    / "Grounded-SAM-2"
    / "gdino_checkpoints"
    / "groundingdino_swint_ogc.pth",
    device=None,
)


MANUAL_REVIEW_CONFIG = ManualReviewConfig(
    image_source=REPO_ROOT / "dataset" / "cherry-detection-1" / "valid" / "images",
    dino_labels=REPO_ROOT / "dataset" / "dino_out" / "labels",
    yolo_labels=REPO_ROOT / "dataset" / "yolo_predictions" / "yolo_pt_bootstrap_pred" / "labels",
    output_labels=REPO_ROOT / "dataset" / "autolabels" / "manual_review_labels",
)


def run_grounding_dino_step(config: BootstrapWorkflowConfig) -> Path:
    """Run Grounding DINO and return the output directory."""
    return run_grounding_dino_directory(
        image_dir=config.image_dir,
        output_dir=config.dino_output_dir,
        text_prompt=config.text_prompt,
        box_threshold=config.box_threshold,
        text_threshold=config.text_threshold,
        confidence_threshold=config.confidence_threshold,
        nms_iou_threshold=config.nms_iou_threshold,
        device=config.device,
        grounding_dino_config=config.grounding_dino_config,
        grounding_dino_checkpoint=config.grounding_dino_checkpoint,
    )


def prepare_yolo_step(config: BootstrapWorkflowConfig, dino_output_dir: Path) -> Path:
    """Create the YOLO train/validation split and data.yaml."""
    return prepare_yolo_dataset(
        image_dir=config.image_dir,
        label_dir=dino_output_dir / "labels",
        yolo_data_dir=config.yolo_dataset_dir,
        class_names=config.class_names,
        train_ratio=config.train_ratio,
        seed=config.random_seed,
    )


def train_yolo_step(config: BootstrapWorkflowConfig, data_yaml: Path) -> Path:
    """Train YOLO on the generated labels and return the best weights path."""
    return train_yolo(
        data_yaml=data_yaml,
        project_dir=config.yolo_project_dir,
        run_name=config.train_run_name,
        epochs=config.train_epochs,
        imgsz=config.train_imgsz,
        batch_size=config.train_batch_size,
        resume=True,
        workers=config.train_workers,
        device=config.train_device or config.device,
    )


def run_yolo_inference_step(config: BootstrapWorkflowConfig, best_weights: Path) -> Path:
    """Run YOLO inference and return the prediction directory."""
    return run_yolo_inference(
        weights_path=best_weights,
        source_dir=config.image_dir,
        project_dir=config.yolo_prediction_dir,
        run_name=config.prediction_run_name,
        conf=config.inference_conf,
        iou=config.inference_iou,
        device=config.device,
    )


def merge_labels_step(
    config: BootstrapWorkflowConfig,
    dino_output_dir: Path,
    prediction_dir: Path,
) -> Path:
    """Merge YOLO and DINO labels automatically into the final label directory."""
    return merge_label_sets(
        image_dir=config.image_dir,
        yolo_label_dir=prediction_dir / "labels",
        dino_label_dir=dino_output_dir / "labels",
        merged_label_dir=config.merged_label_dir,
        iou_threshold=config.merge_iou_threshold,
    )


def launch_manual_review() -> None:
    """Launch the interactive manual review tool for the merge step."""
    ManualReviewSession(MANUAL_REVIEW_CONFIG).run()


def main() -> None:
    """Execute the workflow one stage at a time."""
    config = WORKFLOW_CONFIG

    dino_output_dir = run_grounding_dino_step(config)
    print(f"Grounding DINO outputs: {dino_output_dir}")

    data_yaml = prepare_yolo_step(config, dino_output_dir)
    print(f"YOLO dataset YAML: {data_yaml}")

    best_weights = train_yolo_step(config, data_yaml)
    print(f"YOLO weights: {best_weights}")

    prediction_dir = run_yolo_inference_step(config, best_weights)
    print(f"YOLO predictions: {prediction_dir}")

    if config.merge_mode == "auto":
        merged_dir = merge_labels_step(config, dino_output_dir, prediction_dir)
        print(f"Merged labels: {merged_dir}")
    elif config.merge_mode == "manual":
        launch_manual_review()
    else:
        raise ValueError(f"Unsupported merge_mode: {config.merge_mode}")


if __name__ == "__main__":
    main()
