"""Interactive manual review for merging DINO and YOLO labels.

This module exposes a reusable class that receives explicit data
sources, making it easy to invoke from a workflow or notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ManualReviewConfig:
    """Configuration for the manual label review UI."""

    image_source: Path
    dino_labels: Path
    yolo_labels: Path
    output_labels: Path
    iou_threshold: float = 0.5


class ManualReviewSession:
    """Run interactive label review against a set of image and label paths."""

    def __init__(self, config: ManualReviewConfig) -> None:
        """Store the review configuration and prepare the output directory."""
        self.config = config
        self.config.output_labels.mkdir(parents=True, exist_ok=True)

    def read_labels(self, label_path: Path) -> list[dict[str, float | int | None]]:
        """Read a YOLO-format label file into a list of box dictionaries."""
        boxes: list[dict[str, float | int | None]] = []

        if not label_path.exists():
            return boxes

        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                values = list(map(float, line.strip().split()))
                if len(values) < 5:
                    continue

                class_id, x, y, w, h = values[:5]
                confidence = values[5] if len(values) >= 6 else None
                boxes.append(
                    {
                        "class": int(class_id),
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "conf": confidence,
                    }
                )

        return boxes

    def yolo_to_xyxy(
        self, box: dict[str, float | int | None], img_w: int, img_h: int
    ) -> list[int]:
        """Convert a normalized YOLO box into pixel xyxy coordinates."""
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]

        x1 = int((x - w / 2) * img_w)
        y1 = int((y - h / 2) * img_h)
        x2 = int((x + w / 2) * img_w)
        y2 = int((y + h / 2) * img_h)

        return [x1, y1, x2, y2]

    def iou(self, box1: list[int], box2: list[int]) -> float:
        """Compute IoU between two xyxy boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter_area

        return inter_area / union if union > 0 else 0.0

    def save_labels(
        self, boxes: list[dict[str, float | int | None]], stem: str
    ) -> Path:
        """Write reviewed boxes back to a YOLO label file."""
        label_path = self.config.output_labels / f"{stem}.txt"
        with label_path.open("w", encoding="utf-8") as handle:
            for box in boxes:
                handle.write(f"{box['class']} {box['x']} {box['y']} {box['w']} {box['h']}\n")

        return label_path

    def review_image(self, img_path: Path) -> Path | None:
        """Run the interactive review loop for a single image."""
        print(f"Processing {img_path.name}...")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            return None

        h, w = img.shape[:2]
        stem = img_path.stem

        yolo_boxes = self.read_labels(self.config.yolo_labels / f"{stem}.txt")
        dino_boxes = self.read_labels(self.config.dino_labels / f"{stem}.txt")

        yolo_xy = [self.yolo_to_xyxy(box, w, h) for box in yolo_boxes]
        dino_xy = [self.yolo_to_xyxy(box, w, h) for box in dino_boxes]

        matched_yolo: set[int] = set()
        matched_dino: set[int] = set()
        matches: list[tuple[int, int, float]] = []

        for i, yolo_box in enumerate(yolo_xy):
            best_iou = 0.0
            best_j = -1

            for j, dino_box in enumerate(dino_xy):
                score = self.iou(yolo_box, dino_box)
                if score > best_iou:
                    best_iou = score
                    best_j = j

            if best_iou > self.config.iou_threshold and best_j >= 0:
                matches.append((i, best_j, best_iou))
                matched_yolo.add(i)
                matched_dino.add(best_j)

        yolo_only = [idx for idx in range(len(yolo_boxes)) if idx not in matched_yolo]
        dino_only = [idx for idx in range(len(dino_boxes)) if idx not in matched_dino]

        needs_review = len(matches) < len(yolo_boxes) or len(matches) < len(dino_boxes)
        if not needs_review:
            return None

        print(f"[REVIEW] {stem}")
        print(
            f"  - {len(matches)} matches, {len(yolo_only)} YOLO-only, {len(dino_only)} DINO-only"
        )

        current_boxes: list[dict[str, float | int | None]] = [
            yolo_boxes[yolo_index] for yolo_index, _, _ in matches
        ]
        review_list = [*(('yolo', idx) for idx in yolo_only), *(('dino', idx) for idx in dino_only)]

        current_idx = 0
        final_idx = 0

        while True:
            canvas = img.copy()

            if current_idx < len(review_list):
                for box in current_boxes:
                    x1, y1, x2, y2 = self.yolo_to_xyxy(box, w, h)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)

                for idx in yolo_only:
                    x1, y1, x2, y2 = yolo_xy[idx]
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 100, 0), 2)

                for idx in dino_only:
                    x1, y1, x2, y2 = dino_xy[idx]
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 100), 2)

                source, idx = review_list[current_idx]
                if source == "yolo":
                    box_xy = yolo_xy[idx]
                    box_raw = yolo_boxes[idx]
                else:
                    box_xy = dino_xy[idx]
                    box_raw = dino_boxes[idx]

                x1, y1, x2, y2 = box_xy
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(
                    canvas,
                    f"{source.upper()} {current_idx + 1}/{len(review_list)} | accepted: {len(current_boxes)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    canvas,
                    f"FINAL REVIEW {final_idx + 1}/{len(current_boxes)} | arrows/h/l move | d delete | b back | s save",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                for idx, box in enumerate(current_boxes):
                    x1, y1, x2, y2 = self.yolo_to_xyxy(box, w, h)
                    thickness = 3 if idx == final_idx else 2
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), thickness)

            cv2.imshow("Annotator", canvas)
            key = cv2.waitKey(0)

            if key == ord("a"):
                if current_idx < len(review_list):
                    current_boxes.append(box_raw)
                    current_idx += 1
            elif key == ord("d"):
                if current_idx < len(review_list):
                    current_idx += 1
                elif current_boxes:
                    current_boxes.pop(final_idx)
                    final_idx = min(final_idx, len(current_boxes) - 1)
            elif key == ord("b"):
                if current_idx < len(review_list):
                    current_idx = max(0, current_idx - 1)
                else:
                    current_idx = len(review_list) - 1
            elif key == 81 or key == ord("h"):
                if current_idx >= len(review_list) and current_boxes:
                    final_idx = max(0, final_idx - 1)
            elif key == 83 or key == ord("l"):
                if current_idx >= len(review_list) and current_boxes:
                    final_idx = min(len(current_boxes) - 1, final_idx + 1)
            elif key == ord("s"):
                self.save_labels(current_boxes, stem)
                break
            elif key == ord("k"):
                break
            elif key == ord("f"):
                if current_idx < len(review_list):
                    for idx in range(current_idx, len(review_list)):
                        source, source_idx = review_list[idx]
                        if source == "yolo":
                            current_boxes.append(yolo_boxes[source_idx])
                        else:
                            current_boxes.append(dino_boxes[source_idx])

                self.save_labels(current_boxes, stem)
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                sys.exit(0)

        cv2.destroyAllWindows()
        return self.config.output_labels / f"{stem}.txt"

    def run(self) -> None:
        """Review all images in the configured source directory."""
        for img_path in self.config.image_source.glob("*.jpg"):
            self.review_image(img_path)


def main() -> None:
    """Run the manual review UI with the repository default locations."""
    config = ManualReviewConfig(
        image_source=REPO_ROOT / "dataset" / "cherry-detection-1" / "valid" / "images",
        dino_labels=REPO_ROOT / "dataset" / "dino_out" / "labels",
        yolo_labels=REPO_ROOT / "dataset" / "yolo_predictions" / "yolo_pt_pred" / "labels",
        output_labels=REPO_ROOT / "dataset" / "autolabels" / "manual_review_labels",
    )
    ManualReviewSession(config).run()


if __name__ == "__main__":
    main()