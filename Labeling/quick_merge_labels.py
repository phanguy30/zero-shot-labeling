from pathlib import Path

# -------- PATHS --------
REPO_ROOT = Path(__file__).parent.parent

IMAGE_SOURCE = REPO_ROOT / "dataset" / "cherry-detection-1" / "valid" / "images"
DINO_LABELS = REPO_ROOT / "dataset" / "dino_out" / "labels"
YOLO_LABELS = REPO_ROOT / "dataset" / "yolo_predictions" / "yolo_pt_pred" / "labels"
MERGED_LABELS = REPO_ROOT / "dataset" / "autolabels" / "merged_labels"
MERGED_LABELS.mkdir(parents=True, exist_ok=True)


# -------- HELPERS --------
def read_labels(label_path):
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            class_id, x, y, w, h = values[:5]

            boxes.append({
                "class": int(class_id),
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })
    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]

    x1 = (x - w/2) * img_w
    y1 = (y - h/2) * img_h
    x2 = (x + w/2) * img_w
    y2 = (y + h/2) * img_h

    return [x1, y1, x2, y2]


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def save_labels(boxes, path):
    with open(path, "w") as f:
        for b in boxes:
            f.write(f"{b['class']} {b['x']} {b['y']} {b['w']} {b['h']}\n")


# -------- MAIN --------
for img_path in IMAGE_SOURCE.glob("*.jpg"):
    stem = img_path.stem
    print(f"Merging {stem}...")

    # fake image size (YOLO format is normalized, so we can use 1x1)
    w, h = 1, 1

    yolo_boxes = read_labels(YOLO_LABELS / f"{stem}.txt")
    dino_boxes = read_labels(DINO_LABELS / f"{stem}.txt")

    yolo_xy = [yolo_to_xyxy(b, w, h) for b in yolo_boxes]
    dino_xy = [yolo_to_xyxy(b, w, h) for b in dino_boxes]

    matched_dino = set()
    merged = []

    # match YOLO → DINO, then keep YOLO if there is match, otherwise keep YOLO anyway
    for i, yb in enumerate(yolo_xy):
        best_iou = 0
        best_j = -1

        for j, db in enumerate(dino_xy):
            if j in matched_dino:
                continue

            score = iou(yb, db)
            if score > best_iou:
                best_iou = score
                best_j = j

        if best_iou > 0.5:
            merged.append(yolo_boxes[i])  # keep YOLO
            matched_dino.add(best_j)
        else:
            merged.append(yolo_boxes[i])  # unmatched YOLO still kept

    # add unmatched DINO
    for j, box in enumerate(dino_boxes):
        if j not in matched_dino:
            merged.append(box)

    # save
    save_labels(merged, MERGED_LABELS / f"{stem}.txt")