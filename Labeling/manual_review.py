from pathlib import Path
import sys
import cv2

# -------- PATHS --------
REPO_ROOT = Path(__file__).parent.parent

IMAGE_SOURCE = REPO_ROOT / "dataset" / "cherry-detection-1" / "valid" / "images"
DINO_LABELS = REPO_ROOT / "dataset" / "dino_out" / "labels"
YOLO_LABELS = REPO_ROOT / "dataset" / "yolo_predictions" / "yolo_pt_pred" / "labels"
NEW_LABELS_DIR = REPO_ROOT / "dataset" / "autolabels" / "manual_review_labels"
NEW_LABELS_DIR.mkdir(parents=True, exist_ok=True)


# Read labels
def read_labels(label_path):
    boxes = []
    
    if not label_path.exists():
        return boxes
    
    with open(label_path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            
            class_id, x, y, w, h = values[:5]
            conf = values[5] if len(values) == 6 else None
            
            boxes.append({
                "class": int(class_id),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "conf": conf
            })
    
    return boxes


# Convert YOLO format (x_center, y_center, width, height) to (x1, y1, x2, y2)
def yolo_to_xyxy(box, img_w, img_h):
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    
    x1 = int((x - w/2) * img_w)
    y1 = int((y - h/2) * img_h)
    x2 = int((x + w/2) * img_w)
    y2 = int((y + h/2) * img_h)
    
    return [x1, y1, x2, y2]


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - inter_area
    
    return inter_area / union if union > 0 else 0


def save_labels(boxes, stem):
    label_path = NEW_LABELS_DIR / f"{stem}.txt"
    with open(label_path, "w") as f:
        for box in boxes:
            line = f"{box['class']} {box['x']} {box['y']} {box['w']} {box['h']}\n"
            f.write(line)


if __name__ == "__main__":
    for img_path in IMAGE_SOURCE.glob("*.jpg"):
        print(f"Processing {img_path.name}...")
        
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        stem = img_path.stem
        
        yolo_boxes = read_labels(YOLO_LABELS / f"{stem}.txt")
        dino_boxes = read_labels(DINO_LABELS / f"{stem}.txt")
        
        yolo_xy = [yolo_to_xyxy(b, w, h) for b in yolo_boxes]
        dino_xy = [yolo_to_xyxy(b, w, h) for b in dino_boxes]
        
        # --- matching ---
        matched_yolo = set()
        matched_dino = set()
        matches = set()

        for i, yb in enumerate(yolo_xy):
            best_iou = 0
            best_j = -1
            
            for j, db in enumerate(dino_xy):
                score = iou(yb, db)
                if score > best_iou:
                    best_iou = score
                    best_j = j
            
            if best_iou > 0.5:
                matches.add((i, best_j, best_iou))
                matched_yolo.add(i)
                matched_dino.add(best_j)

        yolo_only = [i for i in range(len(yolo_boxes)) if i not in matched_yolo]
        dino_only = [j for j in range(len(dino_boxes)) if j not in matched_dino]

        needs_review = len(matches) < len(yolo_boxes) or len(matches) < len(dino_boxes)

        if not needs_review:
            continue

        print(f"[REVIEW] {stem}")
        print(f"  - {len(matches)} matches, {len(yolo_only)} YOLO-only, {len(dino_only)} DINO-only")

        # --- init ---
        current_boxes = []

        # auto-accept matches
        for i, j, score in matches:
            current_boxes.append(yolo_boxes[i])

        review_list = (
            [("yolo", i) for i in yolo_only] +
            [("dino", j) for j in dino_only]
        )

        current_idx = 0
        final_idx = 0

        # --- UI loop ---
        while True:
            canvas = img.copy()

            # -----------------------------
            # PHASE 1: Candidate Review
            # -----------------------------
            if current_idx < len(review_list):

                # accepted (blue)
                for box in current_boxes:
                    x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
                    cv2.rectangle(canvas, (x1,y1), (x2,y2), (255,0,0), 2)

                # candidates
                for i in yolo_only:
                    x1, y1, x2, y2 = yolo_xy[i]
                    cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,100,0), 2)

                for j in dino_only:
                    x1, y1, x2, y2 = dino_xy[j]
                    cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,0,100), 2)

                # current highlight
                source, idx = review_list[current_idx]

                if source == "yolo":
                    box_xy = yolo_xy[idx]
                    box_raw = yolo_boxes[idx]
                    color = (0,255,0)
                else:
                    box_xy = dino_xy[idx]
                    box_raw = dino_boxes[idx]
                    color = (0,0,255)

                x1, y1, x2, y2 = box_xy
                cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,255,255), 3)

                cv2.putText(
                    canvas,
                    f"{source.upper()} {current_idx+1}/{len(review_list)} | accepted: {len(current_boxes)}",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    2
                )

            # -----------------------------
            # PHASE 2: FINAL REVIEW
            # -----------------------------
            else:
                cv2.putText(
                    canvas,
                    f"FINAL REVIEW {final_idx+1}/{len(current_boxes)} | arrows/h/l move | d delete | b back | s save",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,255),
                    2
                )

                for i, box in enumerate(current_boxes):
                    x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)

                    if i == final_idx:
                        cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,255,255), 3)
                    else:
                        cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,255,255), 2)

            # -----------------------------
            # SHOW
            # -----------------------------
            cv2.imshow("Annotator", canvas)
            key = cv2.waitKey(0)

            # -----------------------------
            # CONTROLS
            # -----------------------------

            # ACCEPT
            if key == ord("a"):
                if current_idx < len(review_list):
                    current_boxes.append(box_raw)
                    current_idx += 1

            # DISCARD / DELETE
            elif key == ord("d"):
                if current_idx < len(review_list):
                    current_idx += 1
                else: #handle delete in review mode
                    if current_boxes:
                        current_boxes.pop(final_idx)
                        final_idx = min(final_idx, len(current_boxes) - 1)

            # BACK
            elif key == ord("b"):
                if current_idx < len(review_list):
                    current_idx = max(0, current_idx - 1)
                else:
                    current_idx = len(review_list) - 1

            # MOVE IN FINAL REVIEW
            elif key == 81 or key == ord("h"):  # left
                if current_idx >= len(review_list) and current_boxes:
                    final_idx = max(0, final_idx - 1)

            elif key == 83 or key == ord("l"):  # right
                if current_idx >= len(review_list) and current_boxes:
                    final_idx = min(len(current_boxes) - 1, final_idx + 1)

            # SAVE
            elif key == ord("s"):
                save_labels(current_boxes, stem)
                break

            # SKIP IMAGE
            elif key == ord("k"):
                break
            
            #SAVE ALL OF REMAINING
            elif key == ord("f"):  # SAVE ALL (accept everything)
                # If still in candidate review, add all remaining boxes
                if current_idx < len(review_list):
                    for k in range(current_idx, len(review_list)):
                        source, idx = review_list[k]

                        if source == "yolo":
                            current_boxes.append(yolo_boxes[idx])
                        else:
                            current_boxes.append(dino_boxes[idx])

                save_labels(current_boxes, stem)
                break

            # QUIT
            elif key == ord("q"):
                cv2.destroyAllWindows()
                sys.exit(0)

        cv2.destroyAllWindows()