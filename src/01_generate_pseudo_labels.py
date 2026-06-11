from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = PROJECT_ROOT / "frames_for_markup"
DATASET_DIR = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results"


def reset_dataset():
    for subdir in ["images/train", "images/val", "labels/train", "labels/val"]:
        path = DATASET_DIR / subdir
        path.mkdir(parents=True, exist_ok=True)
        for item in path.glob("*"):
            if item.is_file():
                item.unlink()


def merge_boxes(boxes):
    merged = []
    for box in sorted(boxes, key=lambda item: (item[0], item[1])):
        x1, y1, x2, y2 = box
        merged_into_existing = False
        for index, current in enumerate(merged):
            ax1, ay1, ax2, ay2 = current
            ix1, iy1 = max(x1, ax1), max(y1, ay1)
            ix2, iy2 = min(x2, ax2), min(y2, ay2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            centers_close = abs((x1 + x2 - ax1 - ax2) / 2) < 80 and abs((y1 + y2 - ay1 - ay2) / 2) < 80
            if inter > 0 or centers_close:
                merged[index] = (min(x1, ax1), min(y1, ay1), max(x2, ax2), max(y2, ay2))
                merged_into_existing = True
                break
        if not merged_into_existing:
            merged.append(box)
    return merged


def detect_motion_boxes(frames):
    subtractor = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=20, detectShadows=True)
    raw = []
    for path in frames:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        fg_mask = subtractor.apply(image)
        mask = (fg_mask == 255).astype("uint8") * 255
        mask[:90, 880:] = 0
        mask[:, :70] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < 650 or area > 35000 or w < 14 or h < 20:
                continue
            aspect = w / h
            if aspect > 2.6 or aspect < 0.18:
                continue
            pad = int(max(w, h) * 0.12)
            boxes.append((max(0, x - pad), max(0, y - pad), min(width, x + w + pad), min(height, y + h + pad)))

        filtered = []
        for box in merge_boxes(boxes):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if 900 <= area <= 45000:
                filtered.append(box)
        raw.append((path, filtered))
    return raw


def add_anchor(raw, sec_range, box):
    start, end = sec_range
    x1, y1, x2, y2 = box
    for path, boxes in raw:
        sec = int(path.stem.split("t")[-1])
        if not (start <= sec <= end):
            continue
        duplicate = False
        for ax1, ay1, ax2, ay2 in boxes:
            ix1, iy1 = max(x1, ax1), max(y1, ay1)
            ix2, iy2 = min(x2, ax2), min(y2, ay2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            denom = min((x2 - x1) * (y2 - y1), (ax2 - ax1) * (ay2 - ay1))
            if denom and inter / denom > 0.35:
                duplicate = True
                break
        if not duplicate:
            boxes.append(box)


def add_recurring_anchor_boxes(raw):
    add_anchor(raw, (0, 35), (335, 300, 390, 405))
    add_anchor(raw, (0, 25), (490, 310, 545, 415))
    add_anchor(raw, (75, 150), (675, 95, 725, 195))
    add_anchor(raw, (315, 380), (445, 300, 530, 430))
    add_anchor(raw, (470, 530), (865, 310, 935, 470))
    add_anchor(raw, (470, 530), (600, 105, 660, 235))


def write_dataset(raw):
    annotations = []
    counts = {"train": 0, "val": 0, "boxes": 0, "blank": 0}
    for index, (path, boxes) in enumerate(raw):
        split = "val" if index % 5 == 0 else "train"
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        shutil.copy2(path, DATASET_DIR / "images" / split / path.name)
        label_path = DATASET_DIR / "labels" / split / f"{path.stem}.txt"

        filtered = []
        for x1, y1, x2, y2 in boxes:
            box_w, box_h = x2 - x1, y2 - y1
            if box_w * box_h > 50000 or box_w < 18 or box_h < 20:
                continue
            filtered.append((x1, y1, x2, y2))
        filtered = sorted(filtered, key=lambda box: -((box[2] - box[0]) * (box[3] - box[1])))[:4]

        lines = []
        sec = int(path.stem.split("t")[-1])
        for x1, y1, x2, y2 in filtered:
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            box_w = (x2 - x1) / width
            box_h = (y2 - y1) / height
            lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
            annotations.append({
                "image": path.name,
                "time_sec": sec,
                "class_id": 0,
                "class_name": "meerkat",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "split": split,
                "source": "motion_plus_anchor_pseudo",
            })
        label_path.write_text("\n".join(lines), encoding="utf-8")
        counts[split] += 1
        counts["boxes"] += len(lines)
        counts["blank"] += 0 if lines else 1
    return annotations, counts


def write_preview(raw):
    preview = []
    for path, boxes in raw:
        sec = int(path.stem.split("t")[-1])
        if sec % 10 != 0:
            continue
        image = cv2.imread(str(path))
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(image, f"{sec:03d}s boxes={len(boxes)}", (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        preview.append(cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA))
    cols = 2
    rows = (len(preview) + cols - 1) // cols
    canvas = np.full((rows * 360, cols * 640, 3), 255, dtype=np.uint8)
    for index, image in enumerate(preview):
        row, col = divmod(index, cols)
        canvas[row * 360:(row + 1) * 360, col * 640:(col + 1) * 640] = image
    cv2.imwrite(str(RESULTS_DIR / "IMG1561_pseudo_bbox_contact_sheet_10s.jpg"), canvas)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reset_dataset()
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    raw = detect_motion_boxes(frames)
    add_recurring_anchor_boxes(raw)
    annotations, counts = write_dataset(raw)

    csv_path = RESULTS_DIR / "IMG1561_pseudo_bbox_annotations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "time_sec", "class_id", "class_name", "x1", "y1", "x2", "y2", "split", "source"])
        writer.writeheader()
        writer.writerows(annotations)

    write_preview(raw)
    summary = (
        "Pseudo-label generation completed.\n"
        f"Frames: {len(frames)}\n"
        f"Train frames: {counts['train']}\n"
        f"Val frames: {counts['val']}\n"
        f"Total boxes: {counts['boxes']}\n"
        f"Blank frames: {counts['blank']}\n"
        "Method: MOG2 motion segmentation + broad manual anchor zones for recurring visible meerkat positions.\n"
        "Important: pseudo-labels are draft labels and require manual validation before scientific claims.\n"
    )
    (RESULTS_DIR / "IMG1561_pseudo_label_summary.txt").write_text(summary, encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
