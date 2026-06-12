from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = PROJECT_ROOT / "dataset"
OUT_DATASET = PROJECT_ROOT / "dataset_video_finetune"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_TEACHER = PROJECT_ROOT / "runs" / "detect" / "IMG1561_meerkat_detector" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a YOLO finetuning dataset from unlabeled videos.")
    parser.add_argument("--video-dir", type=Path, default=Path("Z:/video"))
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--out", type=Path, default=OUT_DATASET)
    parser.add_argument("--sample-every-sec", type=float, default=5.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-det", type=int, default=6)
    return parser.parse_args()


def reset_yolo_dirs(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    for subdir in ["images/train", "images/val", "labels/train", "labels/val"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)


def copy_source_dataset(source: Path, out: Path) -> list[dict[str, str]]:
    rows = []
    for split in ["train", "val"]:
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        for image_path in sorted(image_dir.glob("*.jpg")):
            label_path = label_dir / f"{image_path.stem}.txt"
            out_image = out / "images" / split / image_path.name
            out_label = out / "labels" / split / label_path.name
            shutil.copy2(image_path, out_image)
            shutil.copy2(label_path, out_label)
            rows.append({
                "image": image_path.name,
                "split": split,
                "source": "original_IMG1561_dataset",
                "time_sec": "",
                "detections": str(count_label_lines(label_path)),
            })
    return rows


def count_label_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def write_data_yaml(out: Path) -> None:
    (out / "data.yaml").write_text(
        "path: dataset_video_finetune\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: meerkat\n",
        encoding="utf-8",
    )


def write_yolo_label(path: Path, boxes, width: int, height: int) -> int:
    lines = []
    for box in boxes:
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(width), x2), min(float(height), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height
        box_w = (x2 - x1) / width
        box_h = (y2 - y1) / height
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def predict_batch(model: YOLO, batch: list[tuple[Path, np.ndarray]], args: argparse.Namespace) -> list[tuple[Path, np.ndarray, object]]:
    images = [frame for _, frame in batch]
    results = model.predict(
        source=images,
        conf=args.conf,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device="cpu",
        verbose=False,
    )
    return [(path, frame, result) for (path, frame), result in zip(batch, results)]


def add_video_frames(args: argparse.Namespace, model: YOLO, out: Path) -> list[dict[str, str]]:
    rows = []
    videos = sorted(args.video_dir.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No .mp4 files found in {args.video_dir}")

    for video_index, video_path in enumerate(videos):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(fps * args.sample_every_sec)))
        safe_stem = video_path.stem.replace(" ", "_").replace("-", "_")

        batch: list[tuple[Path, np.ndarray]] = []
        frame_index = 0
        sampled = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                sec = frame_index / fps if fps else float(sampled * args.sample_every_sec)
                split = "val" if (sampled + video_index) % 5 == 0 else "train"
                image_name = f"{safe_stem}_t{int(round(sec)):05d}.jpg"
                image_path = out / "images" / split / image_name
                cv2.imwrite(str(image_path), frame)
                batch.append((image_path, frame))
                sampled += 1

                if len(batch) >= args.batch:
                    rows.extend(write_batch_labels(model, batch, args, out, split_by_path=True))
                    batch.clear()
            frame_index += 1

        if batch:
            rows.extend(write_batch_labels(model, batch, args, out, split_by_path=True))
        cap.release()

        print({
            "video": str(video_path),
            "fps": round(fps, 3),
            "frames": total_frames,
            "sampled": sampled,
        })
    return rows


def split_from_image_path(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    return "val" if "val" in parts else "train"


def write_batch_labels(model: YOLO, batch: list[tuple[Path, np.ndarray]], args: argparse.Namespace, out: Path, split_by_path: bool) -> list[dict[str, str]]:
    rows = []
    for image_path, frame, result in predict_batch(model, batch, args):
        split = split_from_image_path(image_path) if split_by_path else "train"
        label_path = out / "labels" / split / f"{image_path.stem}.txt"
        height, width = frame.shape[:2]
        detections = write_yolo_label(label_path, result.boxes, width, height)
        rows.append({
            "image": image_path.name,
            "split": split,
            "source": "teacher_model_pseudo_label",
            "time_sec": image_path.stem.rsplit("_t", 1)[-1],
            "detections": str(detections),
        })
    return rows


def make_preview(out: Path, rows: list[dict[str, str]], max_images: int = 40) -> None:
    selected = [row for row in rows if row["source"] == "teacher_model_pseudo_label" and row["detections"] != "0"]
    if not selected:
        return
    every = max(1, len(selected) // max_images)
    thumbs = []
    for row in selected[::every][:max_images]:
        split = row["split"]
        image_path = out / "images" / split / row["image"]
        label_path = out / "labels" / split / f"{Path(row['image']).stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, xc, yc, bw, bh = map(float, parts)
            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, f"{row['image']} boxes={row['detections']}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        thumbs.append(cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA))

    if not thumbs:
        return
    cols = 4
    rows_count = (len(thumbs) + cols - 1) // cols
    canvas = np.full((rows_count * 270, cols * 480, 3), 255, dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        row, col = divmod(index, cols)
        canvas[row * 270:(row + 1) * 270, col * 480:(col + 1) * 480] = thumb
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(RESULTS_DIR / "video_finetune_pseudo_label_preview.jpg"), canvas)


def write_summary(out: Path, rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    annotations_csv = RESULTS_DIR / "video_finetune_dataset_manifest.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with annotations_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "split", "source", "time_sec", "detections"])
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    train = sum(1 for row in rows if row["split"] == "train")
    val = sum(1 for row in rows if row["split"] == "val")
    pseudo = [row for row in rows if row["source"] == "teacher_model_pseudo_label"]
    boxes = sum(int(row["detections"]) for row in rows)
    pseudo_boxes = sum(int(row["detections"]) for row in pseudo)
    summary = (
        "Video finetune dataset prepared.\n"
        f"Dataset: {out}\n"
        f"Video source: {args.video_dir}\n"
        f"Teacher: {args.teacher}\n"
        f"Sample every seconds: {args.sample_every_sec}\n"
        f"Confidence threshold: {args.conf}\n"
        f"Images total/train/val: {total}/{train}/{val}\n"
        f"Pseudo images: {len(pseudo)}\n"
        f"Boxes total/pseudo: {boxes}/{pseudo_boxes}\n"
        "Labels from the new videos are pseudo-labels generated by the previous detector.\n"
    )
    (RESULTS_DIR / "video_finetune_dataset_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def main() -> None:
    args = parse_args()
    if not args.teacher.exists():
        raise RuntimeError(f"Teacher weights not found: {args.teacher}")
    reset_yolo_dirs(args.out)
    rows = copy_source_dataset(SOURCE_DATASET, args.out)
    model = YOLO(str(args.teacher))
    rows.extend(add_video_frames(args, model, args.out))
    write_data_yaml(args.out)
    make_preview(args.out, rows)
    write_summary(args.out, rows, args)


if __name__ == "__main__":
    main()
