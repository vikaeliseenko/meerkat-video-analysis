from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

from ultralytics import YOLO


DEFAULT_DATA = PROJECT_ROOT / "dataset_video_finetune" / "data.yaml"
DEFAULT_BASE = PROJECT_ROOT / "runs" / "detect" / "IMG1561_meerkat_detector" / "weights" / "best.pt"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the meerkat detector on the video pseudo-label dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="video_finetuned_meerkat_detector")
    return parser.parse_args()


def count_non_empty_labels(label_dir: Path) -> int:
    return sum(1 for path in label_dir.glob("*.txt") if path.read_text(encoding="utf-8").strip())


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise RuntimeError(f"Dataset YAML not found: {args.data}. Run src/05_prepare_video_finetune_dataset.py first.")
    if not args.base.exists():
        raise RuntimeError(f"Base weights not found: {args.base}")

    train_labels = args.data.parent / "labels" / "train"
    annotated = count_non_empty_labels(train_labels)
    if annotated == 0:
        raise RuntimeError("No non-empty train labels found in dataset_video_finetune.")

    model = YOLO(str(args.base))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        name=args.name,
        project=str(RUNS_DIR),
        patience=max(3, args.epochs // 2),
        exist_ok=True,
        pretrained=True,
        plots=True,
    )
    print("Model:", RUNS_DIR / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
