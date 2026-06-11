import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

from ultralytics import YOLO


DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
BASE_MODEL = PROJECT_ROOT / "yolov8n.pt"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
RUN_NAME = "IMG1561_meerkat_detector"


def main():
    labels = list((PROJECT_ROOT / "dataset" / "labels" / "train").glob("*.txt"))
    annotated = [path for path in labels if path.read_text(encoding="utf-8").strip()]
    if not annotated:
        raise RuntimeError("No non-empty YOLO labels found. Run src/01_generate_pseudo_labels.py first.")

    model = YOLO(str(BASE_MODEL) if BASE_MODEL.exists() else "yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=35,
        imgsz=512,
        batch=16,
        device="cpu",
        workers=0,
        name=RUN_NAME,
        project=str(RUNS_DIR),
        patience=10,
        exist_ok=True,
        pretrained=True,
        plots=True,
    )
    print("Model:", RUNS_DIR / RUN_NAME / "weights" / "best.pt")


if __name__ == "__main__":
    main()
