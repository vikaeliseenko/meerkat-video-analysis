import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

from ultralytics import YOLO


VIDEO_PATH = PROJECT_ROOT / "videos" / "IMG_1561.MP4"
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "IMG1561_meerkat_detector" / "weights" / "best.pt"
ZONE_JSON = PROJECT_ROOT / "results" / "enrichment_zone.json"
OUTPUT_RAW = PROJECT_ROOT / "results" / "IMG1561_raw_frame_results.csv"
OUTPUT_TABLE = PROJECT_ROOT / "results" / "IMG1561_ethogram_table.csv"
OUTPUT_VIDEO = PROJECT_ROOT / "results" / "IMG1561_annotated_video.mp4"

ANALYSIS_FPS = 3
BIN_SIZE_SEC = 30
LOW_SPEED_THRESHOLD = 4.0
HIGH_SPEED_THRESHOLD = 18.0
MIN_BOX_AREA = 500
CONF_THRESHOLD = 0.20
ZONE_KEYS = {"x1", "y1", "x2", "y2"}
OPTIONAL_ZONE_KEYS = {"status", "note"}


def load_zone(frame_width, frame_height):
    zone = json.loads(ZONE_JSON.read_text(encoding="utf-8"))
    if not ZONE_KEYS.issubset(zone):
        raise RuntimeError(f"Missing required enrichment zone keys in {ZONE_JSON}")
    unknown = set(zone) - ZONE_KEYS - OPTIONAL_ZONE_KEYS
    if unknown:
        raise RuntimeError(f"Unknown enrichment zone keys: {sorted(unknown)}")
    for key in ZONE_KEYS:
        if not isinstance(zone[key], int):
            raise RuntimeError(f"Zone coordinate {key} must be an integer")
    if not (0 <= zone["x1"] < zone["x2"] < frame_width and 0 <= zone["y1"] < zone["y2"] < frame_height):
        raise RuntimeError(f"Zone {zone} is outside frame {frame_width}x{frame_height}")
    return zone


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def point_in_zone(point, zone):
    x, y = point
    return zone["x1"] <= x <= zone["x2"] and zone["y1"] <= y <= zone["y2"]


def speed(current, previous):
    if previous is None:
        return 0.0
    return float(np.hypot(current[0] - previous[0], current[1] - previous[1]))


def overlap_ratio(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    denom = min(box_area(box_a), box_area(box_b))
    return 0.0 if denom <= 0 else inter / denom


def classify(box, animal_speed, in_zone, overlaps):
    area = box_area(box)
    if area < MIN_BOX_AREA:
        return "uncertain", 0.40
    if overlaps >= 1:
        return "social_contact", 0.70
    if in_zone and animal_speed <= HIGH_SPEED_THRESHOLD:
        return "olfactory_interest", 0.85
    if animal_speed >= HIGH_SPEED_THRESHOLD:
        return "active_movement", 0.80
    if animal_speed <= LOW_SPEED_THRESHOLD:
        return "inactive", 0.75
    return "exploratory", 0.70


def aggregate(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time_bin_start"] = (df["time_sec"] // BIN_SIZE_SEC) * BIN_SIZE_SEC
    df["time_bin_end"] = df["time_bin_start"] + BIN_SIZE_SEC
    grouped = (
        df.groupby(["video_id", "time_bin_start", "time_bin_end", "individual_id", "behavior"], dropna=False)
        .agg(count_frames=("behavior", "count"), mean_confidence=("confidence", "mean"))
        .reset_index()
    )
    grouped["total_duration_sec"] = grouped["count_frames"] / ANALYSIS_FPS
    grouped["percent_time"] = (grouped["total_duration_sec"] / BIN_SIZE_SEC * 100).round(2)
    grouped["time_bin"] = grouped["time_bin_start"].astype(int).astype(str) + "-" + grouped["time_bin_end"].astype(int).astype(str) + " sec"
    cols = [
        "video_id",
        "time_bin",
        "time_bin_start",
        "time_bin_end",
        "individual_id",
        "behavior",
        "count_frames",
        "total_duration_sec",
        "percent_time",
        "mean_confidence",
    ]
    return grouped[cols]


def main():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model is missing: {MODEL_PATH}. Run train_yolo.py first.")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps / ANALYSIS_FPS)))
    zone = load_zone(width, height)

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), ANALYSIS_FPS, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {OUTPUT_VIDEO}")

    model = YOLO(str(MODEL_PATH))
    prev_centers = {}
    rows = []
    frame_index = 0
    video_id = VIDEO_PATH.stem

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step != 0:
            frame_index += 1
            continue

        time_sec = frame_index / fps
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=CONF_THRESHOLD, verbose=False)[0]
        boxes, ids = [], []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = np.arange(len(xyxy))
            for box, track_id in zip(xyxy, track_ids):
                boxes.append(tuple(map(float, box)))
                ids.append(int(track_id))

        cv2.rectangle(frame, (zone["x1"], zone["y1"]), (zone["x2"], zone["y2"]), (255, 255, 255), 2)
        cv2.putText(frame, "enrichment_zone", (zone["x1"], max(25, zone["y1"] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if not boxes:
            rows.append({
                "video_id": video_id,
                "time_sec": time_sec,
                "frame_index": frame_index,
                "individual_id": "focal_group",
                "behavior": "out_of_view",
                "confidence": 1.0,
                "x_center": np.nan,
                "y_center": np.nan,
                "speed": np.nan,
            })
            cv2.putText(frame, "out_of_view", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            for box, track_id in zip(boxes, ids):
                center = box_center(box)
                animal_speed = speed(center, prev_centers.get(track_id))
                prev_centers[track_id] = center
                in_zone = point_in_zone(center, zone)
                overlaps = sum(1 for other in boxes if other != box and overlap_ratio(box, other) >= 0.25)
                behavior, confidence = classify(box, animal_speed, in_zone, overlaps)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, f"ID {track_id} {behavior} {confidence:.2f}", (x1, max(25, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                rows.append({
                    "video_id": video_id,
                    "time_sec": time_sec,
                    "frame_index": frame_index,
                    "individual_id": f"meerkat_{track_id}",
                    "behavior": behavior,
                    "confidence": confidence,
                    "x_center": center[0],
                    "y_center": center[1],
                    "speed": animal_speed,
                })
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    pd.DataFrame(rows).to_csv(OUTPUT_RAW, index=False, encoding="utf-8-sig")
    aggregate(rows).to_csv(OUTPUT_TABLE, index=False, encoding="utf-8-sig")
    print("Saved:", OUTPUT_RAW)
    print("Saved:", OUTPUT_TABLE)
    print("Saved:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
