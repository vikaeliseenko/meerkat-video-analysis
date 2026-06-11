from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_CSV = RESULTS_DIR / "IMG1561_raw_frame_results.csv"
VIDEO_PATH = RESULTS_DIR / "IMG1561_annotated_video.mp4"
ANALYSIS_FPS = 3
BIN_SIZE_SEC = 30


def save_behavior_summary(raw):
    summary = raw.groupby("behavior").size().sort_values(ascending=False).reset_index(name="count_records")
    summary["duration_event_sec"] = summary["count_records"] / ANALYSIS_FPS
    summary["percent_of_records"] = (summary["count_records"] / len(raw) * 100).round(2)
    out = RESULTS_DIR / "IMG1561_behavior_event_summary.csv"
    summary.to_csv(out, index=False, encoding="utf-8-sig")
    return summary


def save_interval_summary(raw):
    raw = raw.copy()
    raw["time_bin_start"] = (raw["time_sec"] // BIN_SIZE_SEC) * BIN_SIZE_SEC
    raw["time_bin_end"] = raw["time_bin_start"] + BIN_SIZE_SEC
    grouped = raw.groupby(["time_bin_start", "time_bin_end", "behavior"]).size().reset_index(name="count_records")
    grouped["duration_event_sec"] = grouped["count_records"] / ANALYSIS_FPS
    totals = grouped.groupby("time_bin_start")["count_records"].transform("sum")
    grouped["percent_within_interval_records"] = (grouped["count_records"] / totals * 100).round(2)
    grouped["time_bin"] = grouped["time_bin_start"].astype(int).astype(str) + "-" + grouped["time_bin_end"].astype(int).astype(str) + " sec"
    out = RESULTS_DIR / "IMG1561_interval_behavior_summary.csv"
    grouped.to_csv(out, index=False, encoding="utf-8-sig")
    return grouped


def plot_behavior_counts(summary):
    plt.figure(figsize=(9, 5))
    plt.bar(summary["behavior"], summary["count_records"], color="#5B8FF9")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Record count")
    plt.title("Behavior categories from automatic video analysis")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "IMG1561_behavior_counts.png", dpi=200)
    plt.close()


def plot_interval_stack(interval_summary):
    pivot = interval_summary.pivot_table(
        index="time_bin",
        columns="behavior",
        values="duration_event_sec",
        aggfunc="sum",
        fill_value=0,
    )
    ordered_bins = interval_summary.sort_values("time_bin_start")["time_bin"].drop_duplicates()
    pivot = pivot.loc[ordered_bins]
    ax = pivot.plot(kind="bar", stacked=True, figsize=(13, 6), colormap="tab20")
    ax.set_ylabel("Summed event-observation seconds")
    ax.set_xlabel("Interval")
    ax.set_title("Dynamics of automatically registered behavior categories")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "IMG1561_interval_behavior_stacked.png", dpi=200)
    plt.close()


def plot_trajectory(raw):
    points = raw.dropna(subset=["x_center", "y_center"])
    plt.figure(figsize=(10, 6))
    if not points.empty:
        scatter = plt.scatter(
            points["x_center"],
            points["y_center"],
            c=points["time_sec"],
            s=8,
            cmap="viridis",
            alpha=0.65,
        )
        plt.colorbar(scatter, label="Time, s")
    plt.gca().invert_yaxis()
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.xlabel("X, pixels")
    plt.ylabel("Y, pixels")
    plt.title("Trajectories of detected bounding-box centers")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "IMG1561_detection_centers_trajectory.png", dpi=200)
    plt.close()


def make_annotated_contact_sheet():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open annotated video: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0
    seconds = list(range(0, int(duration) + 1, 60))
    frames = []
    for sec in seconds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.putText(frame, f"{sec}s", (24, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
        frames.append(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))
    cap.release()

    cols = 2
    rows = (len(frames) + cols - 1) // cols
    canvas = np.full((rows * 360, cols * 640, 3), 255, dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, col = divmod(index, cols)
        canvas[row * 360:(row + 1) * 360, col * 640:(col + 1) * 640] = frame
    cv2.imwrite(str(RESULTS_DIR / "IMG1561_annotated_contact_sheet.jpg"), canvas)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_CSV)
    summary = save_behavior_summary(raw)
    interval_summary = save_interval_summary(raw)
    plot_behavior_counts(summary)
    plot_interval_stack(interval_summary)
    plot_trajectory(raw)
    make_annotated_contact_sheet()
    print("Saved visual summaries to", RESULTS_DIR)


if __name__ == "__main__":
    main()
