from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = PROJECT_ROOT / "videos" / "IMG_1561.MP4"
OUT_DIR = PROJECT_ROOT / "frames_for_markup"
CONTACT_SHEET = PROJECT_ROOT / "results" / "IMG1561_contact_sheet.jpg"
OVERVIEW_20S = PROJECT_ROOT / "results" / "IMG1561_overview_20s.jpg"
FPS_OUT = 1


def make_contact_sheet(frames, out_path, thumb_size=(455, 256), cols=4):
    thumbs = []
    for sec, path in frames:
        image = cv2.imread(str(path))
        image = cv2.resize(image, thumb_size, interpolation=cv2.INTER_AREA)
        cv2.putText(image, f"{sec:03d}s", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        thumbs.append(image)
    rows = (len(thumbs) + cols - 1) // cols
    canvas = np.full((rows * thumb_size[1], cols * thumb_size[0], 3), 255, dtype=np.uint8)
    for idx, image in enumerate(thumbs):
        row, col = divmod(idx, cols)
        y1, y2 = row * thumb_size[1], (row + 1) * thumb_size[1]
        x1, x2 = col * thumb_size[0], (col + 1) * thumb_size[0]
        canvas[y1:y2, x1:x2] = image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps else 0
    step = max(1, int(round(fps / FPS_OUT)))

    saved = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step == 0:
            sec = int(round(frame_index / fps))
            out_path = OUT_DIR / f"IMG1561_t{sec:03d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append((sec, out_path))
        frame_index += 1
    cap.release()

    make_contact_sheet(saved, CONTACT_SHEET)
    overview = [(sec, path) for sec, path in saved if sec % 20 == 0]
    make_contact_sheet(overview, OVERVIEW_20S, thumb_size=(640, 360), cols=2)

    print({
        "video": str(VIDEO_PATH),
        "fps": round(fps, 3),
        "frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 3),
        "saved_frames": len(saved),
    })


if __name__ == "__main__":
    main()
