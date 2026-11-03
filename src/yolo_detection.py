"""
YOLO object detection for CCA 3 (Task 3).
Run from project root:
    python src/yolo_detection.py
The first run downloads yolov8n.pt automatically (internet required).
"""
from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "task3"
OUT.mkdir(parents=True, exist_ok=True)

# Use the supplied photographs and the supplied video.
IMAGE_FILES = [
    DATA / "original_portrait.png",
    DATA / "second_same_scene_photo.jpg",
]
VIDEO_FILE = DATA / "original_video.mp4"

# Small COCO pretrained model; Ultralytics downloads the weights on first run.
MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.25
model = YOLO(MODEL_NAME)

rows = []
for image_path in IMAGE_FILES:
    if not image_path.exists():
        print(f"Skipping missing image: {image_path}")
        continue
    results = model.predict(source=str(image_path), conf=CONFIDENCE, verbose=False)
    result = results[0]
    annotated = result.plot()
    output_path = OUT / f"yolo_{image_path.stem}.jpg"
    cv2.imwrite(str(output_path), annotated)

    names = result.names
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [float(x) for x in box.xyxy[0].tolist()]
            rows.append({
                "source": image_path.name,
                "type": "image",
                "class_id": cls_id,
                "class_name": names[cls_id],
                "confidence": round(conf, 5),
                "x1": round(x1, 2), "y1": round(y1, 2),
                "x2": round(x2, 2), "y2": round(y2, 2),
                "annotated_output": output_path.name,
            })
    print(f"Saved {output_path.name}: {len(result.boxes) if result.boxes is not None else 0} detections")

# Sample the supplied video at evenly spaced timestamps, capped at 8 frames.
if VIDEO_FILE.exists():
    cap = cv2.VideoCapture(str(VIDEO_FILE))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_FILE}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    sample_count = min(8, max(1, int(duration // 2) + 1))
    if frame_count > 0:
        indices = sorted(set(
            round(i * (frame_count - 1) / max(1, sample_count - 1))
            for i in range(sample_count)
        ))
    else:
        indices = [0]

    for sample_no, frame_index in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        result = model.predict(source=frame, conf=CONFIDENCE, verbose=False)[0]
        annotated = result.plot()
        output_path = OUT / f"yolo_video_frame_{sample_no:02d}.jpg"
        cv2.imwrite(str(output_path), annotated)
        timestamp = frame_index / fps if fps > 0 else 0

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [float(x) for x in box.xyxy[0].tolist()]
                rows.append({
                    "source": VIDEO_FILE.name,
                    "type": f"video_frame_{sample_no}",
                    "timestamp_seconds": round(timestamp, 3),
                    "frame_index": frame_index,
                    "class_id": cls_id,
                    "class_name": result.names[cls_id],
                    "confidence": round(conf, 5),
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "x2": round(x2, 2), "y2": round(y2, 2),
                    "annotated_output": output_path.name,
                })
        print(f"Saved {output_path.name} at {timestamp:.2f}s")
    cap.release()
else:
    print(f"Video not found: {VIDEO_FILE}")

csv_path = OUT / "yolo_detections.csv"
columns = [
    "source", "type", "timestamp_seconds", "frame_index",
    "class_id", "class_name", "confidence",
    "x1", "y1", "x2", "y2", "annotated_output",
]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

# Summarize per-class counts.
counts = {}
for row in rows:
    key = row["class_name"]
    counts[key] = counts.get(key, 0) + 1
with (OUT / "yolo_class_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["class_name", "detection_count"])
    writer.writerows(sorted(counts.items(), key=lambda item: item[0]))

print(f"\nYOLO inference complete. {len(rows)} detections recorded.")
print(f"Results: {OUT}")
