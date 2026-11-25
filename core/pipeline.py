import cv2
import json
from pathlib import Path

from core.detector import WasteDetector
from core.classifier import WasteClassifier
from core.tracker import WasteTracker
from core.config import OUTPUT_DIR


class WasteProcessingPipeline:

    def __init__(self):
        self.detector = WasteDetector()
        self.classifier = WasteClassifier()
        self.tracker = WasteTracker()

    def process_video(self, video_path, progress_callback=None, output_dir=None):
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        run_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        run_dir.mkdir(parents=True, exist_ok=True)

        output_path = run_dir / f"{video_path.stem}_annotated.mp4"
        frame_review_path = run_dir / "frame_review.json"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Unable to open input video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Unable to create annotated video.")

        object_records = {}
        frame_records = []
        frame_number = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1

                # Keep all YOLO detections, including boxes without a ByteTrack ID.
                detections = self.detector.detect(frame)
                tracked_objects = self.tracker.update(detections)
                tracked_by_id = {
                    obj["track_id"]: obj for obj in tracked_objects
                    if obj.get("track_id") is not None
                }

                frame_detections = []

                for detection in detections:
                    x1, y1, x2, y2 = detection["bbox"]
                    track_id = detection.get("track_id")
                    track_info = None
                    if track_id is not None:
                        track_info = tracked_by_id.get(int(track_id))
                    is_confirmed = (
                        track_info is not None
                        and track_info.get("confirmed", False)
                    )
                    x1 = max(0, min(width, int(x1)))
                    y1 = max(0, min(height, int(y1)))
                    x2 = max(0, min(width, int(x2)))
                    y2 = max(0, min(height, int(y2)))

                    track_id = detection.get("track_id")
                    crop = frame[y1:y2, x1:x2]

                    classification = None
                    if crop.size > 0:
                        # Classify every detected crop in every frame so the review
                        # screen exposes frame-level classification changes.
                        classification = self.classifier.classify(crop)

                    if (track_id is not None and is_confirmed and classification is not None):
                        track_id = int(track_id)
                        if track_id not in object_records:
                            object_records[track_id] = {
                                "object_id": track_id,
                                "category": classification["category"],
                                "confidence": classification["confidence"],
                                "recyclability": classification["recyclability"],
                                "detection_confidence": round(
                                    detection["confidence"] * 100, 2
                                ),
                            }

                    category = classification["category"] if classification else "Unclassified"
                    class_confidence = classification["confidence"] if classification else None
                    recyclability = classification["recyclability"] if classification else "Unknown"

                    label_id = f"ID {track_id}" if track_id is not None else "Untracked"
                    label = f"{label_id}: {category}"

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                    frame_detections.append({
                        "track_id": track_id,
                        "yolo_class": detection.get("class_name"),
                        "detection_confidence_percent": round(
                            detection["confidence"] * 100, 2
                        ),
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "efficientnet_category": category,
                        "classification_confidence_percent": class_confidence,
                        "recyclability": recyclability,
                    })

                cv2.putText(
                    frame,
                    f"Unique Tracked Objects: {self.tracker.get_total_count()}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                writer.write(frame)

                frame_records.append({
                    "frame_number": frame_number,
                    "timestamp_seconds": round((frame_number - 1) / fps, 3),
                    "detections": frame_detections,
                })

                if progress_callback and total_frames > 0:
                    progress_callback(min(frame_number / total_frames, 1.0))

        finally:
            cap.release()
            writer.release()

        with open(frame_review_path, "w", encoding="utf-8") as file:
            json.dump(frame_records, file, indent=2)

        return {
            "input_video_path": str(video_path),
            "video_path": str(output_path),
            "frame_review_path": str(frame_review_path),
            "total_objects": self.tracker.get_total_count(),
            "classified_objects": len(object_records),
            "objects": list(object_records.values()),
            "processed_frames": frame_number,
        }
