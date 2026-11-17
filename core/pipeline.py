
import cv2
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

    def process_video(self, video_path, progress_callback=None):

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output_path = OUTPUT_DIR / f"{video_path.stem}_annotated.mp4"

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
            raise RuntimeError("Unable to create output video.")

        object_records = {}
        frame_number = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_number += 1

                detections = self.detector.detect(frame)
                tracked_objects = self.tracker.update(detections)

                for obj in tracked_objects:

                    track_id = obj["track_id"]

                    x1, y1, x2, y2 = obj["bbox"]

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    if track_id not in object_records:

                        crop = frame[y1:y2, x1:x2]

                        if crop.size == 0:
                            continue

                        classification = self.classifier.classify(crop)

                        object_records[track_id] = {
                            "object_id": track_id,
                            "category": classification["category"],
                            "confidence": classification["confidence"],
                            "recyclability": classification["recyclability"],
                            "detection_confidence": round(
                                obj["confidence"] * 100, 2
                            ),
                        }

                    category = object_records[track_id]["category"]

                    label = f"ID {track_id}: {category}"

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                total_count = self.tracker.get_total_count()

                cv2.putText(
                    frame,
                    f"Unique Objects: {total_count}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

                writer.write(frame)

                if progress_callback and total_frames > 0:
                    progress_callback(
                        min(frame_number / total_frames, 1.0)
                    )

        finally:
            cap.release()
            writer.release()

        return {
            "video_path": str(output_path),
            "total_objects": self.tracker.get_total_count(),
            "classified_objects": len(object_records),
            "objects": list(object_records.values()),
            "processed_frames": frame_number,
        }
