from ultralytics import YOLO

from core.config import (
    DETECTION_MODEL,
    DEVICE,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
)


class WasteDetector:

    def __init__(self):
        if not DETECTION_MODEL.exists():
            raise FileNotFoundError(
                f"Detection model not found: {DETECTION_MODEL}"
            )
        self.model = YOLO(str(DETECTION_MODEL))

    def detect(self, frame):
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].cpu())
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

                # A detection without a ByteTrack ID is still a valid YOLO
                # detection and must remain visible in frame-level review.
                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0].cpu())

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(box.conf[0].cpu()),
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "track_id": track_id,
                })

        return detections
