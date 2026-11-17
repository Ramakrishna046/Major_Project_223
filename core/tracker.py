
from collections import defaultdict


class WasteTracker:
    def __init__(self):
        self.track_history = defaultdict(list)
        self.counted_ids = set()
        self.total_unique_objects = 0

    def update(self, detections):
        """
        Update tracking using YOLO tracking IDs.

        detections: list of dictionaries containing
        bbox, confidence, class_id, class_name and track_id.
        """
        tracked_objects = []

        for detection in detections:
            track_id = detection.get("track_id")

            if track_id is None:
                continue

            track_id = int(track_id)

            self.track_history[track_id].append(
                detection["bbox"]
            )

            if track_id not in self.counted_ids:
                self.counted_ids.add(track_id)
                self.total_unique_objects += 1

            tracked_objects.append({
                **detection,
                "track_id": track_id,
            })

        return tracked_objects

    def get_total_count(self):
        return self.total_unique_objects
