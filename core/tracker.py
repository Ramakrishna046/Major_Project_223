
from collections import defaultdict


class WasteTracker:
    MIN_CONFIRMATION_HITS = 3

    def __init__(self):
        self.track_history = defaultdict(list)
        self.track_hits = defaultdict(int)
        self.counted_ids = set()
        self.total_unique_objects = 0

    def update(self, detections):
        tracked_objects = []

        for detection in detections:
            track_id = detection.get("track_id")

            if track_id is None:
                continue

            track_id = int(track_id)

            self.track_hits[track_id] += 1

            bbox = detection.get("bbox")
            if bbox is not None:
                self.track_history[track_id].append(bbox)

            confirmed = (
                self.track_hits[track_id]
                >= self.MIN_CONFIRMATION_HITS
            )

            if confirmed and track_id not in self.counted_ids:
                self.counted_ids.add(track_id)
                self.total_unique_objects += 1

            tracked_objects.append({
                **detection,
                "track_id": track_id,
                "confirmed": confirmed,
            })

        return tracked_objects

    def get_total_count(self):
        return self.total_unique_objects
