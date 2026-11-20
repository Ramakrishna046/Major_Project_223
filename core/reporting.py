import csv
import json
from pathlib import Path

from core.config import OUTPUT_DIR


def generate_reports(results, output_dir=None):
    report_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    objects = results["objects"]
    csv_path = report_dir / "waste_detection_report.csv"

    csv_columns = [
        "object_id",
        "category",
        "confidence",
        "recyclability",
        "detection_confidence",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(objects)

    category_counts = {}
    for obj in objects:
        category = obj["category"]
        category_counts[category] = category_counts.get(category, 0) + 1

    recyclable_count = sum(
        1 for obj in objects if obj["recyclability"] == "Recyclable"
    )
    non_recyclable_count = sum(
        1 for obj in objects if obj["recyclability"] == "Non-recyclable"
    )

    summary = {
        "total_unique_objects": results["total_objects"],
        "classified_objects": results["classified_objects"],
        "processed_frames": results["processed_frames"],
        "recyclable_objects": recyclable_count,
        "non_recyclable_objects": non_recyclable_count,
        "category_counts": category_counts,
        "objects": objects,
    }

    json_path = report_dir / "waste_detection_summary.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "summary": summary,
    }
