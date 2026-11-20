import csv
import json
import uuid
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from core.pipeline import WasteProcessingPipeline
from core.reporting import generate_reports
from core.config import OUTPUT_DIR


st.set_page_config(
    page_title="AI Waste Management System",
    page_icon="♻️",
    layout="wide",
)

st.title("♻️ AI-Driven Automated Waste Management System")
st.write("Upload a video to detect, track, classify and count waste objects.")
st.caption("Frame Review lets you inspect YOLO detections and EfficientNet predictions individually.")

st.divider()

uploaded_video = st.file_uploader(
    "Upload Waste Video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_video is not None:
    st.video(uploaded_video)

    if st.button("Start Waste Detection", type="primary"):
        progress_bar = st.progress(0)
        status = st.empty()

        try:
            run_id = uuid.uuid4().hex[:10]
            run_dir = OUTPUT_DIR / "frame_review_runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=False)

            suffix = Path(uploaded_video.name).suffix.lower() or ".mp4"
            input_path = run_dir / f"original{suffix}"
            input_path.write_bytes(uploaded_video.getvalue())

            status.info("Loading trained AI models...")
            pipeline = WasteProcessingPipeline()

            status.info("Processing video and recording frame-level predictions...")
            results = pipeline.process_video(
                input_path,
                progress_callback=progress_bar.progress,
                output_dir=run_dir,
            )

            status.info("Generating reports...")
            reports = generate_reports(results, output_dir=run_dir)

            st.session_state["results"] = results
            st.session_state["reports"] = reports
            st.session_state["review_run_dir"] = str(run_dir)
            st.session_state["review_frame_index"] = 0
            st.session_state["review_marks"] = []

            progress_bar.progress(1.0)
            status.success("Video processing completed. Open Frame-by-Frame Review below.")

        except Exception as error:
            status.error(f"Processing failed: {error}")


if "results" in st.session_state:
    results = st.session_state["results"]
    reports = st.session_state["reports"]
    summary = reports["summary"]

    st.divider()
    st.header("Detection Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Tracked Objects", summary["total_unique_objects"])
    col2.metric("Recyclable Objects", summary["recyclable_objects"])
    col3.metric("Non-recyclable Objects", summary["non_recyclable_objects"])

    st.subheader("Waste Category Distribution")
    category_counts = summary["category_counts"]

    if category_counts:
        category_df = pd.DataFrame(
            list(category_counts.items()),
            columns=["Category", "Count"],
        )
        st.bar_chart(category_df.set_index("Category"))
        st.dataframe(category_df, use_container_width=True)
    else:
        st.info("No classified waste objects were found.")

    st.subheader("Object Details")
    if results["objects"]:
        st.dataframe(pd.DataFrame(results["objects"]), use_container_width=True)
    else:
        st.info("No tracked objects were recorded.")

    # Frame-by-frame inspection
    st.divider()
    st.header("Frame-by-Frame Model Error Review")

    original_path = Path(results["input_video_path"])
    annotated_path = Path(results["video_path"])
    frame_json_path = Path(results["frame_review_path"])

    with open(frame_json_path, "r", encoding="utf-8") as file:
        frame_records = json.load(file)

    total_frames = len(frame_records)
    if total_frames:
        current_index = int(st.session_state.get("review_frame_index", 0))
        current_index = max(0, min(current_index, total_frames - 1))

        nav1, nav2, nav3 = st.columns([1, 3, 1])
        with nav1:
            if st.button("← Previous", disabled=current_index == 0, use_container_width=True):
                current_index = max(0, current_index - 1)
                st.session_state["review_frame_slider"] = current_index + 1
        with nav3:
            if st.button("Next →", disabled=current_index >= total_frames - 1, use_container_width=True):
                current_index = min(total_frames - 1, current_index + 1)
                st.session_state["review_frame_slider"] = current_index + 1

        with nav2:
            selected_frame = st.slider(
                "Frame number",
                min_value=1,
                max_value=total_frames,
                value=current_index + 1,
                key="review_frame_slider",
            )
            current_index = selected_frame - 1

        st.session_state["review_frame_index"] = current_index
        frame_info = frame_records[current_index]

        def read_frame(video_path, frame_no):
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                return None
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = capture.read()
            capture.release()
            if not ok:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        original_frame = read_frame(original_path, current_index)
        annotated_frame = read_frame(annotated_path, current_index)

        st.caption(
            f"Frame {current_index + 1} of {total_frames} "
            f"| Timestamp: {frame_info.get('timestamp_seconds', 0):.2f} seconds "
            f"| Detections: {len(frame_info.get('detections', []))}"
        )

        image_col1, image_col2 = st.columns(2)
        with image_col1:
            st.markdown("**Original frame**")
            if original_frame is not None:
                st.image(original_frame, use_container_width=True)
            else:
                st.warning("Could not read the original frame.")
        with image_col2:
            st.markdown("**YOLO + EfficientNet annotated frame**")
            if annotated_frame is not None:
                st.image(annotated_frame, use_container_width=True)
            else:
                st.warning("Could not read the annotated frame.")

        st.markdown("**Model predictions in this frame**")
        frame_detections = frame_info.get("detections", [])
        if frame_detections:
            st.dataframe(pd.DataFrame(frame_detections), use_container_width=True)
        else:
            st.warning("No detections were produced for this frame. Check the original frame for possible missed objects.")

        st.info(
            "A missed object cannot be identified automatically from model predictions alone. "
            "Compare the original and annotated frames, then record any observed error below."
        )

        st.markdown("**Record a review finding**")
        mark_col1, mark_col2 = st.columns(2)
        with mark_col1:
            error_type = st.selectbox(
                "Review finding",
                [
                    "No issue observed",
                    "Missed waste object (false negative)",
                    "False detection (false positive)",
                    "Wrong material classification",
                    "Incorrect recyclability label",
                    "Tracking / ID error",
                    "Unclear / needs closer inspection",
                ],
                key=f"error_type_{current_index}",
            )
        with mark_col2:
            related_detection = st.selectbox(
                "Related detection",
                ["Not applicable"] + [
                    f"Detection {i + 1} | Track ID: {d.get('track_id') or 'Untracked'}"
                    for i, d in enumerate(frame_detections)
                ],
                key=f"related_detection_{current_index}",
            )

        notes = st.text_area(
            "Notes (e.g. expected class, object location, what went wrong)",
            key=f"review_notes_{current_index}",
        )

        if st.button("Save review finding for this frame"):
            mark = {
                "frame_number": current_index + 1,
                "timestamp_seconds": frame_info.get("timestamp_seconds", 0),
                "review_finding": error_type,
                "related_detection": related_detection,
                "notes": notes,
            }
            marks = st.session_state.get("review_marks", [])
            # Replace an existing mark for this frame rather than duplicating it.
            marks = [m for m in marks if m["frame_number"] != current_index + 1]
            marks.append(mark)
            st.session_state["review_marks"] = marks
            st.success("Review finding saved for this frame.")

        marks = st.session_state.get("review_marks", [])
        if marks:
            st.markdown("**Saved review findings**")
            st.dataframe(pd.DataFrame(sorted(marks, key=lambda x: x["frame_number"])), use_container_width=True)
            review_csv = run_dir = Path(st.session_state["review_run_dir"]) / "frame_review_notes.csv"
            with open(review_csv, "w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "frame_number", "timestamp_seconds",
                        "review_finding", "related_detection", "notes",
                    ],
                )
                writer.writeheader()
                writer.writerows(sorted(marks, key=lambda x: x["frame_number"]))
            st.download_button(
                "Download Frame Review Notes (CSV)",
                data=review_csv.read_bytes(),
                file_name="frame_review_notes.csv",
                mime="text/csv",
            )
    else:
        st.warning("No frame review records were generated.")

    st.subheader("Annotated Video")
    if annotated_path.exists():
        video_bytes = annotated_path.read_bytes()
        st.video(video_bytes)
        st.download_button(
            "Download Annotated Video",
            data=video_bytes,
            file_name=annotated_path.name,
            mime="video/mp4",
        )

    st.subheader("Download Reports")
    csv_path = Path(reports["csv_path"])
    json_path = Path(reports["json_path"])
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV Report",
            data=csv_path.read_bytes(),
            file_name=csv_path.name,
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "Download JSON Summary",
            data=json_path.read_bytes(),
            file_name=json_path.name,
            mime="application/json",
        )

st.divider()
st.caption("Powered by YOLOv8s, ByteTrack and EfficientNet-B0 | CPU Processing")
