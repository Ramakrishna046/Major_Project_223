
import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path

from core.pipeline import WasteProcessingPipeline
from core.reporting import generate_reports


st.set_page_config(
    page_title="AI Waste Management System",
    page_icon="♻️",
    layout="wide",
)

st.title("♻️ AI-Driven Automated Waste Management System")
st.write(
    "Upload a video to detect, track, classify and count waste objects."
)

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
            with tempfile.TemporaryDirectory() as temp_dir:

                video_path = Path(temp_dir) / uploaded_video.name

                video_path.write_bytes(uploaded_video.getvalue())

                status.info("Loading trained AI models...")

                pipeline = WasteProcessingPipeline()

                status.info("Processing video...")

                results = pipeline.process_video(
                    video_path,
                    progress_callback=progress_bar.progress,
                )

                status.info("Generating reports...")

                reports = generate_reports(results)

                st.session_state["results"] = results
                st.session_state["reports"] = reports

            progress_bar.progress(1.0)
            status.success("Video processing completed!")

        except Exception as error:
            status.error(f"Processing failed: {error}")


if "results" in st.session_state:

    results = st.session_state["results"]
    reports = st.session_state["reports"]
    summary = reports["summary"]

    st.divider()
    st.header("Detection Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Unique Tracked Objects",
        summary["total_unique_objects"],
    )

    col2.metric(
        "Recyclable Objects",
        summary["recyclable_objects"],
    )

    col3.metric(
        "Non-recyclable Objects",
        summary["non_recyclable_objects"],
    )

    st.subheader("Waste Category Distribution")

    category_counts = summary["category_counts"]

    if category_counts:
        category_df = pd.DataFrame(
            list(category_counts.items()),
            columns=["Category", "Count"],
        )

        st.bar_chart(
            category_df.set_index("Category")
        )

        st.dataframe(category_df, use_container_width=True)

    else:
        st.info("No classified waste objects were found.")

    st.subheader("Object Details")

    if results["objects"]:
        object_df = pd.DataFrame(results["objects"])
        st.dataframe(object_df, use_container_width=True)

    st.subheader("Annotated Video")

    output_video = Path(results["video_path"])

    if output_video.exists():
        video_bytes = output_video.read_bytes()

        st.video(video_bytes)

        st.download_button(
            "Download Annotated Video",
            data=video_bytes,
            file_name=output_video.name,
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
            file_name="waste_detection_report.csv",
            mime="text/csv",
        )

    with col2:
        st.download_button(
            "Download JSON Summary",
            data=json_path.read_bytes(),
            file_name="waste_detection_summary.json",
            mime="application/json",
        )

st.divider()

st.caption(
    "Powered by YOLOv8s, ByteTrack and EfficientNet-B0 | CPU Processing"
)
