
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "trained_models"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"

# Trained model paths
DETECTION_MODEL = MODEL_DIR / "YOLO26s_pLitterStreet" / "best.pt"
CLASSIFICATION_MODEL = MODEL_DIR / "efficientnet_b0_best.pth"

# Supporting configuration files
CLASS_NAMES_FILE = MODEL_DIR / "class_names.json"
RECYCLABILITY_FILE = MODEL_DIR / "recyclability_mapping.json"

# Create output directories if missing
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Processing settings
DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640

# Video settings
SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv"]
