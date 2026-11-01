"""
Global configuration and settings for SmartObjectDetector.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
SNAPSHOTS_DIR = OUTPUTS_DIR / "snapshots"
LOGS_DIR = OUTPUTS_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
TEST_VIDEOS_DIR = DATA_DIR / "test_videos"

# Database & Zone Config
DB_PATH = LOGS_DIR / "events.db"
ZONE_CONFIG_PATH = CONFIG_DIR / "zone.json"

# Models
MODEL_NANO_PATH = MODELS_DIR / "yolov8n.pt"
MODEL_SMALL_PATH = MODELS_DIR / "yolov8s.pt"

# Video / Camera Settings
DEFAULT_FRAME_WIDTH = 960
DEFAULT_FRAME_HEIGHT = 540
DEFAULT_SOURCE = 0  # 0 for webcam, or string path/URL

# Detection Thresholds
DEFAULT_CONF_THRESHOLD = 0.50
DEFAULT_IOU_THRESHOLD = 0.45

# COCO Class IDs commonly monitored for security / surveillance
# (person=0, backpack=24, handbag=26, bottle=39, laptop=63, cell phone=67)
TARGET_CLASS_IDS = {
    0: "person",
    24: "backpack",
    26: "handbag",
    39: "bottle",
    63: "laptop",
    67: "cell phone"
}

# Zone & Alert Settings
DEBOUNCE_FRAMES = 3             # Consecutive frames inside polygon before firing ZONE_ENTRY
ALERT_COOLDOWN_SECONDS = 5.0    # Minimum seconds between alerts for the same track ID
ALERT_BANNER_DURATION_SECONDS = 2.0  # Duration the visual red warning banner stays active
DWELL_TIME_LOGGING = True       # Whether to track and log dwell times in seconds

# Visualization Colors (BGR format for OpenCV)
COLOR_ZONE_NORMAL = (0, 255, 128)      # Soft green/cyan for normal zone
COLOR_ZONE_ALERT = (0, 0, 255)         # Bright red for intrusion alert
ZONE_ALPHA = 0.25                      # Transparency for zone polygon

# Class Color Palette (BGR)
CLASS_COLORS = {
    "person": (255, 100, 0),       # Blueish
    "backpack": (0, 165, 255),     # Orange
    "handbag": (180, 105, 255),    # Pinkish
    "bottle": (255, 255, 0),       # Cyan
    "laptop": (0, 255, 255),       # Yellow
    "cell phone": (0, 200, 0),     # Green
    "default": (200, 200, 200)     # Gray
}

def get_class_color(cls_name: str):
    """Return BGR color tuple for a given class name."""
    return CLASS_COLORS.get(cls_name.lower(), CLASS_COLORS["default"])
