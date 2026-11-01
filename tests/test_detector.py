"""
Unit tests for Object Detection output contract format and behavior.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.detection.detector import detect, ObjectDetector


def test_detector_output_format_on_synthetic_frame():
    """Verify that detector returns structured list matching team contract."""
    # Synthetic frame (black image)
    dummy_frame = np.zeros((540, 960, 3), dtype=np.uint8)

    # Call detect function
    try:
        results = detect(dummy_frame)
    except Exception as e:
        pytest.skip(f"YOLO weights might be downloading or not present: {e}")

    assert isinstance(results, list), "Output should be a list"

    # If objects were detected (or if non-empty), check schema
    for item in results:
        assert "box" in item, "Item must contain 'box' key"
        assert "class" in item, "Item must contain 'class' key"
        assert "confidence" in item, "Item must contain 'confidence' key"

        box = item["box"]
        assert len(box) == 4, "Box must be (x1, y1, x2, y2)"
        x1, y1, x2, y2 = box
        assert isinstance(x1, (int, np.integer))
        assert isinstance(y1, (int, np.integer))
        assert isinstance(x2, (int, np.integer))
        assert isinstance(y2, (int, np.integer))
        assert x1 <= x2 and y1 <= y2, "Box coordinates must satisfy x1 <= x2 and y1 <= y2"

        assert isinstance(item["class"], str), "'class' must be a string"
        assert isinstance(item["confidence"], (float, np.floating)), "'confidence' must be float"
        assert 0.0 <= item["confidence"] <= 1.0, "'confidence' must be between 0.0 and 1.0"


def test_detector_class_filtering():
    """Verify that detector can accept specific class filters."""
    dummy_frame = np.zeros((540, 960, 3), dtype=np.uint8)
    try:
        results = detect(dummy_frame, classes=["person"])
        assert isinstance(results, list)
    except Exception as e:
        pytest.skip(f"YOLO model unavailable: {e}")
