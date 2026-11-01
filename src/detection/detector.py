"""
YOLO Object Detection Module.
Provides object detection using Ultralytics YOLOv8 models (Nano / Small).
Implements the contract interface for downstream tracking and zone detection.
"""

import sys
from pathlib import Path
from ultralytics import YOLO

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    MODEL_NANO_PATH,
    MODEL_SMALL_PATH,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
)


class ObjectDetector:
    """Wrapper around YOLOv8 model for inference and structured output parsing."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = str(MODEL_NANO_PATH)
        self.model_path = str(model_path)
        self.model = YOLO(self.model_path)
        self.names = self.model.names

    def detect(self, frame, conf=DEFAULT_CONF_THRESHOLD, iou=DEFAULT_IOU_THRESHOLD, classes=None):
        """
        Runs object detection on a single frame.

        Args:
            frame (np.ndarray): BGR image frame.
            conf (float): Confidence threshold (0.0 to 1.0).
            iou (float): IoU threshold for NMS.
            classes (list[int] or list[str], optional): Class IDs or names to filter.

        Returns:
            list[dict]: List of detections matching the contract:
                [{"box": (x1, y1, x2, y2), "class": "person", "confidence": 0.89}, ...]
        """
        # Resolve class names to IDs if names were provided
        class_ids = None
        if classes is not None:
            class_ids = []
            for c in classes:
                if isinstance(c, int):
                    class_ids.append(c)
                elif isinstance(c, str):
                    for cid, name in self.names.items():
                        if name.lower() == c.lower():
                            class_ids.append(cid)

        # Run inference
        results = self.model(frame, conf=conf, iou=iou, classes=class_ids, verbose=False)
        result = results[0]

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for b in result.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cls_id = int(b.cls[0])
                confidence = float(b.conf[0])
                class_name = self.names.get(cls_id, str(cls_id))

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "class": class_name,
                    "confidence": confidence,
                    "class_id": cls_id
                })

        return detections


# Module-level default singleton for functional contract
_default_detector = None


def get_default_detector(model_path=None):
    global _default_detector
    if _default_detector is None or (model_path and _default_detector.model_path != str(model_path)):
        _default_detector = ObjectDetector(model_path or MODEL_NANO_PATH)
    return _default_detector


def detect(frame, conf=DEFAULT_CONF_THRESHOLD, iou=DEFAULT_IOU_THRESHOLD, classes=None, model_path=None):
    """
    Direct contract function matching Section 2.2 of the project specification.

    Returns:
        list[dict]: [{"box": (x1, y1, x2, y2), "class": "...", "confidence": 0.xx}]
    """
    detector = get_default_detector(model_path)
    return detector.detect(frame, conf=conf, iou=iou, classes=classes)


if __name__ == "__main__":
    import numpy as np
    import cv2

    print(f"Loading YOLO detector with weights: {MODEL_NANO_PATH}")
    det = ObjectDetector(MODEL_NANO_PATH)
    # Test on a dummy blank frame
    dummy_frame = np.zeros((540, 960, 3), dtype=np.uint8)
    res = det.detect(dummy_frame)
    print(f"Detector initialized successfully. Detections on blank frame: {res}")
