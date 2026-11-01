"""
Visual overlay and drawing utilities for object detection and tracking.
Renders bounding boxes, class labels, confidence scores, track IDs, and FPS counters.
"""

import cv2
import time
from collections import deque
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import get_class_color


class FPSCounter:
    """Calculates instantaneous and smoothed moving average FPS."""

    def __init__(self, moving_window=30):
        self.moving_window = moving_window
        self.frame_times = deque(maxlen=moving_window)
        self.start_time = time.time()
        self.total_frames = 0

    def update(self):
        """Record frame arrival time and return current smoothed FPS."""
        now = time.time()
        self.frame_times.append(now)
        self.total_frames += 1

        if len(self.frame_times) < 2:
            return 0.0

        elapsed = self.frame_times[-1] - self.frame_times[0]
        if elapsed > 0:
            return (len(self.frame_times) - 1) / elapsed
        return 0.0

    def get_average_fps(self):
        """Calculate overall average FPS since initialization."""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.total_frames / elapsed
        return 0.0


def draw_detection_boxes(frame, detections, draw_ids=True, highlight_inside=False):
    """
    Draws bounding boxes and labels for detections or tracked objects.

    Args:
        frame (np.ndarray): Image to draw on (modified in-place or copied).
        detections (list[dict]): List of objects, each containing:
            - 'box': (x1, y1, x2, y2)
            - 'class': class name
            - 'confidence': confidence score (0 to 1)
            - 'id' (optional): tracker ID
            - 'is_inside' (optional): boolean indicating if inside zone
        draw_ids (bool): Whether to include tracker ID in the label.
        highlight_inside (bool): Whether to colorize boxes in red if inside restricted zone.

    Returns:
        np.ndarray: Annotated frame.
    """
    for det in detections:
        box = det.get("box")
        if not box:
            continue
        x1, y1, x2, y2 = box
        cls_name = det.get("class", "object")
        conf = det.get("confidence", 0.0)
        track_id = det.get("id", None)
        is_inside = det.get("is_inside", False)

        # Color: bright red if inside restricted zone and highlighted, else per-class color
        if highlight_inside and is_inside:
            color = (0, 0, 255)  # Alert Red
        else:
            color = get_class_color(cls_name)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Prepare label string (e.g. "#4 person 87%" or "person 87%")
        conf_pct = int(round(conf * 100)) if conf <= 1.0 else int(conf)
        if draw_ids and track_id is not None:
            label = f"#{track_id} {cls_name} {conf_pct}%"
        else:
            label = f"{cls_name} {conf_pct}%"

        # Text size and background badge
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Badge position (above box, or just inside if near top border)
        badge_y1 = max(0, y1 - text_h - baseline - 4)
        badge_y2 = y1
        badge_x1 = x1
        badge_x2 = min(frame.shape[1], x1 + text_w + 6)

        # Draw filled background badge
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)

        # Text color (white or black depending on badge brightness)
        text_color = (255, 255, 255)
        text_origin = (badge_x1 + 3, badge_y2 - baseline - 1)
        cv2.putText(frame, label, text_origin, font, font_scale, text_color, thickness, cv2.LINE_AA)

        # Draw reference point (feet position or center point)
        if "ref_point" in det and det["ref_point"] is not None:
            ref_x, ref_y = det["ref_point"]
        else:
            ref_x = (x1 + x2) // 2
            ref_y = y2
        cv2.circle(frame, (int(ref_x), int(ref_y)), 5, (0, 255, 255), -1)

    return frame


def draw_fps_overlay(frame, fps, position=(20, 35)):
    """Renders sleek FPS badge on frame."""
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 2), (0, 0, 0), -1)
    cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 2), (0, 255, 0), 1)
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return frame


def draw_counts_hud(frame, counts):
    """
    Renders top-right HUD displaying active counts.
    counts: dict like {
        "current": {"person": 2, ...},
        "unique": {"person": 5, ...},
        "inside_zone": 1
    }
    """
    if not counts:
        return frame

    current = counts.get("current", {})
    inside_count = counts.get("inside_zone", 0)

    # Build lines
    lines = [f"Inside Zone: {inside_count}"]
    for cls_name, cnt in current.items():
        lines.append(f"{cls_name.capitalize()}: {cnt}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20

    max_w = 0
    for l in lines:
        (tw, _), _ = cv2.getTextSize(l, font, font_scale, thickness)
        max_w = max(max_w, tw)

    hud_w = max_w + 20
    hud_h = len(lines) * line_height + 15
    top_x = frame.shape[1] - hud_w - 15
    top_y = 15

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (top_x, top_y), (top_x + hud_w, top_y + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (top_x, top_y), (top_x + hud_w, top_y + hud_h), (80, 80, 80), 1)

    for i, line in enumerate(lines):
        y_pos = top_y + 18 + i * line_height
        color = (0, 0, 255) if i == 0 and inside_count > 0 else (220, 220, 220)
        cv2.putText(frame, line, (top_x + 10, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)

    return frame
