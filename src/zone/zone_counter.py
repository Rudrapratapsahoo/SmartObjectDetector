"""
Zone Intrusion Detection, Debounced State Machine, and Video Pipeline Integration.
Implements point-in-polygon testing, entry/exit state transitions, and zone drawing.
"""

import cv2
import time
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    ZONE_CONFIG_PATH,
    COLOR_ZONE_NORMAL,
    COLOR_ZONE_ALERT,
    ZONE_ALPHA,
    DEBOUNCE_FRAMES,
)
from src.zone.zone_selector import load_zone
from src.zone.tracker import MultiObjectTracker
from src.zone.counters import ObjectMetricsCounter


class ZoneCounter:
    """Manages zone polygon geometry, tracking state machine, debounced events, and counts."""

    def __init__(self, zone_config_path=ZONE_CONFIG_PATH, debounce_n=DEBOUNCE_FRAMES, reference_mode="bottom_center"):
        self.zone_config_path = zone_config_path
        self.debounce_n = debounce_n
        self.reference_mode = reference_mode  # 'bottom_center' (CCTV/feet) or 'center' (webcam/desk)
        self.tracker = MultiObjectTracker()
        self.metrics = ObjectMetricsCounter()

        # Load polygon points
        self.load_polygon()

        # State machine tracking:
        # confirmed_state: track_id -> "OUTSIDE" or "INSIDE"
        # inside_streak: track_id -> int (consecutive frames inside)
        # outside_streak: track_id -> int (consecutive frames outside)
        # last_seen_frame: track_id -> int
        self.confirmed_state = {}
        self.inside_streak = {}
        self.outside_streak = {}
        self.last_seen_frame = {}
        self.frame_index = 0

    def load_polygon(self):
        """Loads and converts zone points into an int32 NumPy array."""
        data = load_zone(self.zone_config_path)
        points = data.get("points", [])
        if points and len(points) >= 3:
            self.polygon = np.array(points, dtype=np.int32)
        else:
            # Fallback polygon
            self.polygon = np.array([[200, 100], [760, 100], [760, 520], [200, 520]], dtype=np.int32)
        self.zone_name = data.get("zone_name", "Restricted Area")

    def set_polygon_points(self, points):
        """Dynamically update polygon points in memory."""
        if len(points) >= 3:
            self.polygon = np.array(points, dtype=np.int32)

    def set_reference_mode(self, mode):
        """Update reference point calculation mode."""
        if mode in ("bottom_center", "center"):
            self.reference_mode = mode

    def is_point_inside(self, x, y):
        """Tests if (x, y) is inside or on the boundary of the polygon."""
        if self.polygon is None or len(self.polygon) < 3:
            return False
        return cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0

    def get_reference_point(self, box):
        """
        Calculates reference point for zone testing:
        - 'bottom_center' (default): (cx, y2), representing where feet touch the ground (CCTV standard).
        - 'center': (cx, (y1 + y2) // 2), geometric center of bounding box (ideal for desk/webcam).
        """
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        if self.reference_mode == "center":
            cy = (y1 + y2) // 2
        else:
            cy = y2
        return (cx, cy)

    def process(self, detections, frame_time=None):
        """
        Main pipeline processing contract function.

        Args:
            detections (list[dict]): Objects detected in the current frame.
            frame_time (float, optional): Timestamp in seconds. Defaults to time.time().

        Returns:
            tuple: (tracks, events, counts)
                tracks: list of tracked objects with 'id', 'box', 'class', 'confidence', 'is_inside'
                events: list of newly triggered event dicts (ZONE_ENTRY, ZONE_EXIT)
                counts: dict with 'current', 'unique', 'inside_zone'
        """
        if frame_time is None:
            frame_time = time.time()
        now_str = datetime.fromtimestamp(frame_time).strftime("%Y-%m-%d %H:%M:%S")

        self.frame_index += 1

        # 1. Update multi-object tracker
        tracks = self.tracker.update(detections)

        active_track_ids = set()
        inside_track_ids = set()
        events = []

        # 2. Check each tracked object against the zone polygon
        for t in tracks:
            tid = t["id"]
            box = t["box"]
            cls_name = t.get("class", "object")
            active_track_ids.add(tid)
            self.last_seen_frame[tid] = self.frame_index

            ref_x, ref_y = self.get_reference_point(box)
            is_currently_inside = self.is_point_inside(ref_x, ref_y)
            t["is_inside"] = is_currently_inside
            t["ref_point"] = (ref_x, ref_y)

            # Update consecutive streaks
            if is_currently_inside:
                self.inside_streak[tid] = self.inside_streak.get(tid, 0) + 1
                self.outside_streak[tid] = 0
            else:
                self.outside_streak[tid] = self.outside_streak.get(tid, 0) + 1
                self.inside_streak[tid] = 0

            prev_state = self.confirmed_state.get(tid, "OUTSIDE")

            # Debounced state transitions
            if prev_state == "OUTSIDE":
                # Fire ZONE_ENTRY when object remains inside for debounce_n consecutive frames
                if self.inside_streak[tid] >= self.debounce_n:
                    self.confirmed_state[tid] = "INSIDE"
                    self.metrics.record_entry(tid, frame_time)
                    events.append({
                        "id": tid,
                        "class": cls_name,
                        "event": "ZONE_ENTRY",
                        "time": now_str,
                        "timestamp": frame_time,
                        "box": box
                    })
            elif prev_state == "INSIDE":
                # Fire ZONE_EXIT when object remains outside for debounce_n consecutive frames
                if self.outside_streak[tid] >= self.debounce_n:
                    self.confirmed_state[tid] = "OUTSIDE"
                    dwell_sec = self.metrics.record_exit(tid, frame_time)
                    events.append({
                        "id": tid,
                        "class": cls_name,
                        "event": "ZONE_EXIT",
                        "time": now_str,
                        "timestamp": frame_time,
                        "box": box,
                        "dwell_time": round(dwell_sec, 1)
                    })

            # Check if this track is officially inside (confirmed or raw)
            if self.confirmed_state.get(tid) == "INSIDE" or is_currently_inside:
                inside_track_ids.add(tid)

        # 3. Clean up stale tracks that disappeared (e.g. absent for > 60 frames)
        stale_threshold = 60
        for tid in list(self.confirmed_state.keys()):
            if tid not in active_track_ids:
                frames_absent = self.frame_index - self.last_seen_frame.get(tid, self.frame_index)
                if frames_absent > stale_threshold:
                    if self.confirmed_state[tid] == "INSIDE":
                        # Record departure on disappear
                        dwell_sec = self.metrics.record_exit(tid, frame_time)
                    self.confirmed_state.pop(tid, None)
                    self.inside_streak.pop(tid, None)
                    self.outside_streak.pop(tid, None)
                    self.last_seen_frame.pop(tid, None)

        # 4. Compute metrics
        counts = self.metrics.compute_counts(tracks, inside_track_ids)

        return tracks, events, counts

    def draw_zone(self, frame, is_alert=False):
        """
        Renders the semi-transparent polygon onto the video frame.
        Draws green when calm and glowing red when an intrusion is detected.
        """
        if self.polygon is None or len(self.polygon) < 3:
            return frame

        overlay = frame.copy()
        color = COLOR_ZONE_ALERT if is_alert else COLOR_ZONE_NORMAL
        border_color = (0, 0, 255) if is_alert else (0, 220, 100)

        # Fill semi-transparent polygon
        cv2.fillPoly(overlay, [self.polygon], color)
        cv2.addWeighted(overlay, ZONE_ALPHA, frame, 1.0 - ZONE_ALPHA, 0, frame)

        # Draw crisp outline
        cv2.polylines(frame, [self.polygon], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

        # Render zone label at top-left vertex
        pts = self.polygon
        min_y_idx = np.argmin(pts[:, 1])
        top_vertex = tuple(pts[min_y_idx])
        label_text = f"RESTRICTED ZONE [{'ALERT' if is_alert else 'SECURE'}]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, 1)
        bx, by = max(5, top_vertex[0] - 10), max(th + 5, top_vertex[1] - 8)

        cv2.rectangle(frame, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (bx, by), font, font_scale, border_color, 1, cv2.LINE_AA)

        return frame

    def reset(self):
        """Reset state machine and internal tracker."""
        self.tracker.reset()
        self.metrics.reset()
        self.confirmed_state.clear()
        self.inside_streak.clear()
        self.outside_streak.clear()
        self.last_seen_frame.clear()
        self.frame_index = 0
