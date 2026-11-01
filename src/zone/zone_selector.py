"""
Interactive mouse tool for defining polygonal restricted zones.
Saves polygon coordinates and reference frame dimensions to config/zone.json.
"""

import cv2
import json
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    ZONE_CONFIG_PATH,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_SOURCE,
)
from src.detection.video_source import open_video_source


def load_zone(config_path=ZONE_CONFIG_PATH):
    """
    Load polygon points and dimensions from JSON config.
    Returns:
        dict: {"points": [[x, y], ...], "frame_width": int, "frame_height": int, ...}
    """
    path = Path(config_path)
    if not path.exists():
        # Fallback default box
        return {
            "points": [[400, 150], [850, 150], [850, 480], [400, 480]],
            "frame_width": DEFAULT_FRAME_WIDTH,
            "frame_height": DEFAULT_FRAME_HEIGHT,
            "zone_name": "Default Restricted Area"
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"[Zone Error] Failed to read {config_path}: {e}")
        return {
            "points": [[400, 150], [850, 150], [850, 480], [400, 480]],
            "frame_width": DEFAULT_FRAME_WIDTH,
            "frame_height": DEFAULT_FRAME_HEIGHT,
            "zone_name": "Fallback Restricted Area"
        }


def save_zone(points, width=DEFAULT_FRAME_WIDTH, height=DEFAULT_FRAME_HEIGHT, config_path=ZONE_CONFIG_PATH):
    """
    Save polygon points and reference dimensions to JSON config.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "points": points,
        "frame_width": width,
        "frame_height": height,
        "zone_name": "Restricted Zone 1"
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[ZoneSelector] Successfully saved {len(points)} zone points to {path.resolve()}")


def select_zone_interactive(source=DEFAULT_SOURCE, config_path=ZONE_CONFIG_PATH):
    """
    Opens first frame of video source and lets user click polygon vertices.
    Controls:
        - Left Click: Add polygon vertex
        - 'u': Undo last point
        - 'c': Clear all points
        - Enter / Space: Save zone to zone.json and exit
        - 'q' / ESC: Quit without saving
    """
    vs = open_video_source(source)
    ret, frame = vs.read()
    vs.release()

    if not ret or frame is None:
        print(f"[ZoneSelector] Could not grab frame from source: {source}")
        return False

    h, w = frame.shape[:2]
    existing = load_zone(config_path)
    # Scale existing points if loaded from different resolution
    points = existing.get("points", [])
    window_name = "Restricted Zone Selector (Left Click=Add | u=Undo | c=Clear | Enter=Save | q=Quit)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([int(x), int(y)])

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\n--- Zone Selector Instructions ---")
    print(" Left Click : Add vertex point")
    print(" 'u' Key    : Undo last vertex")
    print(" 'c' Key    : Clear all vertices")
    print(" Enter / Space : Save polygon to config/zone.json")
    print(" 'q' / ESC  : Cancel and exit without saving\n")

    saved = False
    while True:
        display = frame.copy()

        # Draw existing lines and points
        if len(points) > 0:
            pts = np.array(points, dtype=np.int32)
            # Draw translucent fill if at least 3 points
            if len(points) >= 3:
                overlay = display.copy()
                cv2.fillPoly(overlay, [pts], (0, 165, 255))
                cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)
                cv2.polylines(display, [pts], isClosed=True, color=(0, 200, 255), thickness=2)
            else:
                cv2.polylines(display, [pts], isClosed=False, color=(0, 200, 255), thickness=2)

            for idx, pt in enumerate(points):
                cv2.circle(display, tuple(pt), 5, (0, 0, 255), -1)
                cv2.putText(display, str(idx + 1), (pt[0] + 7, pt[1] - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Header instructions on the frame
        cv2.rectangle(display, (10, 10), (w - 10, 45), (30, 30, 30), -1)
        status_text = f"Points: {len(points)} | Left Click=Add | 'u'=Undo | 'c'=Clear | Enter=Save | 'q'=Quit"
        cv2.putText(display, status_text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # Enter or Space
            if len(points) >= 3:
                save_zone(points, width=w, height=h, config_path=config_path)
                saved = True
                break
            else:
                print("[ZoneSelector] Need at least 3 points to form a polygon.")
        elif key == ord('u'):
            if len(points) > 0:
                removed = points.pop()
                print(f"[ZoneSelector] Removed point {removed}")
        elif key == ord('c'):
            points.clear()
            print("[ZoneSelector] Cleared all points.")
        elif key in (ord('q'), 27):  # 'q' or ESC
            print("[ZoneSelector] Exited without saving.")
            break

    cv2.destroyAllWindows()
    return saved


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Draw restricted zone polygon")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Webcam index, video file path, or stream URL")
    args = parser.parse_args()

    select_zone_interactive(args.source)
