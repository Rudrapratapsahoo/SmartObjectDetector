"""
SmartObjectDetector - Main Pipeline Execution Script.
Orchestrates: Video Capture -> YOLOv8 Detection -> Multi-Target Tracking (ByteTrack)
               -> Polygonal Zone Analysis -> Audio/Visual Alerts -> SQLite Logging with Evidence.
"""

import cv2
import argparse
import time
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from config.settings import (
    DEFAULT_SOURCE,
    MODEL_NANO_PATH,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    ZONE_CONFIG_PATH,
    TARGET_CLASS_IDS,
)
from src.detection.video_source import VideoSource
from src.detection.detector import ObjectDetector
from src.detection.drawing import (
    draw_detection_boxes,
    draw_fps_overlay,
    draw_counts_hud,
    FPSCounter,
)
from src.zone.zone_counter import ZoneCounter
from src.ui.alert_manager import AlertManager
from src.ui.logger import EventLogger


def run_pipeline(
    source=DEFAULT_SOURCE,
    model_path=MODEL_NANO_PATH,
    conf_thresh=DEFAULT_CONF_THRESHOLD,
    iou_thresh=DEFAULT_IOU_THRESHOLD,
    zone_config=ZONE_CONFIG_PATH,
    save_output_path=None,
    filter_classes=None,
):
    """Executes the full real-time surveillance pipeline in an OpenCV window."""

    print("=" * 65)
    print("      🛡️  SmartObjectDetector - Real-Time Surveillance  🛡️")
    print("=" * 65)
    print(f" Source        : {source}")
    print(f" Model Weights : {model_path}")
    print(f" Confidence    : {conf_thresh}")
    print(f" Zone Config   : {zone_config}")
    print(" Press 'q' or ESC in the video window to quit gracefully.")
    print("=" * 65)

    # 1. Initialize Video Source
    try:
        vs = VideoSource(source)
    except Exception as e:
        print(f"[Fatal Error] Could not start video source: {e}")
        return

    # 2. Initialize Processing Modules
    print("[System] Loading YOLO model...")
    detector = ObjectDetector(model_path)
    print("[System] Initializing Zone Counter & Tracker...")
    zone_counter = ZoneCounter(zone_config_path=zone_config)
    alert_mgr = AlertManager()
    logger = EventLogger()
    fps_counter = FPSCounter()

    # Optional Video Writer
    video_writer = None
    if save_output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(save_output_path, fourcc, vs.get_fps(), (vs.width, vs.height))
        print(f"[System] Recording output stream to: {save_output_path}")

    window_name = "SmartObjectDetector (Press 'q' to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, vs.width, vs.height)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = vs.read()
            if not ret or frame is None:
                # If reading from a video file, loop back for demonstration convenience
                if isinstance(source, str) and Path(source).exists():
                    vs.restart()
                    ret, frame = vs.read()
                    if not ret:
                        print("\n[System] Reached end of video file.")
                        break
                else:
                    print("\n[System] Video stream disconnected.")
                    break

            now = time.time()
            frame_count += 1
            fps = fps_counter.update()

            # --- STEP 1: Object Detection ---
            detections = detector.detect(
                frame,
                conf=conf_thresh,
                iou=iou_thresh,
                classes=filter_classes
            )

            # --- STEP 2: Tracking & Zone Breach Analysis ---
            tracks, events, counts = zone_counter.process(detections, frame_time=now)

            # --- STEP 3: Handle Triggered Events ---
            for ev in events:
                event_type = ev.get("event")
                track_id = ev.get("id")
                cls_name = ev.get("class", "object")

                if event_type == "ZONE_ENTRY":
                    print(f"🚨 [BREACH DETECTED] {cls_name} (ID #{track_id}) entered restricted zone at {ev.get('time')}!")
                    # Asynchronous sound beep & visual alert activation
                    alert_mgr.trigger_alert(ev, current_time=now)
                    # Persist event to SQLite and save evidence snapshot
                    logger.log_event(ev, frame)

                elif event_type == "ZONE_EXIT":
                    dwell = ev.get("dwell_time", 0.0)
                    print(f"ℹ️  [ZONE EXIT] {cls_name} (ID #{track_id}) exited zone. Dwell time: {dwell}s")
                    logger.log_event(ev, None)

            # --- STEP 4: Render Overlays ---
            display_frame = frame.copy()
            is_alert = alert_mgr.is_alert_active(now)

            # 4a. Draw Zone Polygon (green when secure, red when breached)
            display_frame = zone_counter.draw_zone(display_frame, is_alert=is_alert)

            # 4b. Draw Bounding Boxes with Labels & Tracking IDs
            display_frame = draw_detection_boxes(display_frame, tracks, draw_ids=True, highlight_inside=True)

            # 4c. Render FPS Overlay
            display_frame = draw_fps_overlay(display_frame, fps)

            # 4d. Render Real-Time Counts HUD
            display_frame = draw_counts_hud(display_frame, counts)

            # 4e. Render Red Alert Flashing Border & Banner if intrusion active
            display_frame = alert_mgr.draw_alert_overlay(display_frame, current_time=now)

            # --- STEP 5: Output & Display ---
            if video_writer is not None:
                video_writer.write(display_frame)

            cv2.imshow(window_name, display_frame)

            # Key handling (q or ESC to exit)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("\n[System] Exit requested by user.")
                break

    finally:
        total_time = max(0.001, time.time() - start_time)
        avg_fps = frame_count / total_time
        print(f"\n[System] Pipeline finished. Processed {frame_count} frames in {total_time:.2f}s (Average FPS: {avg_fps:.1f})")

        vs.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="SmartObjectDetector Surveillance Pipeline")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Webcam index (0), video path, or RTSP URL")
    parser.add_argument("--model", default=str(MODEL_NANO_PATH), help="Path to YOLO weights (.pt)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD, help="IoU NMS threshold (0.0 to 1.0)")
    parser.add_argument("--zone", default=str(ZONE_CONFIG_PATH), help="Path to zone polygon JSON config")
    parser.add_argument("--save", default=None, help="Optional output MP4 video path")
    parser.add_argument("--classes", nargs="*", default=None, help="Filter specific classes (e.g. person backpack)")

    args = parser.parse_args()

    # Convert numeric string to int for webcam index
    source = int(args.source) if args.source.isdigit() else args.source

    run_pipeline(
        source=source,
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        zone_config=args.zone,
        save_output_path=args.save,
        filter_classes=args.classes,
    )


if __name__ == "__main__":
    main()
