"""
SmartObjectDetector - Real-Time Computer Vision & Restricted Zone Surveillance Dashboard.
Streamlit application featuring live video annotation, zone monitoring, metric cards, and alert log.
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path
from PIL import Image

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from config.settings import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    TARGET_CLASS_IDS,
    MODEL_NANO_PATH,
    MODEL_SMALL_PATH,
    DATA_DIR,
    SNAPSHOTS_DIR,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_FRAME_HEIGHT,
)
from src.detection.video_source import VideoSource
from src.detection.detector import ObjectDetector
from src.detection.drawing import draw_detection_boxes, draw_fps_overlay, FPSCounter
from src.zone.zone_counter import ZoneCounter
from src.ui.alert_manager import AlertManager
from src.ui.logger import EventLogger

# Streamlit Page Configuration
st.set_page_config(
    page_title="SmartObjectDetector | AI Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark surveillance theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2638 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        color: #9ca3af;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #f9fafb;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-alert {
        color: #ef4444 !important;
    }
    .metric-safe {
        color: #10b981 !important;
    }
    .alert-banner {
        background-color: rgba(239, 68, 68, 0.2);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        font-weight: 600;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state variables
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "total_breaches" not in st.session_state:
    st.session_state.total_breaches = 0
if "detector" not in st.session_state:
    st.session_state.detector = None
if "zone_counter" not in st.session_state:
    st.session_state.zone_counter = None
if "alert_mgr" not in st.session_state:
    st.session_state.alert_mgr = None
if "event_logger" not in st.session_state:
    st.session_state.event_logger = None


def init_components(model_choice):
    """Lazy initialize backend components into session state with dynamic model and zone reloading."""
    model_file = MODEL_NANO_PATH if "Nano" in model_choice else MODEL_SMALL_PATH
    if st.session_state.detector is None or getattr(st.session_state.detector, "model_path", "") != str(model_file):
        st.session_state.detector = ObjectDetector(model_file)
    if st.session_state.zone_counter is None:
        st.session_state.zone_counter = ZoneCounter()
    else:
        st.session_state.zone_counter.load_polygon()
    if st.session_state.alert_mgr is None:
        st.session_state.alert_mgr = AlertManager()
    if st.session_state.event_logger is None:
        st.session_state.event_logger = EventLogger()


# --- SIDEBAR CONTROLS ---
st.sidebar.title("🛡️ Control Panel")
st.sidebar.markdown("---")

# Video Source Selector
st.sidebar.subheader("📹 Video Source")

# Discover test videos in data/test_videos or demo_backup.mp4
test_video_dir = DATA_DIR / "test_videos"
available_files = []
if (DATA_DIR / "demo_backup.mp4").exists():
    available_files.append("data/demo_backup.mp4")
if test_video_dir.exists():
    for f in test_video_dir.glob("*.mp4"):
        available_files.append(str(f.relative_to(BASE_DIR)).replace("\\", "/"))

source_options = ["Webcam (Index 0)"] + available_files + ["Custom Video File / URL"]
source_selection = st.sidebar.selectbox("Select Input Source", source_options)

active_source = 0
if source_selection == "Webcam (Index 0)":
    active_source = 0
elif source_selection == "Custom Video File / URL":
    active_source = st.sidebar.text_input("Enter path or RTSP URL", value="data/demo_backup.mp4")
else:
    active_source = str(BASE_DIR / source_selection)

# Model & Inference Settings
st.sidebar.subheader("⚙️ Detection Settings")
model_type = st.sidebar.radio("YOLO Architecture", ["YOLOv8 Nano (Fastest)", "YOLOv8 Small (Higher Accuracy)"])

conf_slider = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=DEFAULT_CONF_THRESHOLD, step=0.05)
iou_slider = st.sidebar.slider("IoU (NMS) Threshold", min_value=0.1, max_value=1.0, value=DEFAULT_IOU_THRESHOLD, step=0.05)

available_classes = list(TARGET_CLASS_IDS.values())
selected_classes = st.sidebar.multiselect("Monitored Classes", available_classes, default=["person", "backpack", "laptop", "cell phone"])

# Restricted Zone & Trigger Mode
st.sidebar.subheader("📐 Restricted Zone & Trigger")
ref_mode_choice = st.sidebar.radio(
    "Intrusion Trigger Point",
    ["Object Center (Webcam / Sitting at Desk)", "Feet / Base (CCTV / Standing Full-Body)"],
    help="Use 'Object Center' when sitting at a desk so the system doesn't wait for hidden feet to enter."
)
ref_mode = "center" if "Center" in ref_mode_choice else "bottom_center"

debounce_slider = st.sidebar.slider(
    "Sensitivity (Debounce Frames)",
    min_value=1,
    max_value=5,
    value=2,
    help="Consecutive frames inside zone before triggering an alert. 1 or 2 is instant!"
)

st.sidebar.markdown("**Quick Zone Presets:**")
preset_col1, preset_col2 = st.sidebar.columns(2)
if preset_col1.button("🎯 Center Desk", use_container_width=True):
    from src.zone.zone_selector import save_zone
    save_zone([[200, 100], [760, 100], [760, 520], [200, 520]])
    if st.session_state.zone_counter:
        st.session_state.zone_counter.load_polygon()
    st.sidebar.success("Zone set to Center Desk!")

if preset_col2.button("🎯 Left Area", use_container_width=True):
    from src.zone.zone_selector import save_zone
    save_zone([[50, 60], [450, 60], [450, 520], [50, 520]])
    if st.session_state.zone_counter:
        st.session_state.zone_counter.load_polygon()
    st.sidebar.success("Zone set to Left Area!")

if st.sidebar.button("🔄 Reload from zone.json", use_container_width=True):
    if st.session_state.zone_counter:
        st.session_state.zone_counter.load_polygon()
        st.sidebar.success(f"Reloaded zone with {len(st.session_state.zone_counter.polygon)} points!")

st.sidebar.markdown("---")

# Execution Buttons
col_start, col_stop = st.sidebar.columns(2)
if col_start.button("▶ START", use_container_width=True, type="primary"):
    st.session_state.is_running = True
    if st.session_state.zone_counter is not None:
        st.session_state.zone_counter.reset()
        st.session_state.zone_counter.load_polygon()

if col_stop.button("⏹ STOP", use_container_width=True):
    st.session_state.is_running = False

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Alert Log"):
    logger = EventLogger()
    logger.clear_events()
    st.session_state.total_breaches = 0
    st.sidebar.success("Database logs cleared!")


# --- MAIN INTERFACE LAYOUT ---
st.title("🛡️ SmartObjectDetector & Zone Surveillance")
st.caption("Real-Time YOLOv8 Object Classification, ByteTrack Multi-Target Tracking & Debounced Restricted Zone Breach Detection")

# Top KPI Metric Cards
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">System Status</div>
        <div class="metric-value {}">{}</div>
    </div>
    """.format("metric-safe" if st.session_state.is_running else "metric-title",
               "ACTIVE" if st.session_state.is_running else "STANDBY"), unsafe_allow_html=True)

inside_kpi_placeholder = kpi_col2.empty()
unique_kpi_placeholder = kpi_col3.empty()
fps_kpi_placeholder = kpi_col4.empty()

# Layout: Left column for Video Stream, Right column for Analytics & Alerts
col_video, col_analytics = st.columns([1.7, 1.3])

with col_video:
    st.subheader("Live Surveillance Feed")
    video_placeholder = st.empty()
    alert_banner_placeholder = st.empty()

with col_analytics:
    st.subheader("📊 Zone Analytics & Incidents")
    counts_table_placeholder = st.empty()
    st.markdown("### 🚨 Incident History")
    alert_log_placeholder = st.empty()


# --- VIDEO STREAMING LOOP ---
if st.session_state.is_running:
    init_components(model_type)
    detector = st.session_state.detector
    zone_counter = st.session_state.zone_counter
    zone_counter.set_reference_mode(ref_mode)
    zone_counter.debounce_n = debounce_slider
    zone_counter.load_polygon()

    alert_mgr = st.session_state.alert_mgr
    event_logger = st.session_state.event_logger
    fps_counter = FPSCounter()

    # Open video capture
    try:
        vs = VideoSource(active_source)
    except Exception as e:
        st.error(f"Failed to open video source: {e}")
        st.session_state.is_running = False
        st.stop()

    frame_count = 0

    while st.session_state.is_running:
        ret, frame = vs.read()
        if not ret or frame is None:
            # If video file ended, loop back for continuous surveillance demo
            if isinstance(active_source, str) and os.path.exists(active_source):
                vs.restart()
                ret, frame = vs.read()
                if not ret:
                    break
            else:
                break

        now = time.time()
        frame_count += 1
        current_fps = fps_counter.update()

        # 1. Detection
        detections = detector.detect(
            frame,
            conf=conf_slider,
            iou=iou_slider,
            classes=selected_classes if selected_classes else None
        )

        # 2. Tracking + Zone processing
        tracks, events, counts = zone_counter.process(detections, frame_time=now)

        # 3. Process new events
        for ev in events:
            if ev["event"] == "ZONE_ENTRY":
                st.session_state.total_breaches += 1
                # Trigger sound & visual cooldown
                alert_mgr.trigger_alert(ev, current_time=now)
                # Save snapshot & SQLite record
                event_logger.log_event(ev, frame)

        # 4. Visual Overlays
        is_alerting = alert_mgr.is_alert_active(now)
        annotated_frame = frame.copy()

        # Draw restricted zone polygon
        annotated_frame = zone_counter.draw_zone(annotated_frame, is_alert=is_alerting)

        # Draw bounding boxes
        annotated_frame = draw_detection_boxes(annotated_frame, tracks, draw_ids=True, highlight_inside=True)

        # Draw FPS overlay
        annotated_frame = draw_fps_overlay(annotated_frame, current_fps)

        # Draw red border and alert banner if intrusion active
        annotated_frame = alert_mgr.draw_alert_overlay(annotated_frame, current_time=now)

        # Convert BGR to RGB for Streamlit rendering
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        # Render Alert Banner in UI if breach active
        if is_alerting:
            alert_banner_placeholder.markdown(f"""
            <div class="alert-banner">
                🚨 {alert_mgr.current_alert_msg}
            </div>
            """, unsafe_allow_html=True)
        else:
            alert_banner_placeholder.empty()

        # Update Top KPI Cards
        inside_count = counts.get("inside_zone", 0)
        inside_cls = "metric-alert" if inside_count > 0 else "metric-safe"
        inside_kpi_placeholder.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Inside Restricted Zone</div>
            <div class="metric-value {inside_cls}">{inside_count}</div>
        </div>
        """, unsafe_allow_html=True)

        total_unique = sum(counts.get("unique", {}).values())
        unique_kpi_placeholder.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Cumulative Objects Seen</div>
            <div class="metric-value">{total_unique}</div>
        </div>
        """, unsafe_allow_html=True)

        fps_kpi_placeholder.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Real-Time FPS</div>
            <div class="metric-value">{current_fps:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Update Analytics table every 5 frames
        if frame_count % 5 == 0:
            current_dict = counts.get("current", {})
            unique_dict = counts.get("unique", {})
            table_rows = []
            all_classes = set(list(current_dict.keys()) + list(unique_dict.keys()))
            for c in sorted(all_classes):
                table_rows.append({
                    "Class": c.capitalize(),
                    "Visible Now": current_dict.get(c, 0),
                    "Total Unique Seen": unique_dict.get(c, 0)
                })

            if table_rows:
                counts_table_placeholder.dataframe(table_rows, use_container_width=True)
            else:
                counts_table_placeholder.info("No targets detected in current frame.")

            # Update Incident Log Table
            df_events = event_logger.get_events_dataframe(limit=8)
            if not df_events.empty:
                display_df = df_events[["id", "track_id", "class", "event", "timestamp"]]
                alert_log_placeholder.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                alert_log_placeholder.caption("No intrusion events recorded yet.")

        # Brief yield to keep UI responsive
        time.sleep(0.01)

    vs.release()

else:
    # Standby State
    video_placeholder.info("Surveillance feed is on STANDBY. Select your video source in the sidebar and press '▶ START' to begin real-time monitoring.")
    inside_kpi_placeholder.markdown("""
    <div class="metric-card">
        <div class="metric-title">Inside Restricted Zone</div>
        <div class="metric-value">0</div>
    </div>
    """, unsafe_allow_html=True)
    unique_kpi_placeholder.markdown("""
    <div class="metric-card">
        <div class="metric-title">Cumulative Objects Seen</div>
        <div class="metric-value">0</div>
    </div>
    """, unsafe_allow_html=True)
    fps_kpi_placeholder.markdown("""
    <div class="metric-card">
        <div class="metric-title">Real-Time FPS</div>
        <div class="metric-value">0.0</div>
    </div>
    """, unsafe_allow_html=True)

    # Display existing database log if available
    logger = EventLogger()
    df_events = logger.get_events_dataframe(limit=10)
    if not df_events.empty:
        display_df = df_events[["id", "track_id", "class", "event", "timestamp"]]
        alert_log_placeholder.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        alert_log_placeholder.caption("No intrusion events recorded yet.")
