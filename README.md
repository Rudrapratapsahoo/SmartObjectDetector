# 🛡️ SmartObjectDetector

> **Real-Time Computer Vision System for Multi-Object Detection, Multi-Target Tracking, Counting, and Polygonal Restricted Zone Breach Surveillance.**

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n%20%7C%20v8s-brightgreen.svg)](https://ultralytics.com/)
[![Tracking](https://img.shields.io/badge/Tracker-ByteTrack-orange.svg)](https://github.com/roboflow/supervision)
[![UI](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📌 Project Overview

**SmartObjectDetector** transforms commodity CCTV cameras or webcams into intelligent security sensors. It monitors video feeds in real time, classifies multiple objects, assigns persistent tracking IDs, maintains live and cumulative counts, and alerts operators immediately when targets enter a user-defined polygonal restricted zone.

### Core Capabilities:
- 👁️ **Multi-Object Detection**: Leverages pre-trained **YOLOv8** (COCO dataset) for detecting humans, backpacks, phones, laptops, and more.
- 🎯 **Persistent Multi-Target Tracking**: Integrates **ByteTrack** to maintain consistent IDs across frames even during brief occlusions.
- 📐 **Custom Polygonal Zones**: Interactive mouse selection tool enabling operators to define arbitrary polygonal security zones for any camera perspective.
- 🧠 **Debounced Breach Detection**: Point-in-polygon testing using the target's physical contact point (feet level) combined with an $N$-frame debounced state machine to eliminate jitter false alarms.
- 🚨 **Multi-Modal Alerting**: Asynchronous non-blocking audio beeps, glowing perimeter screen flash, and visual HUD alerts.
- 📸 **Incident Evidence Logging**: Automatically logs breach incidents into a persistent **SQLite** database accompanied by photographic evidence snapshots.
- 📊 **Security Supervisor Dashboard**: Modern **Streamlit** browser dashboard featuring live annotated video, dynamic threshold sliders, real-time KPI metrics, and searchable incident history.

---

## 👥 Team Work Distribution

| Member | Focus Area | Key Modules |
| :--- | :--- | :--- |
| **Member 1 (Sanskriti)** | **Object Detection (The Eyes)** | `src/detection/video_source.py`, `src/detection/detector.py`, `src/detection/drawing.py`, `models/` |
| **Member 2 (Rudra)** | **Zone Logic & Tracking (The Brain)** | `src/zone/zone_selector.py`, `src/zone/tracker.py`, `src/zone/zone_counter.py`, `src/zone/counters.py` |
| **Member 3 (Upasana)** | **Alerts, UI & Evaluation (The Face)** | `src/ui/alert_manager.py`, `src/ui/logger.py`, `src/ui/dashboard.py`, `data/`, `evaluation/`, `docs/` |

---

## 📁 Repository Structure

```
SmartObjectDetector/
├── src/
│   ├── detection/
│   │   ├── video_source.py        # Video capture (Webcam / File / IP camera RTSP)
│   │   ├── detector.py            # YOLOv8 load & detect() contract function
│   │   └── drawing.py             # Bounding boxes, labels, FPS overlay, HUD
│   ├── zone/
│   │   ├── zone_selector.py       # Interactive mouse tool to draw polygon
│   │   ├── tracker.py             # ByteTrack multi-target tracking wrapper
│   │   ├── zone_counter.py        # Point-in-polygon test & debounced state machine
│   │   └── counters.py            # Current, unique, inside-zone metrics & dwell time
│   └── ui/
│       ├── alert_manager.py       # Asynchronous sound, red border flash, cooldown
│       ├── logger.py              # SQLite event persistence & snapshot saving
│       └── dashboard.py           # Interactive Streamlit surveillance dashboard
├── config/
│   ├── settings.py                # Global thresholds, paths, class IDs, and colors
│   └── zone.json                  # Saved polygon coordinates & reference resolution
├── models/
│   ├── yolov8n.pt                 # YOLOv8 Nano weights (fastest)
│   └── yolov8s.pt                 # YOLOv8 Small weights (high precision)
├── data/
│   ├── test_videos/               # Benchmark surveillance clips
│   ├── ground_truth.csv           # Ground truth counts and breach occurrences
│   └── demo_backup.mp4            # Recorded backup demonstration video
├── outputs/
│   ├── snapshots/                 # Photographic evidence captured on breach
│   └── logs/
│       └── events.db              # SQLite incident records database
├── evaluation/
│   ├── model_comparision.md       # YOLOv8n vs YOLOv8s benchmark report
│   ├── results_table.xlsx         # Per-clip precision, recall, and FPS results
│   └── failure_cases/             # Documentation of occlusion / low light edge cases
├── docs/
│   ├── report.docx                # Formal project documentation report
│   ├── demo_script.md             # Evaluator demonstration walkthrough script
│   └── Project_Development_Guide.docx
├── tests/
│   ├── test_detector.py           # Unit tests for detector output contract
│   ├── test_zone_logic.py         # Unit tests for polygon geometry & debounce
│   └── test_counters.py           # Unit tests for unique counts & dwell time
├── main.py                        # Standalone OpenCV surveillance execution script
├── requirements.txt               # Python package dependencies
└── README.md                      # Installation and usage guide
```

---

## ⚡ Quick Start & Installation

### Step 1: Clone and Enter Repository
```bash
git clone https://github.com/Rudrapratapsahoo/SmartObjectDetector.git
cd SmartObjectDetector
```

### Step 2: Create & Activate Virtual Environment
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import ultralytics, cv2, numpy, supervision, shapely, streamlit, pygame, pandas, pytest; print('All OK')"
```

---

## 🚀 How to Run

### 1. Draw the Restricted Zone (One-time Setup)
Define the boundaries of the security zone on the first camera frame:
```bash
python src/zone/zone_selector.py
```
- **Left-Click**: Add polygon vertex.
- **`u`**: Undo last point.
- **`c`**: Clear all vertices.
- **Enter / Space**: Save polygon coordinates to `config/zone.json`.
- **`q` / ESC**: Quit without saving.

---

### 2. Run the Full Surveillance Pipeline (OpenCV Window)
Execute the complete detection, tracking, zone analysis, and alert system:
```bash
# Run with default laptop webcam:
python main.py

# Run with a recorded video file:
python main.py --source data/demo_backup.mp4

# Run with custom confidence threshold and model:
python main.py --model models/yolov8s.pt --conf 0.60
```
- Press **`q`** or **ESC** in the OpenCV video window to stop.

---

### 3. Run the Web Surveillance Dashboard (Browser UI)
Launch the operator dashboard:
```bash
streamlit run src/ui/dashboard.py
```
Opens in your browser at `http://localhost:8501`.
- View live annotated surveillance stream.
- Monitor real-time KPI metrics (Inside Zone, Cumulative Count, FPS).
- Adjust confidence and IoU sliders dynamically on the fly.
- Inspect the live incident table and photographic evidence.

---

### 4. Run Automated Unit Tests
Verify all geometric, tracking, and counter logic:
```bash
pytest tests/ -v
```

---

## 🛠️ Configuration Options (`config/settings.py`)

| Setting | Default Value | Description |
| :--- | :--- | :--- |
| `DEFAULT_FRAME_WIDTH` | `960` | Normalized frame width for consistent geometry |
| `DEFAULT_FRAME_HEIGHT` | `540` | Normalized frame height |
| `DEFAULT_CONF_THRESHOLD` | `0.50` | Minimum detection confidence score |
| `DEFAULT_IOU_THRESHOLD` | `0.45` | Non-Maximum Suppression (NMS) overlap threshold |
| `DEBOUNCE_FRAMES` | `3` | Consecutive frames inside zone before triggering alert |
| `ALERT_COOLDOWN_SECONDS` | `5.0` | Minimum seconds between audio alerts for the same ID |
| `ALERT_BANNER_DURATION` | `2.0` | Duration the visual red warning banner remains active |

---

## 🔍 Database Inspection
Security breach logs are stored in `outputs/logs/events.db`. You can query them with Python:
```bash
python -c "import sqlite3; print(sqlite3.connect('outputs/logs/events.db').execute('SELECT * FROM events').fetchall())"
```

---

## 📜 License
Developed for academic surveillance and computer vision evaluation under the MIT License.
