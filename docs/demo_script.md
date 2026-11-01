# SmartObjectDetector - Live Demonstration & Evaluation Script

This document provides a comprehensive step-by-step presentation script for demonstrating the SmartObjectDetector system to evaluators, faculty, or stakeholders.

---

## Pre-Demo Checklist
- [ ] Laptop connected to power and display/projector.
- [ ] Webcam connected or demo video file verified in `data/demo_backup.mp4`.
- [ ] Audio volume unmuted to demonstrate audible intrusion beeps.
- [ ] Virtual environment activated:
  ```powershell
  cd "d:\My Projects\SmartObjectDetector"
  .\venv\Scripts\activate
  ```

---

## Demonstration Workflow (Total Duration: 8–10 Minutes)

### Phase 1: Problem Introduction (1 Minute)
> **Presenter:**
> *"Good morning/afternoon, evaluators. Today we are presenting **SmartObjectDetector**, an intelligent real-time surveillance and perimeter security system. Conventional CCTV systems require tedious manual monitoring. Our system automatically detects multiple object classes in real time, maintains persistent tracking across frames, counts objects, detects restricted zone breaches using polygon geometry, and triggers automated audio-visual alarms while logging incident evidence to a database."*

---

### Phase 2: Drawing the Restricted Zone (2 Minutes)
> **Presenter:**
> *"First, we demonstrate how security supervisors can define custom polygonal restricted zones for any camera perspective."*

1. **Execute Command:**
   ```powershell
   python src/zone/zone_selector.py
   ```
2. **Action in Window:**
   - Left-click on 4 or more vertices on the first frame to encircle a doorway, desk, or corridor.
   - Press `'u'` to demonstrate undoing a mistake.
   - Press **Enter** to save the polygon.
   - Explain: *"Notice how the coordinates and reference resolution are saved directly to `config/zone.json`."*

---

### Phase 3: Live Pipeline Execution in OpenCV (3 Minutes)
> **Presenter:**
> *"Now let's launch the core computer vision pipeline. This runs YOLOv8 detection, ByteTrack tracking, point-in-polygon intrusion checks, and our debounced state machine."*

1. **Execute Command:**
   ```powershell
   python main.py
   ```
   *(Or with demo video: `python main.py --source data/demo_backup.mp4`)*
2. **Observe & Explain Features:**
   - **Bounding Boxes & Badges**: Distinct colors per class with confidence scores.
   - **Persistent IDs**: Notice each person retains their unique ID across frames.
   - **Reference Point (Feet Position)**: The cyan dot at the bottom-center of the box represents physical ground location.
   - **Live FPS Overlay**: Point out the smooth 25+ FPS displayed in the top-left corner.
   - **HUD Metrics**: Top-right corner shows real-time visible counts and cumulative counts.
   - **Intrusion Simulation**:
     - Walk into the polygon (or observe target in video).
     - Notice debounce: it waits 3 consecutive frames before triggering to prevent false alarms at boundaries.
     - **Alarm Sound**: Audible alert beep plays asynchronously without freezing video frames.
     - **Screen Flash**: Perimeter flashes red with a prominent top warning banner.
     - **Zone Color Change**: Polygon turns from calm green to glowing alert red.
   - Press `'q'` to gracefully close the pipeline.

---

### Phase 4: Browser Surveillance Dashboard (3 Minutes)
> **Presenter:**
> *"Next, we demonstrate the security supervisor's web dashboard built with Streamlit."*

1. **Execute Command:**
   ```powershell
   streamlit run src/ui/dashboard.py
   ```
2. **Open Browser:** `http://localhost:8501`
3. **Showcase Dashboard Capabilities:**
   - **KPI Metric Cards**: Real-time status, active intrusions, cumulative objects seen, and live FPS.
   - **Live Annotated Stream**: Center display updating smoothly in the browser.
   - **Sidebar Interactive Controls**:
     - Adjust Confidence Slider (e.g. 0.3 to 0.7) and observe detection sensitivity change dynamically.
     - Filter monitored classes using the multi-select dropdown.
     - Switch architecture between **YOLOv8 Nano** (high speed) and **YOLOv8 Small** (high precision).
   - **Incident Log with Evidence**:
     - Show the database table recording every intrusion with timestamp, track ID, and photographic snapshot link.
     - Point out `outputs/snapshots/` where evidence photos are saved.

---

### Phase 5: Automated Test Verification (1 Minute)
> **Presenter:**
> *"To ensure software engineering rigor, we have written comprehensive automated unit test suites."*

1. **Execute Command:**
   ```powershell
   pytest tests/ -v
   ```
2. **Explain Results:**
   - Geometry tests prove point-in-polygon accuracy (inside, outside, edge points).
   - State machine tests prove debounce logic fires exactly one entry and exit.
   - Metrics tests prove accurate unique object counting and dwell time tracking.

---

### Phase 6: Conclusion & Questions
> **Presenter:**
> *"In summary, SmartObjectDetector provides an end-to-end edge-deployable security solution exceeding the 15 FPS objective, eliminating false alarms through debounced tracking, and preserving immutable incident evidence. We are now open for any questions."*
