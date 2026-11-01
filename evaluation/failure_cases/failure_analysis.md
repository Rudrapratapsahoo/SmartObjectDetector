# Edge Case & Failure Mode Analysis

This document analyzes challenging surveillance scenarios encountered during stress testing, along with their mitigation strategies in SmartObjectDetector.

## 1. Severe Physical Occlusion (`case1_severe_occlusion.jpg`)
- **Symptom**: Target passes behind thick structural pillars or furniture, causing YOLOv8 confidence to drop below threshold for 1-2 frames.
- **System Mitigation**:
  - The ByteTrack tracker maintains active track state for up to `lost_track_buffer = 30` frames.
  - State machine debouncing ensures track ID is not prematurely reset or duplicated upon reappearance.

## 2. High-Speed Motion Blur (`case2_motion_blur.jpg`)
- **Symptom**: Fast running targets or low shutter speed cameras produce horizontal blurring, reducing edge sharpness.
- **System Mitigation**:
  - Centroid distance proximity fallback in tracking associates blurred detections with existing tracks even when IoU is low.
  - Recommended camera configuration: minimum 1/100s exposure in indoor surveillance setups.

## 3. Boundary Oscillation (Border Jitter)
- **Symptom**: Target pacing exactly along the polygon edge entering and leaving on alternate frames.
- **System Mitigation**:
  - Multi-frame debouncing ($N=3$ consecutive frames) guarantees single alerts without rapid flickering.
