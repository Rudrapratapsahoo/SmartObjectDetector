"""
Multi-Object Tracking (ByteTrack) wrapper.
Assigns and maintains persistent track IDs for detected objects across frames.
Supports IoU and Centroid association with Ultralytics / Supervision compatibility.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


class SimpleIoUTracker:
    """Robust tracker using IoU and Centroid proximity association."""

    def __init__(self, max_lost=30, iou_threshold=0.25, max_distance=250.0):
        self.next_id = 1
        self.tracks = {}  # id -> {"box": (x1,y1,x2,y2), "class": str, "lost": 0, "confidence": float}
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance

    @staticmethod
    def _compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = float(boxAArea + boxBArea - interArea)
        return interArea / unionArea if unionArea > 0 else 0.0

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def update(self, detections):
        # If detections already have an assigned ID (e.g., from upstream or test harness)
        if detections and all("id" in d and d["id"] is not None for d in detections):
            return [dict(d) for d in detections]

        matched_det_indices = set()
        matched_track_ids = set()
        active_tracks = []

        # 1. First Pass: Match existing tracks using IoU
        for tid, tinfo in list(self.tracks.items()):
            best_iou = self.iou_threshold
            best_det_idx = -1

            for idx, det in enumerate(detections):
                if idx in matched_det_indices:
                    continue
                if det["class"] != tinfo["class"]:
                    continue
                iou = self._compute_iou(tinfo["box"], det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = idx

            if best_det_idx != -1:
                det = detections[best_det_idx]
                self.tracks[tid]["box"] = det["box"]
                self.tracks[tid]["confidence"] = det.get("confidence", 1.0)
                self.tracks[tid]["lost"] = 0
                matched_det_indices.add(best_det_idx)
                matched_track_ids.add(tid)
                active_tracks.append({
                    "id": tid,
                    "box": det["box"],
                    "class": det["class"],
                    "confidence": det.get("confidence", 1.0)
                })

        # 2. Second Pass: Centroid distance matching for fast motion or non-overlapping steps
        for tid, tinfo in list(self.tracks.items()):
            if tid in matched_track_ids:
                continue

            tc = self._center(tinfo["box"])
            best_dist = self.max_distance
            best_det_idx = -1

            for idx, det in enumerate(detections):
                if idx in matched_det_indices:
                    continue
                if det["class"] != tinfo["class"]:
                    continue
                dc = self._center(det["box"])
                dist = np.hypot(tc[0] - dc[0], tc[1] - dc[1])
                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = idx

            if best_det_idx != -1:
                det = detections[best_det_idx]
                self.tracks[tid]["box"] = det["box"]
                self.tracks[tid]["confidence"] = det.get("confidence", 1.0)
                self.tracks[tid]["lost"] = 0
                matched_det_indices.add(best_det_idx)
                matched_track_ids.add(tid)
                active_tracks.append({
                    "id": tid,
                    "box": det["box"],
                    "class": det["class"],
                    "confidence": det.get("confidence", 1.0)
                })
            else:
                self.tracks[tid]["lost"] += 1

        # 3. Create new tracks for unmatched detections
        for idx, det in enumerate(detections):
            if idx not in matched_det_indices:
                tid = det.get("id") or self.next_id
                if tid >= self.next_id:
                    self.next_id = tid + 1
                self.tracks[tid] = {
                    "box": det["box"],
                    "class": det["class"],
                    "lost": 0,
                    "confidence": det.get("confidence", 1.0)
                }
                active_tracks.append({
                    "id": tid,
                    "box": det["box"],
                    "class": det["class"],
                    "confidence": det.get("confidence", 1.0)
                })

        # 4. Remove expired tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["lost"] > self.max_lost:
                del self.tracks[tid]

        return active_tracks

    def reset(self):
        self.next_id = 1
        self.tracks.clear()


class MultiObjectTracker:
    """ByteTrack wrapper integrating supervision library with graceful fallback."""

    def __init__(self, track_thresh=0.25, track_buffer=30, match_thresh=0.8):
        self.use_supervision = False
        self.sv_tracker = None
        self.fallback_tracker = SimpleIoUTracker(max_lost=track_buffer, iou_threshold=0.25)

        try:
            import supervision as sv
            if hasattr(sv, "ByteTrack"):
                self.sv = sv
                self.sv_tracker = sv.ByteTrack(
                    track_activation_threshold=track_thresh,
                    lost_track_buffer=track_buffer,
                    minimum_matching_threshold=match_thresh
                )
                self.use_supervision = True
            else:
                self.use_supervision = False
        except Exception as e:
            self.use_supervision = False

    def update(self, detections):
        """
        Takes detection list:
            [{"box": (x1, y1, x2, y2), "class": "...", "confidence": 0.xx}, ...]
        Returns tracked list:
            [{"id": int, "box": (x1, y1, x2, y2), "class": "...", "confidence": float}, ...]
        """
        if not detections:
            return []

        if self.use_supervision and self.sv_tracker is not None:
            try:
                boxes = np.array([d["box"] for d in detections], dtype=float)
                confs = np.array([d.get("confidence", 0.9) for d in detections], dtype=float)
                class_names = [d.get("class", "object") for d in detections]

                unique_classes = list(set(class_names))
                cls_to_id = {name: idx for idx, name in enumerate(unique_classes)}
                class_ids = np.array([cls_to_id[name] for name in class_names], dtype=int)

                sv_detections = self.sv.Detections(
                    xyxy=boxes,
                    confidence=confs,
                    class_id=class_ids
                )

                tracked = self.sv_tracker.update_with_detections(sv_detections)

                output = []
                for i in range(len(tracked)):
                    xyxy = tracked.xyxy[i]
                    box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else (i + 1)
                    cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                    cls_name = unique_classes[cls_id] if cls_id < len(unique_classes) else "object"
                    conf = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0

                    output.append({
                        "id": track_id,
                        "box": box,
                        "class": cls_name,
                        "confidence": conf
                    })
                return output
            except Exception:
                return self.fallback_tracker.update(detections)
        else:
            return self.fallback_tracker.update(detections)

    def reset(self):
        """Reset internal tracking state."""
        if self.use_supervision and self.sv_tracker is not None:
            try:
                self.sv_tracker.reset()
            except Exception:
                pass
        self.fallback_tracker.reset()
