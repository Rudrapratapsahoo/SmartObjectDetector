"""
Object Counting and Metrics Management.
Maintains per-frame current counts, cumulative unique counts, inside-zone counts,
and dwell time calculations.
"""

from collections import Counter, defaultdict
import time


class ObjectMetricsCounter:
    """Tracks current frame counts, cumulative unique IDs per class, and zone metrics."""

    def __init__(self):
        self.seen_track_ids = defaultdict(set)  # class_name -> set(track_ids)
        self.entry_timestamps = {}              # track_id -> float (time entered)
        self.dwell_durations = {}               # track_id -> float (final or current dwell seconds)

    def reset(self):
        """Reset all counters and tracked IDs."""
        self.seen_track_ids.clear()
        self.entry_timestamps.clear()
        self.dwell_durations.clear()

    def update_seen_tracks(self, tracks):
        """Record all tracks visible in the current frame."""
        for t in tracks:
            tid = t.get("id")
            cls = t.get("class", "object")
            if tid is not None:
                self.seen_track_ids[cls].add(tid)

    def record_entry(self, track_id, timestamp=None):
        """Record start time when an object enters the restricted zone."""
        if timestamp is None:
            timestamp = time.time()
        self.entry_timestamps[track_id] = timestamp

    def record_exit(self, track_id, timestamp=None):
        """Record exit and compute total dwell duration in seconds."""
        if timestamp is None:
            timestamp = time.time()
        enter_time = self.entry_timestamps.pop(track_id, None)
        if enter_time is not None:
            duration = max(0.0, timestamp - enter_time)
            self.dwell_durations[track_id] = duration
            return duration
        return 0.0

    def get_current_dwell_time(self, track_id, current_time=None):
        """Get elapsed dwell time for an object currently inside the zone."""
        if track_id in self.entry_timestamps:
            if current_time is None:
                current_time = time.time()
            return max(0.0, current_time - self.entry_timestamps[track_id])
        return self.dwell_durations.get(track_id, 0.0)

    def compute_counts(self, tracks, inside_ids):
        """
        Calculates the three required count categories:
        1. current: {cls: count} for objects visible in current frame
        2. unique: {cls: count} for all distinct objects seen across entire video
        3. inside_zone: count of objects currently inside the zone
        """
        self.update_seen_tracks(tracks)

        current_counts = Counter(t.get("class", "object") for t in tracks)
        unique_counts = {cls: len(ids) for cls, ids in self.seen_track_ids.items()}

        return {
            "current": dict(current_counts),
            "unique": unique_counts,
            "inside_zone": len(inside_ids)
        }
