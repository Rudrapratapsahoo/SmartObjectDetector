"""
Unit tests for Object Metrics Counter: current counts, unique counts, and inside-zone counts.
"""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.zone.counters import ObjectMetricsCounter


def test_object_metrics_counter_basic():
    counter = ObjectMetricsCounter()

    # Frame 1: Person 1 and Backpack 2
    tracks_f1 = [
        {"id": 1, "class": "person"},
        {"id": 2, "class": "backpack"}
    ]
    inside_f1 = {1}  # Person 1 is inside zone

    c1 = counter.compute_counts(tracks_f1, inside_f1)
    assert c1["current"] == {"person": 1, "backpack": 1}
    assert c1["unique"] == {"person": 1, "backpack": 1}
    assert c1["inside_zone"] == 1

    # Frame 2: Person 1 still present, new Person 3 arrives, Backpack 2 left
    tracks_f2 = [
        {"id": 1, "class": "person"},
        {"id": 3, "class": "person"}
    ]
    inside_f2 = set()  # Neither inside zone

    c2 = counter.compute_counts(tracks_f2, inside_f2)
    assert c2["current"] == {"person": 2}
    # Unique should remember Person 1, Person 3 (total 2) and Backpack 2 (total 1)
    assert c2["unique"]["person"] == 2
    assert c2["unique"]["backpack"] == 1
    assert c2["inside_zone"] == 0


def test_dwell_time_tracking():
    counter = ObjectMetricsCounter()

    # Object 10 enters at timestamp 100.0
    counter.record_entry(track_id=10, timestamp=100.0)

    # Check current dwell at timestamp 112.5
    current_dwell = counter.get_current_dwell_time(track_id=10, current_time=112.5)
    assert round(current_dwell, 1) == 12.5

    # Object 10 exits at timestamp 120.0
    final_dwell = counter.record_exit(track_id=10, timestamp=120.0)
    assert round(final_dwell, 1) == 20.0
