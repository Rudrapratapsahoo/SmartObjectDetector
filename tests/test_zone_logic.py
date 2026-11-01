"""
Unit tests for restricted zone point-in-polygon logic and debounced state machine transitions.
Directly implements Section 2.2 and Task 8 validation requirements.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.zone.zone_counter import ZoneCounter


@pytest.fixture
def square_zone():
    """Returns a ZoneCounter initialized with a 200x200 square: [100,100] to [300,300]."""
    zc = ZoneCounter(debounce_n=3)
    # Define a square: (100,100) -> (300,100) -> (300,300) -> (100,300)
    zc.polygon = np.array([
        [100, 100],
        [300, 100],
        [300, 300],
        [100, 300]
    ], dtype=np.int32)
    return zc


def test_point_in_polygon_geometry(square_zone):
    """Test points clearly inside, clearly outside, and on the edge of the polygon."""
    # Clearly inside
    assert square_zone.is_point_inside(200, 200) is True
    assert square_zone.is_point_inside(150, 250) is True

    # Clearly outside
    assert square_zone.is_point_inside(50, 50) is False
    assert square_zone.is_point_inside(350, 200) is False
    assert square_zone.is_point_inside(200, 400) is False

    # Exactly on boundary / edge
    assert square_zone.is_point_inside(100, 200) is True
    assert square_zone.is_point_inside(300, 200) is True
    assert square_zone.is_point_inside(200, 100) is True
    assert square_zone.is_point_inside(200, 300) is True


def test_reference_point_is_bottom_center(square_zone):
    """Verify reference point is feet position: ((x1+x2)//2, y2)."""
    box = (100, 50, 200, 250)
    ref_x, ref_y = square_zone.get_reference_point(box)
    assert ref_x == 150
    assert ref_y == 250


def test_debounced_state_machine_entry_exit(square_zone):
    """
    Simulate track movement across frames with debounce_n = 3.
    Sequence: Outside, Outside, Inside, Inside, Inside, Inside, Outside, Outside, Outside
    Expected: Exactly one ZONE_ENTRY and one ZONE_EXIT event for Track #1.
    """
    # Box clearly outside: bottom-center at (50, 50)
    outside_box = (40, 20, 60, 50)
    # Box clearly inside: bottom-center at (200, 200)
    inside_box = (190, 150, 210, 200)

    # Frame sequence simulation
    sequence = [
        outside_box,  # Frame 1: outside
        outside_box,  # Frame 2: outside
        inside_box,   # Frame 3: inside (streak 1 - no entry yet)
        inside_box,   # Frame 4: inside (streak 2 - no entry yet)
        inside_box,   # Frame 5: inside (streak 3 - ENTRY fires!)
        inside_box,   # Frame 6: inside (already inside - no duplicate)
        outside_box,  # Frame 7: outside (outside streak 1 - no exit yet)
        outside_box,  # Frame 8: outside (outside streak 2 - no exit yet)
        outside_box,  # Frame 9: outside (outside streak 3 - EXIT fires!)
    ]

    all_events = []
    fake_time = 1000.0

    for idx, box in enumerate(sequence):
        fake_time += 1.0  # 1 second per frame
        # Fixed track ID 1 to test state machine transitions specifically
        fake_detections = [{
            "id": 1,
            "box": box,
            "class": "person",
            "confidence": 0.95
        }]
        tracks, events, counts = square_zone.process(fake_detections, frame_time=fake_time)
        all_events.extend(events)

    # Extract event types
    event_types = [e["event"] for e in all_events]

    # Verify exactly 1 entry and 1 exit
    assert event_types.count("ZONE_ENTRY") == 1, f"Expected 1 entry, got: {event_types}"
    assert event_types.count("ZONE_EXIT") == 1, f"Expected 1 exit, got: {event_types}"

    # Verify order: Entry first, then Exit
    entry_idx = event_types.index("ZONE_ENTRY")
    exit_idx = event_types.index("ZONE_EXIT")
    assert entry_idx < exit_idx, "ZONE_ENTRY must precede ZONE_EXIT"

    # Verify dwell time was calculated (entered frame 5 @ 1005s, exited frame 9 @ 1009s -> 4s)
    exit_event = [e for e in all_events if e["event"] == "ZONE_EXIT"][0]
    assert "dwell_time" in exit_event
    assert exit_event["dwell_time"] > 0


def test_reference_point_modes(square_zone):
    """Test switching between bottom_center (feet) and center reference point modes."""
    box = (100, 50, 200, 250)  # x1=100, y1=50, x2=200, y2=250

    # Default: bottom-center (150, 250)
    square_zone.set_reference_mode("bottom_center")
    assert square_zone.get_reference_point(box) == (150, 250)

    # Center mode: cx=(100+200)//2=150, cy=(50+250)//2=150
    square_zone.set_reference_mode("center")
    assert square_zone.get_reference_point(box) == (150, 150)

