"""
Incident and Event Logger with SQLite storage and snapshot capture.
Maintains persistent records of zone intrusions, exits, and saves photographic evidence.
"""

import sqlite3
import cv2
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import DB_PATH, SNAPSHOTS_DIR


class EventLogger:
    """Manages SQLite storage and snapshot image preservation for security events."""

    def __init__(self, db_path=DB_PATH, snapshots_dir=SNAPSHOTS_DIR):
        self.db_path = Path(db_path)
        self.snapshots_dir = Path(snapshots_dir)

        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.init_database()

    def get_connection(self):
        """Creates and returns a connection to the SQLite database."""
        return sqlite3.connect(str(self.db_path), timeout=10.0)

    def init_database(self):
        """Initializes the database schema matching the project specification."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER,
                    class TEXT,
                    event TEXT,
                    timestamp TEXT,
                    snapshot_path TEXT
                )
            """)
            conn.commit()

    def log_event(self, event, frame=None):
        """
        Saves snapshot to disk (if frame provided) and inserts row into SQLite.

        Args:
            event (dict): Must contain 'id', 'class', 'event', 'time'.
            frame (np.ndarray, optional): Current video frame image for snapshot evidence.

        Returns:
            int: The inserted database row ID.
        """
        track_id = int(event.get("id", 0))
        cls_name = str(event.get("class", "object"))
        event_type = str(event.get("event", "ZONE_ENTRY"))
        timestamp = event.get("time") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        snapshot_rel_path = ""
        if frame is not None:
            # Create timestamped filename
            clean_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = f"alert_id{track_id}_{cls_name}_{clean_ts}.jpg"
            abs_snap_path = self.snapshots_dir / filename
            cv2.imwrite(str(abs_snap_path), frame)
            # Store relative path for portability
            snapshot_rel_path = f"outputs/snapshots/{filename}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (track_id, class, event, timestamp, snapshot_path)
                VALUES (?, ?, ?, ?, ?)
            """, (track_id, cls_name, event_type, timestamp, snapshot_rel_path))
            conn.commit()
            return cursor.lastrowid

    def fetch_latest_events(self, limit=20):
        """
        Fetches the latest N events as a list of dicts.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, track_id, class, event, timestamp, snapshot_path
                FROM events
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()

        events = []
        for r in rows:
            events.append({
                "id": r[0],
                "track_id": r[1],
                "class": r[2],
                "event": r[3],
                "timestamp": r[4],
                "snapshot_path": r[5]
            })
        return events

    def get_events_dataframe(self, limit=50):
        """
        Fetches events directly as a pandas DataFrame for Streamlit display.
        """
        with self.get_connection() as conn:
            query = """
                SELECT id, track_id, class, event, timestamp, snapshot_path
                FROM events
                ORDER BY id DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(limit,))
        return df

    def clear_events(self):
        """Clear all logged events."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events")
            conn.commit()


# Module-level singleton helper
_global_logger = None

def get_event_logger():
    global _global_logger
    if _global_logger is None:
        _global_logger = EventLogger()
    return _global_logger


if __name__ == "__main__":
    import numpy as np

    logger = EventLogger()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_event = {
        "id": 99,
        "class": "person",
        "event": "ZONE_ENTRY",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    row_id = logger.log_event(dummy_event, dummy_img)
    print(f"Logged test event with DB ID: {row_id}")
    latest = logger.fetch_latest_events(5)
    print(f"Latest logged events:\n{latest}")
