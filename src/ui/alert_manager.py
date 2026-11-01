"""
Alert and Notification Manager.
Handles audio alerts (asynchronously without blocking the video pipeline),
visual screen flashing/banners, and per-object alert cooldowns.
"""

import time
import threading
import sys
import cv2
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import ALERT_COOLDOWN_SECONDS, ALERT_BANNER_DURATION_SECONDS


class AlertManager:
    """Manages audio alerts, visual overlays, and cooldown intervals for intrusions."""

    def __init__(self, cooldown=ALERT_COOLDOWN_SECONDS, banner_duration=ALERT_BANNER_DURATION_SECONDS):
        self.cooldown = cooldown
        self.banner_duration = banner_duration
        self.last_alert_time = {}   # track_id -> timestamp
        self.alert_until = 0.0      # Timestamp until which visual alert banner remains active
        self.current_alert_msg = ""
        self._sound_lock = threading.Lock()

    def can_trigger_alert(self, track_id, current_time=None):
        """Check if enough cooldown time has elapsed for this specific track ID."""
        if current_time is None:
            current_time = time.time()
        last_time = self.last_alert_time.get(track_id, 0.0)
        return (current_time - last_time) >= self.cooldown

    def trigger_alert(self, event, current_time=None):
        """
        Processes a ZONE_ENTRY event.
        Plays sound in a separate thread and activates visual alert banner.
        """
        if current_time is None:
            current_time = time.time()

        track_id = event.get("id")
        cls_name = event.get("class", "Object").capitalize()

        if self.can_trigger_alert(track_id, current_time):
            self.last_alert_time[track_id] = current_time
            self.alert_until = current_time + self.banner_duration
            self.current_alert_msg = f"ALERT: {cls_name} (ID #{track_id}) entered restricted zone!"

            # Play sound in background thread to prevent dropping video frames
            threading.Thread(target=self._play_beep_sound, daemon=True).start()
            return True
        return False

    def is_alert_active(self, current_time=None):
        """Returns True if the visual alert warning is currently active."""
        if current_time is None:
            current_time = time.time()
        return current_time < self.alert_until

    def _play_beep_sound(self):
        """Platform-aware non-blocking sound synthesizer."""
        with self._sound_lock:
            # 1. Try Windows native winsound
            try:
                import winsound
                # Frequency: 1200 Hz, Duration: 250 ms
                winsound.Beep(1200, 250)
                return
            except Exception:
                pass

            # 2. Try pygame mixer
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
                # Generate a short sine wave beep or system bell
                import numpy as np
                sample_rate = 22050
                freq = 1000
                duration = 0.2
                n_samples = int(sample_rate * duration)
                buf = (np.sin(2 * np.pi * np.arange(n_samples) * freq / sample_rate) * 32767).astype(np.int16)
                sound = pygame.sndarray.make_sound(buf)
                sound.play()
                time.sleep(0.2)
                return
            except Exception:
                pass

            # 3. Fallback terminal bell
            try:
                sys.stdout.write("\a")
                sys.stdout.flush()
            except Exception:
                pass

    def draw_alert_overlay(self, frame, current_time=None):
        """
        Renders flashing red border and alert banner on the video frame.
        """
        if not self.is_alert_active(current_time):
            return frame

        h, w = frame.shape[:2]

        # 1. Red perimeter border (8px)
        border_thickness = 8
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), border_thickness)

        # 2. Top warning banner with pulsating high-visibility styling
        banner_h = 50
        banner_y1 = 0
        banner_y2 = banner_h

        # Semi-transparent red banner background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, banner_y1), (w, banner_y2), (0, 0, 220), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # White warning text with black outline for readability
        msg = self.current_alert_msg or "RESTRICTED ZONE BREACH DETECTED!"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.75
        thickness = 2
        (tw, th), _ = cv2.getTextSize(msg, font, font_scale, thickness)
        tx = max(10, (w - tw) // 2)
        ty = 33

        # Black drop shadow
        cv2.putText(frame, msg, (tx + 1, ty + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        # White text
        cv2.putText(frame, msg, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return frame
