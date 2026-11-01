"""
Video source handling module for webcam, video file, or RTSP/HTTP IP stream.
Resizes frames to a standardized resolution for consistent coordinate mapping.
"""

import cv2
import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT, DEFAULT_SOURCE


class VideoSource:
    """Encapsulates OpenCV VideoCapture with resolution normalization and robust error handling."""

    def __init__(self, source=DEFAULT_SOURCE, width=DEFAULT_FRAME_WIDTH, height=DEFAULT_FRAME_HEIGHT):
        self.raw_source = source
        self.width = width
        self.height = height
        self.cap = None
        self._open_source()

    def _open_source(self):
        """Parse source type and open OpenCV VideoCapture."""
        source_val = self.raw_source

        # Handle string representation of integer device IDs (e.g. "0")
        if isinstance(source_val, str) and source_val.isdigit():
            source_val = int(source_val)

        # On Windows, DirectShow backend can speed up webcam initialization
        if isinstance(source_val, int):
            self.cap = cv2.VideoCapture(source_val, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(source_val)
        else:
            self.cap = cv2.VideoCapture(source_val)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[VideoSource Error] Failed to open video source: '{self.raw_source}'. "
                f"Please ensure camera is connected, permissions granted, or file/URL is valid."
            )

        # Attempt to set hardware capture properties if supported
        if isinstance(source_val, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        """
        Reads the next frame, resizes it to standardized dimensions, and returns (ret, frame).
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # Standardize resolution for downstream zone and detection logic
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return True, frame

    def get_fps(self):
        """Return FPS of the source or fallback to 30.0."""
        if self.cap and self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                return fps
        return 30.0

    def restart(self):
        """Restart playback from the beginning (for video files)."""
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        """Release the capture resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def open_video_source(source=DEFAULT_SOURCE, width=DEFAULT_FRAME_WIDTH, height=DEFAULT_FRAME_HEIGHT):
    """Convenience helper function to instantiate VideoSource."""
    return VideoSource(source=source, width=width, height=height)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test video source capture")
    parser.add_argument("--source", default="0", help="Webcam index (0), video file path, or stream URL")
    args = parser.parse_args()

    print(f"Testing video source: {args.source}")
    try:
        vs = open_video_source(args.source)
        print("Video source opened successfully. Displaying frames. Press 'q' to quit.")
        while True:
            ret, frame = vs.read()
            if not ret:
                print("End of stream or cannot fetch frame.")
                break
            cv2.imshow("VideoSource Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vs.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error testing video source: {e}")
