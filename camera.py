"""
camera.py
=========
Thread-safe RTSP camera reader with automatic reconnection.

One RTSPCamera instance per physical camera.  A dedicated daemon thread 
continuously reads frames from the RTSP stream so that get_frame() always
returns the LATEST frame with zero buffering lag.

Features
--------
* Exponential-backoff reconnect (never crashes the main process)
* Configurable reconnect delays
* is_healthy() — checks if a fresh frame arrived within timeout_s
* Thread-safe via threading primitives
* CPU-friendly: only reads as fast as the camera delivers
"""

from __future__ import annotations

import logging
import time
import threading
from collections import deque
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class RTSPCamera:
    """
    Background-threaded RTSP camera reader with auto-reconnect.

    Parameters
    ----------
    rtsp_url : str
        Full RTSP URL including credentials if required.
    name : str
        Human-readable camera name for logging.
    reconnect_initial_delay_s : float
        Initial delay between reconnect attempts.
    reconnect_max_delay_s : float
        Maximum delay cap for exponential backoff.
    frame_timeout_s : float
        Age (seconds) beyond which a frame is considered stale.
    cap_buffer_size : int
        OpenCV VideoCapture internal buffer size (keep at 1 for low latency).
    cap_api : int
        OpenCV capture API (cv2.CAP_ANY = 0).
    """

    def __init__(
        self,
        rtsp_url: str,
        name: str = "camera",
        reconnect_initial_delay_s: float = 1.0,
        reconnect_max_delay_s: float = 16.0,
        frame_timeout_s: float = 0.5,
        cap_buffer_size: int = 1,
        cap_api: int = 0,
    ) -> None:
        self._url = rtsp_url
        self._name = name
        self._reconnect_init = reconnect_initial_delay_s
        self._reconnect_max = reconnect_max_delay_s
        self._frame_timeout = frame_timeout_s
        self._cap_buffer = cap_buffer_size
        self._cap_api = cap_api

        # Frame store — deque(maxlen=1) gives us the latest frame atomically
        self._frame_buf: deque[np.ndarray] = deque(maxlen=1)
        self._last_ts: float = 0.0
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()

        # Control
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Stats
        self._reconnect_count: int = 0
        self._frames_received: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background reader thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"rtsp-{self._name}",
            daemon=True,
        )
        self._thread.start()
        log.info("[%s] Camera thread started (url=%s)", self._name, self._url)

    def stop(self) -> None:
        """Signal the reader thread to stop and wait for it to exit."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                log.warning("[%s] Reader thread did not stop cleanly.", self._name)
        log.info(
            "[%s] Stopped. frames_received=%d reconnects=%d",
            self._name, self._frames_received, self._reconnect_count,
        )

    def get_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        """
        Return the latest frame or None if no fresh frame is available.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for a new frame event.
        """
        self._new_frame_event.wait(timeout=timeout)
        self._new_frame_event.clear()
        with self._lock:
            if self._frame_buf:
                return self._frame_buf[0].copy()
        return None

    def is_healthy(self) -> bool:
        """True if a frame was received within the configured timeout window."""
        with self._lock:
            age = time.monotonic() - self._last_ts
        return age < self._frame_timeout and self._last_ts > 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def frames_received(self) -> int:
        return self._frames_received

    # ── Internal ───────────────────────────────────────────────────────────

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Open an RTSP VideoCapture with low-latency settings."""
        cap = cv2.VideoCapture(self._url, self._cap_api)
        if not cap.isOpened():
            cap.release()
            return None
        # Minimise internal buffer to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self._cap_buffer)
        # Request hardware-accelerated decode if available
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        log.info(
            "[%s] Connected — %dx%d @ %.1f fps",
            self._name,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
        )
        return cap

    def _reader_loop(self) -> None:
        """Main loop: connect → read frames → reconnect on failure."""
        delay = self._reconnect_init

        while not self._stop_evt.is_set():
            cap = self._open_capture()

            if cap is None:
                log.warning(
                    "[%s] Could not connect. Retrying in %.1fs …",
                    self._name, delay,
                )
                self._stop_evt.wait(timeout=delay)
                delay = min(delay * 2, self._reconnect_max)
                self._reconnect_count += 1
                continue

            # Connected — reset backoff
            delay = self._reconnect_init

            while not self._stop_evt.is_set():
                ret, frame = cap.read()
                if not ret:
                    log.warning(
                        "[%s] Read failed — scheduling reconnect.", self._name
                    )
                    break

                ts = time.monotonic()
                with self._lock:
                    self._frame_buf.append(frame)
                    self._last_ts = ts
                self._frames_received += 1
                self._new_frame_event.set()

            cap.release()

            if not self._stop_evt.is_set():
                log.info(
                    "[%s] Disconnected. Reconnect #%d in %.1fs …",
                    self._name, self._reconnect_count + 1, delay,
                )
                self._stop_evt.wait(timeout=delay)
                delay = min(delay * 2, self._reconnect_max)
                self._reconnect_count += 1


class SyntheticCamera:
    """
    Drop-in replacement for RTSPCamera that generates synthetic frames.
    Used in dry-run mode for testing without real cameras.

    Generates a white frame with two thin dark vertical bars simulating
    fabric edges at configurable positions.
    """

    def __init__(
        self,
        name: str = "synthetic",
        width_px: int = 1920,
        height_px: int = 1080,
        fps: float = 30.0,
        left_edge_px: Optional[int] = None,
        right_edge_px: Optional[int] = None,
        jitter_px: float = 0.5,
    ) -> None:
        self._name = name
        self._w = width_px
        self._h = height_px
        self._fps = fps
        self._period = 1.0 / fps
        self._left_px = left_edge_px if left_edge_px is not None else int(width_px * 0.1)
        self._right_px = right_edge_px if right_edge_px is not None else int(width_px * 0.9)
        self._jitter = jitter_px

        self._stop_evt = threading.Event()
        self._frame_buf: deque[np.ndarray] = deque(maxlen=1)
        self._last_ts: float = 0.0
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frames_received: int = 0

    def start(self) -> None:
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._gen_loop,
            name=f"synthetic-{self._name}",
            daemon=True,
        )
        self._thread.start()
        log.info("[%s] Synthetic camera started.", self._name)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        self._new_frame_event.wait(timeout=timeout)
        self._new_frame_event.clear()
        with self._lock:
            if self._frame_buf:
                return self._frame_buf[0].copy()
        return None

    def is_healthy(self) -> bool:
        with self._lock:
            age = time.monotonic() - self._last_ts
        return age < 1.0 and self._last_ts > 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def reconnect_count(self) -> int:
        return 0

    @property
    def frames_received(self) -> int:
        return self._frames_received

    def _make_frame(self) -> np.ndarray:
        frame = np.full((self._h, self._w, 3), 220, dtype=np.uint8)
        rng = np.random.default_rng()
        jl = int(self._left_px + rng.uniform(-self._jitter, self._jitter))
        jr = int(self._right_px + rng.uniform(-self._jitter, self._jitter))
        # Draw dark vertical bars (fabric edges)
        cv2.rectangle(frame, (max(0, jl - 4), 0), (jl + 4, self._h), (20, 20, 20), -1)
        cv2.rectangle(frame, (jr - 4, 0), (min(self._w - 1, jr + 4), self._h), (20, 20, 20), -1)
        return frame

    def _gen_loop(self) -> None:
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            frame = self._make_frame()
            with self._lock:
                self._frame_buf.append(frame)
                self._last_ts = time.monotonic()
            self._frames_received += 1
            self._new_frame_event.set()
            elapsed = time.monotonic() - t0
            sleep = self._period - elapsed
            if sleep > 0:
                time.sleep(sleep)
