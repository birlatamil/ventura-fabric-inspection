"""
watchdog.py
===========
System health watchdog for the Fabric Width Measurement System.

Responsibilities
----------------
* Verify cameras are producing fresh frames (is_healthy checks)
* Verify measurement loop is running above minimum rate (kick-based timer)
* Monitor process memory usage
* Publish health status via MQTT
* Log alerts on any anomaly
* Optionally attempt camera restart on failure

Usage
-----
    wd = SystemWatchdog(cameras=[left_cam, right_cam], mqtt_pub=publisher)
    wd.start()
    ...
    wd.kick()   # call from main loop to signal liveliness
    ...
    wd.stop()
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import List, Optional, Dict, Any, TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from camera import RTSPCamera
    from mqtt_output import MQTTPublisher

log = logging.getLogger(__name__)


class SystemWatchdog:
    """
    Daemon watchdog thread.

    Parameters
    ----------
    cameras : list
        List of RTSPCamera (or SyntheticCamera) instances to monitor.
    mqtt_pub : MQTTPublisher | None
        If provided, publishes health JSON to fabric/health every check.
    check_interval_s : float
        How often to run health checks.
    kick_timeout_s : float
        Seconds since last kick() before raising a measurement-loop alarm.
    max_memory_mb : float
        Upper memory limit (RSS) in MB before alerting.
    min_measurement_rate_hz : float
        Minimum acceptable measurement rate. Below this → alarm.
    """

    def __init__(
        self,
        cameras: List,
        mqtt_pub=None,
        check_interval_s: float = 5.0,
        kick_timeout_s: float = 10.0,
        max_memory_mb: float = 500.0,
        min_measurement_rate_hz: float = 2.0,
    ) -> None:
        self._cameras = cameras
        self._mqtt = mqtt_pub
        self._check_interval = check_interval_s
        self._kick_timeout = kick_timeout_s
        self._max_mem_mb = max_memory_mb
        self._min_rate = min_measurement_rate_hz

        self._last_kick: float = time.monotonic()
        self._kick_count: int = 0
        self._last_kick_count: int = 0
        self._last_check_ts: float = time.monotonic()

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._process = psutil.Process(os.getpid())

        # Alert flags — avoid spamming logs
        self._alerted: Dict[str, bool] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the watchdog daemon thread."""
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="watchdog",
            daemon=True,
        )
        self._thread.start()
        log.info("Watchdog started [check=%.1fs, kick_timeout=%.1fs]",
                 self._check_interval, self._kick_timeout)

    def stop(self) -> None:
        """Stop the watchdog thread."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        log.info("Watchdog stopped.")

    def kick(self) -> None:
        """
        Reset the watchdog timer.  Call this from the main measurement loop
        every time a measurement is successfully produced.
        """
        with self._lock:
            self._last_kick = time.monotonic()
            self._kick_count += 1

    # ── Internal ───────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            self._stop_evt.wait(timeout=self._check_interval)
            if self._stop_evt.is_set():
                break
            try:
                self._check()
            except Exception as exc:
                log.error("Watchdog check exception: %s", exc, exc_info=True)

    def _check(self) -> None:
        now = time.monotonic()
        health: Dict[str, Any] = {
            "ts": time.time(),
            "pid": os.getpid(),
        }
        all_ok = True

        # ── 1. Kick timer (main loop liveliness) ──────────────────────────
        with self._lock:
            last_kick = self._last_kick
            kick_count = self._kick_count

        time_since_kick = now - last_kick
        if time_since_kick > self._kick_timeout:
            self._alert("loop_stall",
                        f"Main loop stalled — no kick for {time_since_kick:.1f}s "
                        f"(timeout={self._kick_timeout}s)")
            health["loop_ok"] = False
            all_ok = False
        else:
            self._alert_clear("loop_stall")
            health["loop_ok"] = True

        # ── 2. Measurement rate ────────────────────────────────────────────
        elapsed = now - self._last_check_ts
        if elapsed > 0:
            rate_hz = (kick_count - self._last_kick_count) / elapsed
            health["measurement_rate_hz"] = round(rate_hz, 2)
            if rate_hz < self._min_rate and kick_count > 10:  # allow warmup
                self._alert("low_rate",
                            f"Measurement rate {rate_hz:.2f} Hz below minimum "
                            f"{self._min_rate:.1f} Hz")
                all_ok = False
            else:
                self._alert_clear("low_rate")
        self._last_kick_count = kick_count
        self._last_check_ts = now

        # ── 3. Camera health ───────────────────────────────────────────────
        cam_health: Dict[str, bool] = {}
        for cam in self._cameras:
            ok = cam.is_healthy()
            cam_health[cam.name] = ok
            if not ok:
                self._alert(f"cam_{cam.name}",
                            f"Camera '{cam.name}' is NOT producing frames")
                all_ok = False
            else:
                self._alert_clear(f"cam_{cam.name}")
        health["cameras"] = cam_health

        # ── 4. Memory ─────────────────────────────────────────────────────
        try:
            mem_mb = self._process.memory_info().rss / (1024 * 1024)
            health["memory_mb"] = round(mem_mb, 1)
            if mem_mb > self._max_mem_mb:
                self._alert("high_mem",
                            f"Memory usage {mem_mb:.1f} MB exceeds limit "
                            f"{self._max_mem_mb:.0f} MB")
                all_ok = False
            else:
                self._alert_clear("high_mem")
        except Exception:
            health["memory_mb"] = -1

        # ── 5. Overall status + MQTT publish ──────────────────────────────
        health["status"] = "OK" if all_ok else "DEGRADED"
        log.info(
            "Watchdog | %s | rate=%.2f Hz | mem=%.1f MB | cameras=%s",
            health["status"],
            health.get("measurement_rate_hz", 0),
            health.get("memory_mb", 0),
            str(cam_health),
        )

        if self._mqtt is not None:
            try:
                self._mqtt.publish_health(health)
            except Exception as exc:
                log.warning("Watchdog MQTT publish failed: %s", exc)

    def _alert(self, key: str, message: str) -> None:
        if not self._alerted.get(key):
            log.error("WATCHDOG ALERT [%s]: %s", key, message)
            self._alerted[key] = True

    def _alert_clear(self, key: str) -> None:
        if self._alerted.get(key):
            log.info("WATCHDOG RECOVER [%s]: back to normal.", key)
            self._alerted[key] = False
