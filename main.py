"""
main.py
=======
Fabric Width Measurement System — Orchestrator

Starts all subsystems, runs the 200 ms measurement/output loop, and handles
graceful shutdown on SIGTERM / SIGINT / KeyboardInterrupt.

Architecture (dual camera, no stitching)
-----------------------------------------
  Left camera  → WidthMeasurer (left)  → left.left_mm  / left.right_mm
  Right camera → WidthMeasurer (right) → right.left_mm / right.right_mm

  Total width = left.right_mm + (right_fov_mm - right.left_mm)

  This assumes:
    - Left camera sees the LEFT half of the fabric (its left edge = fabric left)
    - Right camera sees the RIGHT half of the fabric (its right edge = fabric right)
    - right.left_mm is measured from the RIGHT camera's origin, so
        right_fabric_edge_from_right_origin = right_fov_mm - right.left_mm
        ... more precisely: total width = left.right_mm + (right.right_mm - right.left_mm)
      where right_fov_mm is captured in right calibration.

  Simpler model for cameras that each see one full edge:
    - left cam: measures distance from left origin to RIGHT fabric edge → left.right_mm
    - right cam: measures distance from right origin to LEFT fabric edge → right.left_mm
    - Total width = left.right_mm + right.left_mm

  ** Adjust the combine() function below to match YOUR physical setup **

Usage
-----
  Production:   python main.py
  Dry-run:      python main.py --dry-run
  Custom cfg:   Edit config.py and run python main.py
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from config import SystemConfig, get_config, set_config
from logger_setup import setup_logging, get_logger
from calibration import load_calibration, make_synthetic_calibration, CalibrationData
from camera import RTSPCamera, SyntheticCamera
from measurement import WidthMeasurer, MeasurementResult, SmoothedMeasurement
from mqtt_output import MQTTPublisher
from watchdog import SystemWatchdog

log = get_logger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))

# ── Global shutdown event ───────────────────────────────────────────────────
_shutdown = False


def _handle_signal(signum, frame) -> None:
    global _shutdown
    log.warning("Signal %d received — initiating graceful shutdown …", signum)
    _shutdown = True


# ── Width combination logic ─────────────────────────────────────────────────

def combine_widths(
    left_result: Optional[SmoothedMeasurement],
    right_result: Optional[SmoothedMeasurement],
    cfg: SystemConfig,
) -> Optional[dict]:
    """
    Combine smoothed measurements from both cameras into a single total width.

    MODEL (adjust to match physical layout):
      left cam origin = left fabric edge side
        → left_result.right_mm = position of LEFT fabric edge measured from left-cam origin
        → For a camera pointing at left half of fabric: left fabric edge ≈ 0,
           right edge of its view ≈ left_result.right_mm (but only partial view)
      
    Simple accumulation model used here:
      Left cam measures its portion   → left_result.right_mm  (from its left origin)
      Right cam measures its portion  → right_result.left_mm  (from fabric right, reversed)
      Total = left portion + right portion

    The combining formula MUST be validated against your physical setup.
    """
    if left_result is None and right_result is None:
        return None

    # If only one camera is available, extrapolate (lower confidence)
    if left_result is None:
        log.debug("Left camera measurement unavailable — using right only.")
        width_mm = right_result.width_mm  # type: ignore[union-attr]
        confidence = right_result.confidence * 0.5  # type: ignore[union-attr]
        left_mm = 0.0
        right_mm = width_mm
    elif right_result is None:
        log.debug("Right camera measurement unavailable — using left only.")
        width_mm = left_result.width_mm
        confidence = left_result.confidence * 0.5
        left_mm = left_result.left_mm
        right_mm = left_result.right_mm
    else:
        # Both cameras available — combine
        # Left cam: left_result.right_mm = distance from left origin to right edge of left half
        # Right cam: right_result.left_mm = distance from right origin to left edge of right half
        width_mm = left_result.right_mm + right_result.left_mm
        confidence = (left_result.confidence + right_result.confidence) / 2.0
        left_mm = left_result.left_mm
        right_mm = right_result.right_mm

    # Final range sanity check
    meas_cfg = cfg.measurement
    if not (meas_cfg.fabric_min_mm <= width_mm <= meas_cfg.fabric_max_mm):
        log.warning(
            "Combined width %.1f mm outside valid range [%.0f, %.0f] — skipping publish.",
            width_mm, meas_cfg.fabric_min_mm, meas_cfg.fabric_max_mm,
        )
        return None

    return {
        "width_mm": width_mm,
        "left_mm": left_mm,
        "right_mm": right_mm,
        "confidence": confidence,
    }


# ── Main system class ────────────────────────────────────────────────────────

class FabricWidthSystem:
    """
    Top-level system orchestrator.

    Binds cameras, calibration, measurement, output, and watchdog into a single
    run loop that outputs measurements at 5 Hz (every 200 ms).
    """

    def __init__(self, cfg: Optional[SystemConfig] = None) -> None:
        self.cfg = cfg or get_config()

        # ── Subsystem handles ──────────────────────────────────────────────
        self.left_cam = None
        self.right_cam = None
        self.left_calib: Optional[CalibrationData] = None
        self.right_calib: Optional[CalibrationData] = None
        self.left_measurer: Optional[WidthMeasurer] = None
        self.right_measurer: Optional[WidthMeasurer] = None
        self.mqtt: Optional[MQTTPublisher] = None
        self.watchdog: Optional[SystemWatchdog] = None

        # Output timing
        self._last_output_ts: float = 0.0
        self._output_interval = self.cfg.output.output_interval_s  # 0.200 s

        # Session stats
        self._loop_count: int = 0
        self._publish_count: int = 0
        self._start_time: float = 0.0

    def setup(self) -> None:
        """Initialise all subsystems. Raises on fatal errors."""
        cfg = self.cfg
        log.info("=== Fabric Width Measurement System starting ===")
        log.info("Mode: %s", "DRY-RUN (synthetic)" if cfg.dry_run else "PRODUCTION")

        # ── Calibration ────────────────────────────────────────────────────
        if cfg.dry_run:
            self.left_calib = make_synthetic_calibration(
                "left",
                fov_width_mm=cfg.measurement.left_fov_mm_width,
                fov_height_mm=300.0,
                warp_w_px=cfg.measurement.warp_output_width_px,
                warp_h_px=cfg.measurement.warp_output_height_px,
            )
            self.right_calib = make_synthetic_calibration(
                "right",
                fov_width_mm=cfg.measurement.right_fov_mm_width,
                fov_height_mm=300.0,
                warp_w_px=cfg.measurement.warp_output_width_px,
                warp_h_px=cfg.measurement.warp_output_height_px,
            )
        else:
            calib_cfg = cfg.calibration
            m_cfg = cfg.measurement
            self.left_calib = load_calibration(
                calib_path=calib_cfg.left_path,
                camera_name="left",
                warp_w_px=m_cfg.warp_output_width_px,
                warp_h_px=m_cfg.warp_output_height_px,
                fov_width_mm=m_cfg.left_fov_mm_width,
            )
            self.right_calib = load_calibration(
                calib_path=calib_cfg.right_path,
                camera_name="right",
                warp_w_px=m_cfg.warp_output_width_px,
                warp_h_px=m_cfg.warp_output_height_px,
                fov_width_mm=m_cfg.right_fov_mm_width,
            )
            log.info(
                "Calibration loaded | left reproj=%.4f px | right reproj=%.4f px",
                self.left_calib.reprojection_error,
                self.right_calib.reprojection_error,
            )

        # ── Cameras ────────────────────────────────────────────────────────
        m_cfg = cfg.measurement
        if cfg.dry_run:
            # Synthetic fabric width split across two cameras
            half = cfg.dry_run_fabric_width_mm / 2
            img_w, img_h = self.left_calib.image_size
            # Each synthetic cam shows one half
            self.left_cam = SyntheticCamera(
                name="left",
                width_px=img_w, height_px=img_h,
                left_edge_px=int(img_w * 0.05),
                right_edge_px=int(img_w * 0.50),
            )
            self.right_cam = SyntheticCamera(
                name="right",
                width_px=img_w, height_px=img_h,
                left_edge_px=int(img_w * 0.50),
                right_edge_px=int(img_w * 0.95),
            )
        else:
            lc = cfg.left_camera
            rc = cfg.right_camera
            self.left_cam = RTSPCamera(
                rtsp_url=lc.rtsp_url, name=lc.name,
                reconnect_initial_delay_s=lc.reconnect_initial_delay_s,
                reconnect_max_delay_s=lc.reconnect_max_delay_s,
                frame_timeout_s=lc.frame_timeout_s,
                cap_buffer_size=lc.cap_buffer_size,
                cap_api=lc.cap_api,
            )
            self.right_cam = RTSPCamera(
                rtsp_url=rc.rtsp_url, name=rc.name,
                reconnect_initial_delay_s=rc.reconnect_initial_delay_s,
                reconnect_max_delay_s=rc.reconnect_max_delay_s,
                frame_timeout_s=rc.frame_timeout_s,
                cap_buffer_size=rc.cap_buffer_size,
                cap_api=rc.cap_api,
            )

        # ── Measurers ──────────────────────────────────────────────────────
        self.left_measurer = WidthMeasurer(
            calib=self.left_calib,
            strip_height_px=m_cfg.strip_height_px,
            sobel_ksize=m_cfg.sobel_ksize,
            sobel_threshold=m_cfg.sobel_threshold,
            subpixel_half_window=m_cfg.subpixel_half_window,
            moving_avg_window=m_cfg.moving_avg_window,
            fabric_min_mm=m_cfg.fabric_min_mm,
            fabric_max_mm=m_cfg.fabric_max_mm,
            min_confidence=m_cfg.min_confidence,
        )
        self.right_measurer = WidthMeasurer(
            calib=self.right_calib,
            strip_height_px=m_cfg.strip_height_px,
            sobel_ksize=m_cfg.sobel_ksize,
            sobel_threshold=m_cfg.sobel_threshold,
            subpixel_half_window=m_cfg.subpixel_half_window,
            moving_avg_window=m_cfg.moving_avg_window,
            fabric_min_mm=m_cfg.fabric_min_mm,
            fabric_max_mm=m_cfg.fabric_max_mm,
            min_confidence=m_cfg.min_confidence,
        )

        # ── MQTT ───────────────────────────────────────────────────────────
        mq = cfg.mqtt
        self.mqtt = MQTTPublisher(
            broker=mq.broker, port=mq.port,
            topic_width=mq.topic_width, topic_health=mq.topic_health,
            client_id=mq.client_id, keepalive=mq.keepalive,
            qos=mq.qos, retain=mq.retain,
            reconnect_delay_s=mq.reconnect_delay_s,
            username=mq.username, password=mq.password,
        )
        self.mqtt.connect()

        # ── Watchdog ───────────────────────────────────────────────────────
        wd = cfg.watchdog
        self.watchdog = SystemWatchdog(
            cameras=[self.left_cam, self.right_cam],
            mqtt_pub=self.mqtt,
            check_interval_s=wd.check_interval_s,
            kick_timeout_s=wd.kick_timeout_s,
            max_memory_mb=wd.max_memory_mb,
            min_measurement_rate_hz=wd.min_measurement_rate_hz,
        )

        log.info("All subsystems initialised successfully.")

    def start(self) -> None:
        """Start cameras and watchdog threads."""
        self.left_cam.start()
        self.right_cam.start()
        self.watchdog.start()
        self._start_time = time.monotonic()
        log.info("Cameras and watchdog started.")
        # Brief warmup wait for cameras to produce first frames
        time.sleep(2.0)

    def run(self) -> None:
        """
        Main measurement loop — runs until _shutdown is set.
        Targets cfg.output.measurement_fps internally, publishes at 5 Hz.
        """
        target_period = 1.0 / self.cfg.output.measurement_fps  # e.g. 1/25 = 40ms
        output_period = self._output_interval  # 200ms

        log.info(
            "Entering main loop | internal=%.0f Hz | output=%.1f Hz",
            self.cfg.output.measurement_fps,
            1.0 / output_period,
        )

        left_smooth: Optional[SmoothedMeasurement] = None
        right_smooth: Optional[SmoothedMeasurement] = None

        while not _shutdown:
            t_loop_start = time.monotonic()

            # ── Grab frames ────────────────────────────────────────────────
            left_frame = self.left_cam.get_frame(timeout=target_period * 0.8)
            right_frame = self.right_cam.get_frame(timeout=target_period * 0.8)

            # ── Measure ───────────────────────────────────────────────────
            if left_frame is not None:
                left_raw = self.left_measurer.measure(left_frame)
                if left_raw is not None:
                    left_smooth = self.left_measurer.update_moving_average(left_raw)

            if right_frame is not None:
                right_raw = self.right_measurer.measure(right_frame)
                if right_raw is not None:
                    right_smooth = self.right_measurer.update_moving_average(right_raw)

            # Kick watchdog if at least one measurement succeeded
            if left_smooth is not None or right_smooth is not None:
                self.watchdog.kick()

            # ── Output every 200 ms ────────────────────────────────────────
            now = time.monotonic()
            if now - self._last_output_ts >= output_period:
                self._last_output_ts = now
                combined = combine_widths(left_smooth, right_smooth, self.cfg)
                if combined is not None:
                    jitter = self.left_measurer.get_jitter_mm() if left_smooth else None
                    ok = self.mqtt.publish_width(
                        width_mm=combined["width_mm"],
                        left_mm=combined["left_mm"],
                        right_mm=combined["right_mm"],
                        confidence=combined["confidence"],
                        num_lines=(left_smooth.num_lines if left_smooth else 0) +
                                  (right_smooth.num_lines if right_smooth else 0),
                        jitter_mm=jitter,
                    )
                    self._publish_count += 1
                    log.info(
                        "WIDTH %.3f mm | conf=%.2f | jitter=%s mm | pub=%s",
                        combined["width_mm"],
                        combined["confidence"],
                        f"{jitter:.3f}" if jitter else "N/A",
                        "OK" if ok else "FAIL",
                    )

            self._loop_count += 1

            # ── Pace the loop ──────────────────────────────────────────────
            elapsed = time.monotonic() - t_loop_start
            sleep_time = target_period - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def teardown(self) -> None:
        """Gracefully stop all subsystems and log session summary."""
        log.info("Teardown initiated …")

        if self.watchdog:
            self.watchdog.stop()
        if self.left_cam:
            self.left_cam.stop()
        if self.right_cam:
            self.right_cam.stop()
        if self.mqtt:
            self.mqtt.disconnect()

        runtime_s = time.monotonic() - self._start_time if self._start_time else 0
        log.info(
            "=== Session complete | runtime=%.1f s | loops=%d | publishes=%d ===",
            runtime_s, self._loop_count, self._publish_count,
        )


# ── Entry point ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fabric Width Measurement System"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic cameras (no RTSP hardware required)",
    )
    parser.add_argument(
        "--dry-run-width",
        type=float,
        default=1200.0,
        metavar="MM",
        help="Synthetic fabric width in mm for dry-run (default: 1200)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> int:
    global _shutdown

    args = parse_args()

    # ── Build config from args ─────────────────────────────────────────────
    cfg = get_config()
    if args.dry_run:
        cfg.dry_run = True
        cfg.dry_run_fabric_width_mm = args.dry_run_width
    cfg.logging.level = args.log_level

    # ── Logging ───────────────────────────────────────────────────────────
    setup_logging(
        log_dir=cfg.logging.log_dir,
        log_file=cfg.logging.log_file,
        level=cfg.logging.level,
        max_bytes=cfg.logging.max_bytes,
        backup_count=cfg.logging.backup_count,
        console=cfg.logging.console,
    )

    log.info("Python %s | PID %d", sys.version.split()[0], os.getpid())

    # ── Signal handlers ────────────────────────────────────────────────────
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Run ───────────────────────────────────────────────────────────────
    system = FabricWidthSystem(cfg)
    try:
        system.setup()
        system.start()
        system.run()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received.")
    except Exception as exc:
        log.critical("Fatal error in main: %s", exc, exc_info=True)
        return 1
    finally:
        system.teardown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
