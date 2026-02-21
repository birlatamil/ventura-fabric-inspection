"""
config.py
=========
Central configuration for the Fabric Width Measurement System.
Edit this file to match your hardware setup before running.
"""

from dataclasses import dataclass, field
from typing import Tuple, List
import os


@dataclass
class CameraConfig:
    """Per-camera RTSP and capture settings."""
    rtsp_url: str = "rtsp://admin:password@192.168.1.100:554/stream1"
    name: str = "camera"
    reconnect_initial_delay_s: float = 1.0
    reconnect_max_delay_s: float = 16.0
    frame_timeout_s: float = 0.5          # max age of a "live" frame
    cap_buffer_size: int = 1              # minimize OpenCV buffer to reduce latency
    cap_api: int = 0                      # 0 = cv2.CAP_ANY


@dataclass
class CalibrationConfig:
    """Paths to pre-computed calibration JSON files."""
    calib_dir: str = "calibration_data"
    left_calib_file: str = "left_calib.json"
    right_calib_file: str = "right_calib.json"

    @property
    def left_path(self) -> str:
        return os.path.join(self.calib_dir, self.left_calib_file)

    @property
    def right_path(self) -> str:
        return os.path.join(self.calib_dir, self.right_calib_file)


@dataclass
class MeasurementConfig:
    """Measurement pipeline parameters."""
    # Center strip extraction
    strip_height_px: int = 40         # number of rows to process
    strip_rows_per_line: int = 1      # step between sampled rows

    # Sobel edge detection
    sobel_ksize: int = 3
    sobel_threshold: float = 20.0     # minimum gradient to count as edge

    # Subpixel parabolic fit half-window
    subpixel_half_window: int = 2

    # Moving average
    moving_avg_window: int = 5        # frames

    # Fabric width validity range
    fabric_min_mm: float = 800.0
    fabric_max_mm: float = 1900.0

    # Confidence: minimum SNR ratio to trust a measurement
    min_confidence: float = 0.6

    # Warped bird's-eye output image size (pixels) → represents mm space
    warp_output_width_px: int = 2048
    warp_output_height_px: int = 256

    # Real-world mm extent the warp_output covers (must match calibration)
    # Left camera: covers 0..left_fov_mm_width
    # Right camera: covers 0..right_fov_mm_width
    left_fov_mm_width: float = 1000.0
    right_fov_mm_width: float = 1000.0


@dataclass
class OutputConfig:
    """Output cadence config."""
    output_interval_s: float = 0.200  # 200 ms = 5 Hz
    measurement_fps: int = 25         # target internal measurement loop rate


@dataclass
class MQTTConfig:
    """MQTT broker settings."""
    broker: str = "localhost"
    port: int = 1883
    topic_width: str = "fabric/width"
    topic_health: str = "fabric/health"
    client_id: str = "fabric_width_system"
    keepalive: int = 60
    qos: int = 1
    retain: bool = True
    reconnect_delay_s: float = 5.0
    username: str = ""                # leave empty if no auth
    password: str = ""


@dataclass
class WatchdogConfig:
    """Watchdog and health monitoring settings."""
    check_interval_s: float = 5.0     # how often watchdog checks health
    kick_timeout_s: float = 10.0      # seconds since last kick before alarm
    max_memory_mb: float = 500.0      # RSS memory limit
    min_measurement_rate_hz: float = 2.0  # alarm if below this


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = "logs"
    log_file: str = "fabric_width.log"
    level: str = "INFO"               # DEBUG | INFO | WARNING | ERROR
    max_bytes: int = 20 * 1024 * 1024  # 20 MB per file
    backup_count: int = 30            # keep 30 rotated files
    console: bool = True


@dataclass
class SystemConfig:
    """Root system configuration — single source of truth."""
    left_camera: CameraConfig = field(default_factory=lambda: CameraConfig(
        rtsp_url="rtsp://admin:password@192.168.1.100:554/stream1",
        name="left"
    ))
    right_camera: CameraConfig = field(default_factory=lambda: CameraConfig(
        rtsp_url="rtsp://admin:password@192.168.1.101:554/stream1",
        name="right"
    ))
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mqtt: MQTTConfig = field(default_factory=MQTTConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Dry-run mode: use synthetic frames (no real cameras required)
    dry_run: bool = False
    dry_run_fabric_width_mm: float = 1200.0  # synthetic truth for dry-run


# ── Singleton accessor ──────────────────────────────────────────────────────
_config: SystemConfig | None = None


def get_config() -> SystemConfig:
    """Return the global SystemConfig singleton."""
    global _config
    if _config is None:
        _config = SystemConfig()
    return _config


def set_config(cfg: SystemConfig) -> None:
    """Replace the global config (useful for tests and CLI overrides)."""
    global _config
    _config = cfg
