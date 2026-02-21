"""
measurement.py
==============
Per-frame fabric width measurement pipeline.

Pipeline (per camera, per frame)
---------------------------------
1.  cv2.remap             — apply precomputed undistortion maps
2.  cv2.warpPerspective   — bird's-eye view (output already in mm space)
3.  Extract center strip  — configurable height in pixels
4.  cv2.Sobel(dx=1)       — horizontal gradient per row
5.  Find left + right edge per row (peak gradient magnitude)
6.  Parabolic subpixel refinement around each peak
7.  Convert pixel column → mm (linear, since warp output = mm space)
8.  Average edge positions across strip rows
9.  Width = right_edge_mm − left_edge_mm
10. Compute confidence metric from gradient SNR
11. Accumulate into moving average buffer

Data flow
---------
frame → WidthMeasurer.measure(frame) → MeasurementResult | None
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from calibration import CalibrationData

log = logging.getLogger(__name__)


# ── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class MeasurementResult:
    """Single-frame measurement output."""
    width_mm: float          # fabric width in millimetres
    left_mm: float           # left-edge position in mm (from camera origin)
    right_mm: float          # right-edge position in mm
    confidence: float        # 0.0 (no trust) … 1.0 (perfect)
    num_lines: int           # number of strip rows that contributed
    raw_widths_mm: list      # per-row widths before averaging
    timestamp: float         # monotonic seconds
    camera_name: str


@dataclass
class SmoothedMeasurement:
    """Output after applying the moving-average filter."""
    width_mm: float
    left_mm: float
    right_mm: float
    confidence: float
    num_lines: int
    timestamp: float
    camera_name: str
    window_size: int


# ── Subpixel refinement ─────────────────────────────────────────────────────

def _parabolic_subpixel(arr: np.ndarray, peak_idx: int, half_win: int = 2) -> float:
    """
    Refine a peak index to sub-pixel precision via parabolic (3-point) fit.

    Uses the half_win neighbourhood; falls back to integer peak if out of bounds.
    Returns a float column index.
    """
    n = len(arr)
    if peak_idx <= 0 or peak_idx >= n - 1:
        return float(peak_idx)

    # Use exactly 3 points centred on peak for the parabola
    y_m = float(arr[peak_idx - 1])
    y_0 = float(arr[peak_idx])
    y_p = float(arr[peak_idx + 1])

    denom = 2.0 * (2.0 * y_0 - y_m - y_p)
    if abs(denom) < 1e-9:
        return float(peak_idx)

    delta = (y_m - y_p) / denom
    return float(peak_idx) + delta


# ── Core measurement class ──────────────────────────────────────────────────

class WidthMeasurer:
    """
    Stateful width measurer for a single camera.

    Parameters
    ----------
    calib : CalibrationData
        Pre-loaded calibration (undistort maps + homography).
    strip_height_px : int
        Number of pixel rows in the center strip to sample.
    sobel_ksize : int
        Sobel kernel size (must be 1, 3, 5, or 7).
    sobel_threshold : float
        Minimum gradient magnitude to accept as a fabric edge.
    subpixel_half_window : int
        Half-window for parabolic subpixel fit.
    moving_avg_window : int
        Number of frames to include in the moving average.
    fabric_min_mm, fabric_max_mm : float
        Valid fabric width range; measurements outside are discarded.
    min_confidence : float
        Minimum confidence score (0–1) to accept a measurement.
    """

    def __init__(
        self,
        calib: CalibrationData,
        strip_height_px: int = 40,
        sobel_ksize: int = 3,
        sobel_threshold: float = 20.0,
        subpixel_half_window: int = 2,
        moving_avg_window: int = 5,
        fabric_min_mm: float = 800.0,
        fabric_max_mm: float = 1900.0,
        min_confidence: float = 0.6,
    ) -> None:
        self._calib = calib
        self._strip_h = strip_height_px
        self._sobel_k = sobel_ksize
        self._sobel_th = sobel_threshold
        self._spx_hw = subpixel_half_window
        self._mov_win = moving_avg_window
        self._fabric_min = fabric_min_mm
        self._fabric_max = fabric_max_mm
        self._min_conf = min_confidence

        # Precompute warp output size once
        self._warp_size = (calib.warp_w_px, calib.warp_h_px)  # (w, h)

        # mm-per-pixel conversion for the warped output
        self._mm_per_px = calib.fov_width_mm / calib.warp_w_px

        # Moving average buffer: deque of (width, left, right, confidence)
        self._history: deque[Tuple[float, float, float, float]] = deque(
            maxlen=moving_avg_window
        )

        log.info(
            "[%s] WidthMeasurer ready | fov=%.0f mm | warp=%dx%d | "
            "%.3f mm/px | strip_h=%d",
            calib.camera_name,
            calib.fov_width_mm,
            calib.warp_w_px,
            calib.warp_h_px,
            self._mm_per_px,
            strip_height_px,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def measure(self, frame: np.ndarray) -> Optional[MeasurementResult]:
        """
        Process one frame and return a MeasurementResult, or None if the
        measurement is not trustworthy.
        """
        ts = time.monotonic()

        # Step 1 — Undistort
        undistorted = cv2.remap(
            frame,
            self._calib.map1,
            self._calib.map2,
            interpolation=cv2.INTER_LINEAR,
        )

        # Step 2 — Bird's-eye warp (output is in mm coordinate space)
        warped = cv2.warpPerspective(
            undistorted,
            self._calib.homography,
            self._warp_size,
            flags=cv2.INTER_LINEAR,
        )

        # Step 3 — Extract center horizontal strip
        h, w = warped.shape[:2]
        cy = h // 2
        half = self._strip_h // 2
        y0 = max(0, cy - half)
        y1 = min(h, cy + half)
        strip = warped[y0:y1]

        if strip.shape[0] == 0:
            log.warning("[%s] Strip is empty — frame too small?", self._calib.camera_name)
            return None

        # Step 4 — Convert to grayscale and compute Sobel X
        if strip.ndim == 3:
            gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        else:
            gray = strip

        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self._sobel_k)
        abs_sobel = np.abs(sobel)

        # Step 5 & 6 — Per-row edge detection + subpixel refinement
        left_cols: list[float] = []
        right_cols: list[float] = []
        confidences: list[float] = []

        for row_idx in range(abs_sobel.shape[0]):
            row = abs_sobel[row_idx]
            if row.max() < self._sobel_th:
                continue  # no clear edge in this row

            # Noise floor estimate = median of row
            noise = float(np.median(row)) + 1e-6

            # Left half: find strongest positive gradient
            mid = w // 2
            left_half = row[:mid]
            right_half = row[mid:]

            if left_half.size == 0 or right_half.size == 0:
                continue

            l_peak_idx = int(np.argmax(left_half))
            r_peak_idx = int(np.argmax(right_half)) + mid

            l_val = left_half[l_peak_idx]
            r_val = right_half[r_peak_idx - mid]

            if l_val < self._sobel_th or r_val < self._sobel_th:
                continue

            # Subpixel refinement
            l_sub = _parabolic_subpixel(row, l_peak_idx, self._spx_hw)
            r_sub = _parabolic_subpixel(row, r_peak_idx, self._spx_hw)

            # Confidence = geometric mean of both edge SNRs, clamped to [0,1]
            snr_l = min(l_val / noise / 10.0, 1.0)
            snr_r = min(r_val / noise / 10.0, 1.0)
            conf = float(np.sqrt(snr_l * snr_r))

            left_cols.append(l_sub)
            right_cols.append(r_sub)
            confidences.append(conf)

        num_lines = len(left_cols)
        if num_lines == 0:
            log.debug("[%s] No valid edges found in strip.", self._calib.camera_name)
            return None

        # Step 7 — Average across rows and convert to mm
        mean_left_px = float(np.mean(left_cols))
        mean_right_px = float(np.mean(right_cols))
        mean_conf = float(np.mean(confidences))

        left_mm = mean_left_px * self._mm_per_px
        right_mm = mean_right_px * self._mm_per_px
        width_mm = right_mm - left_mm

        # Per-row widths (for diagnostics)
        raw_widths = [
            (r - l) * self._mm_per_px
            for l, r in zip(left_cols, right_cols)
        ]

        # Step 8 — Validity checks
        if not (self._fabric_min <= width_mm <= self._fabric_max):
            log.debug(
                "[%s] Width %.1f mm outside valid range [%.0f, %.0f] — skipped.",
                self._calib.camera_name, width_mm, self._fabric_min, self._fabric_max,
            )
            return None

        if mean_conf < self._min_conf:
            log.debug(
                "[%s] Confidence %.2f below threshold %.2f — skipped.",
                self._calib.camera_name, mean_conf, self._min_conf,
            )
            return None

        return MeasurementResult(
            width_mm=width_mm,
            left_mm=left_mm,
            right_mm=right_mm,
            confidence=mean_conf,
            num_lines=num_lines,
            raw_widths_mm=raw_widths,
            timestamp=ts,
            camera_name=self._calib.camera_name,
        )

    def update_moving_average(self, result: MeasurementResult) -> SmoothedMeasurement:
        """
        Push a new measurement into the moving average and return smoothed output.
        """
        self._history.append(
            (result.width_mm, result.left_mm, result.right_mm, result.confidence)
        )
        arr = np.array(self._history)
        return SmoothedMeasurement(
            width_mm=float(np.mean(arr[:, 0])),
            left_mm=float(np.mean(arr[:, 1])),
            right_mm=float(np.mean(arr[:, 2])),
            confidence=float(np.mean(arr[:, 3])),
            num_lines=result.num_lines,
            timestamp=result.timestamp,
            camera_name=result.camera_name,
            window_size=len(self._history),
        )

    def get_jitter_mm(self) -> Optional[float]:
        """
        Return the standard deviation of width measurements in the current window.
        Returns None if fewer than 2 samples are available.
        """
        if len(self._history) < 2:
            return None
        widths = [h[0] for h in self._history]
        return float(np.std(widths))

    def reset(self) -> None:
        """Clear the moving average history."""
        self._history.clear()
