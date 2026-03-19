"""
measurement.py
==============
Per-frame fabric width measurement pipeline.

Two measurement modes
---------------------
laser_mode = True  (default, recommended when a green laser is installed)
    The laser line projects across the full fabric width and appears as the
    brightest horizontal row in the image (even on a monochrome B/W camera).
    Steps:
      1. [optional] Pre-crop the raw frame to a narrow horizontal band around
         the expected laser position (speeds up remap + warp significantly).
      2. Undistort (cv2.remap).
      3. Bird's-eye warp (cv2.warpPerspective).
      4. Auto-detect the laser row by finding the row with the highest mean
         brightness (sum-projection along Y).
      5. Average over ±laser_row_band rows for robustness.
      6. Threshold the averaged row at laser_brightness_threshold.
      7. The left/right extents of the thresholded region give the fabric edges.
      8. Convert pixel columns → mm via the calibrated mm-per-pixel scale.

laser_mode = False  (legacy Sobel edge detector)
    Uses horizontal Sobel gradients on a centre strip to find edges.
    Same pipeline as v1 but optionally pre-crops the raw frame first.

Speed optimisation (both modes)
--------------------------------
raw_strip_height_px > 0:
    Crop the raw camera frame to a horizontal strip BEFORE remap+warp.
    For a 1920×1080 source a 120-row crop saves ~90 % of remap pixels.
    The crop is centred on the frame; adjust if your laser is off-centre.

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


# ── Result dataclasses ──────────────────────────────────────────────────────

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


# ── Subpixel refinement (used in Sobel mode) ────────────────────────────────

def _parabolic_subpixel(arr: np.ndarray, peak_idx: int, half_win: int = 2) -> float:
    """
    Refine a peak index to sub-pixel precision via parabolic (3-point) fit.

    Uses the half_win neighbourhood; falls back to integer peak if out of bounds.
    Returns a float column index.
    """
    n = len(arr)
    if peak_idx <= 0 or peak_idx >= n - 1:
        return float(peak_idx)

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
        Number of pixel rows in the centre strip to sample (Sobel mode).
    raw_strip_height_px : int
        Rows of the *raw* frame to keep before remap+warp (0 = full frame).
        Centre of the frame is assumed; tune if laser sits elsewhere.
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
    laser_mode : bool
        If True, use laser-line brightness detection instead of Sobel edges.
    laser_brightness_threshold : int
        Pixel value (0–255) above which a pixel is on the laser line.
    laser_min_width_frac : float
        Minimum fraction of image width the laser must span to be valid.
    laser_row_band : int
        Number of rows above/below the auto-detected laser row to average.
    """

    def __init__(
        self,
        calib: CalibrationData,
        strip_height_px: int = 40,
        raw_strip_height_px: int = 120,
        sobel_ksize: int = 3,
        sobel_threshold: float = 20.0,
        subpixel_half_window: int = 2,
        moving_avg_window: int = 5,
        fabric_min_mm: float = 800.0,
        fabric_max_mm: float = 1900.0,
        min_confidence: float = 0.6,
        laser_mode: bool = True,
        laser_brightness_threshold: int = 200,
        laser_min_width_frac: float = 0.10,
        laser_row_band: int = 5,
    ) -> None:
        self._calib = calib
        self._strip_h = strip_height_px
        self._raw_strip_h = raw_strip_height_px
        self._sobel_k = sobel_ksize
        self._sobel_th = sobel_threshold
        self._spx_hw = subpixel_half_window
        self._mov_win = moving_avg_window
        self._fabric_min = fabric_min_mm
        self._fabric_max = fabric_max_mm
        self._min_conf = min_confidence
        self._laser_mode = laser_mode
        self._laser_thresh = laser_brightness_threshold
        self._laser_min_w = laser_min_width_frac
        self._laser_band = laser_row_band

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
            "%.3f mm/px | strip_h=%d | raw_pre_crop=%d | mode=%s",
            calib.camera_name,
            calib.fov_width_mm,
            calib.warp_w_px,
            calib.warp_h_px,
            self._mm_per_px,
            strip_height_px,
            raw_strip_height_px,
            "LASER" if laser_mode else "SOBEL",
        )

        # Diagnostic counter (prints details once)
        self._diag_done = False

    # ── Public API ──────────────────────────────────────────────────────────

    def measure(self, frame: np.ndarray) -> Optional[MeasurementResult]:
        """
        Process one frame and return a MeasurementResult, or None if the
        measurement is not trustworthy.
        """
        ts = time.monotonic()

        # ── Step 0 — Optional raw-frame pre-crop (major speedup) ──────────
        # Crop to a horizontal strip BEFORE remap+warp so those expensive
        # ops work on far fewer pixels.
        if self._raw_strip_h > 0:
            src_h = frame.shape[0]
            cy_raw = src_h // 2
            half_raw = self._raw_strip_h // 2
            r0 = max(0, cy_raw - half_raw)
            r1 = min(src_h, cy_raw + half_raw)
            frame = frame[r0:r1]

        # ── Step 1 — Undistort ────────────────────────────────────────────
        undistorted = cv2.remap(
            frame,
            self._calib.map1,
            self._calib.map2,
            interpolation=cv2.INTER_LINEAR,
        )

        # ── Step 2 — Bird's-eye warp (output is in mm coordinate space) ──
        warped = cv2.warpPerspective(
            undistorted,
            self._calib.homography,
            self._warp_size,
            flags=cv2.INTER_LINEAR,
        )

        # ── Step 3 — Grayscale ────────────────────────────────────────────
        if warped.ndim == 3:
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            gray = warped

        # ── Dispatch to measurement mode ──────────────────────────────────
        if self._laser_mode:
            return self._measure_laser(gray, ts)
        else:
            return self._measure_sobel(gray, ts)

    # ── Laser-line detection ────────────────────────────────────────────────

    def _measure_laser(
        self, gray: np.ndarray, ts: float
    ) -> Optional[MeasurementResult]:
        """
        Detect fabric width using the laser line brightness.

        The green laser projected across the fabric width appears as the
        brightest horizontal row in the (monochrome) warped image.

        Algorithm
        ---------
        1. Project mean brightness along Y → 1-D vector of length H.
        2. Find the row (laser_row) with maximum mean brightness.
        3. Average the ±laser_row_band rows around laser_row.
        4. Threshold the averaged row at laser_brightness_threshold.
        5. Find the leftmost and rightmost pixels above threshold.
        6. Convert to mm, compute confidence as mean brightness / 255.
        """
        h, w = gray.shape

        # 1. Row-mean projection to find the laser row
        row_means = gray.mean(axis=1)          # shape (H,)
        laser_row = int(np.argmax(row_means))

        # Log detected row in DEBUG so you can tune things
        log.debug(
            "[%s] Laser row detected @ y=%d (mean=%.1f)",
            self._calib.camera_name, laser_row, row_means[laser_row],
        )

        # 2. Average over a band of rows around the peak for noise reduction
        r0 = max(0, laser_row - self._laser_band)
        r1 = min(h, laser_row + self._laser_band + 1)
        band = gray[r0:r1].astype(np.float32)
        avg_row = band.mean(axis=0)            # shape (W,)

        # 3. Threshold
        mask = avg_row >= self._laser_thresh   # boolean array

        bright_cols = np.where(mask)[0]
        if bright_cols.size == 0:
            log.debug(
                "[%s] No pixels above laser threshold %d — "
                "is the laser on? (max brightness in band: %.1f)",
                self._calib.camera_name, self._laser_thresh, avg_row.max(),
            )
            return None

        # 4. Extents
        l_col = int(bright_cols[0])
        r_col = int(bright_cols[-1])

        # 5. Sanity: laser must span a minimum fraction of frame width
        span_frac = (r_col - l_col) / w
        if span_frac < self._laser_min_w:
            log.debug(
                "[%s] Laser span %.1f%% < min %.1f%% — skipping.",
                self._calib.camera_name, span_frac * 100, self._laser_min_w * 100,
            )
            return None

        # 6. Convert to mm
        left_mm  = l_col * self._mm_per_px
        right_mm = r_col * self._mm_per_px
        width_mm = right_mm - left_mm

        # Confidence: normalised mean brightness of the laser pixels
        laser_brightness = float(avg_row[bright_cols].mean())
        confidence = min(laser_brightness / 255.0, 1.0)

        # Validity range check
        if not (self._fabric_min <= width_mm <= self._fabric_max):
            log.debug(
                "[%s] Laser width %.1f mm outside valid range [%.0f, %.0f].",
                self._calib.camera_name, width_mm, self._fabric_min, self._fabric_max,
            )
            return None

        if confidence < self._min_conf:
            log.debug(
                "[%s] Laser confidence %.2f below threshold %.2f.",
                self._calib.camera_name, confidence, self._min_conf,
            )
            return None

        return MeasurementResult(
            width_mm=width_mm,
            left_mm=left_mm,
            right_mm=right_mm,
            confidence=confidence,
            num_lines=r1 - r0,
            raw_widths_mm=[width_mm],
            timestamp=ts,
            camera_name=self._calib.camera_name,
        )

    # ── Sobel edge detection (legacy mode) ─────────────────────────────────

    def _measure_sobel(
        self, gray: np.ndarray, ts: float
    ) -> Optional[MeasurementResult]:
        """Original Sobel-gradient edge detection on the centre strip."""
        h, w = gray.shape

        # Extract centre horizontal strip from the warped image
        cy = h // 2
        half = self._strip_h // 2
        y0 = max(0, cy - half)
        y1 = min(h, cy + half)
        strip = gray[y0:y1]

        if strip.shape[0] == 0:
            log.warning("[%s] Strip is empty — frame too small?", self._calib.camera_name)
            return None

        sobel = cv2.Sobel(strip, cv2.CV_64F, 1, 0, ksize=self._sobel_k)
        abs_sobel = np.abs(sobel)

        # ── One-time diagnostic dump ────────────────────────────────────
        if not self._diag_done:
            self._diag_done = True
            row_maxes = abs_sobel.max(axis=1)
            log.warning(
                "[%s] DIAG | gray shape=%s min=%d max=%d mean=%.1f | "
                "strip shape=%s min=%d max=%d mean=%.1f | "
                "sobel max=%.2f mean=%.2f | "
                "rows_above_thresh=%d/%d (thresh=%.1f) | "
                "sobel row_maxes top5=%s",
                self._calib.camera_name,
                gray.shape, int(gray.min()), int(gray.max()), float(gray.mean()),
                strip.shape, int(strip.min()), int(strip.max()), float(strip.mean()),
                float(abs_sobel.max()), float(abs_sobel.mean()),
                int((row_maxes >= self._sobel_th).sum()), abs_sobel.shape[0],
                self._sobel_th,
                str(sorted(row_maxes, reverse=True)[:5]),
            )

        left_cols: list[float] = []
        right_cols: list[float] = []
        confidences: list[float] = []

        for row_idx in range(abs_sobel.shape[0]):
            row = abs_sobel[row_idx]
            if row.max() < self._sobel_th:
                continue

            noise = float(np.median(row)) + 1e-6
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

            l_sub = _parabolic_subpixel(row, l_peak_idx, self._spx_hw)
            r_sub = _parabolic_subpixel(row, r_peak_idx, self._spx_hw)

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

        mean_left_px  = float(np.mean(left_cols))
        mean_right_px = float(np.mean(right_cols))
        mean_conf     = float(np.mean(confidences))

        left_mm  = mean_left_px  * self._mm_per_px
        right_mm = mean_right_px * self._mm_per_px
        width_mm = right_mm - left_mm

        raw_widths = [(r - l) * self._mm_per_px for l, r in zip(left_cols, right_cols)]

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

    # ── Moving average & diagnostics ───────────────────────────────────────

    def update_moving_average(self, result: MeasurementResult) -> SmoothedMeasurement:
        """Push a new measurement into the moving average and return smoothed output."""
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
