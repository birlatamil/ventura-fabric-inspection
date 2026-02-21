"""
calibration.py
==============
Intrinsic / extrinsic camera calibration pipeline.

Responsibilities
----------------
* Load camera_matrix + dist_coeffs from JSON (produced by run_calibration.py)
* Precompute undistortion remap tables (cv2.initUndistortRectifyMap)
* Load or compute homography from real-world reference points
* Validate reprojection error (must be < 0.3 px)
* Expose a CalibrationData dataclass consumed by the measurement module

JSON schema expected in calibration_data/left_calib.json and right_calib.json
-----------------------------------------------------------------------------
{
  "camera_matrix":  [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs":    [k1, k2, p1, p2, k3],
  "image_width_px": 1920,
  "image_height_px": 1080,
  "homography": [[...], [...], [...]],           // 3x3, pixel → mm
  "ref_px_points": [[x,y], ...],                 // 4 source points (pixel)
  "ref_mm_points": [[x,y], ...],                 // 4 destination points (mm)
  "reprojection_error": 0.12                     // px (informational)
}

If "homography" is absent but "ref_px_points" / "ref_mm_points" are present,
the homography is (re-)computed and the error is validated here.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Reprojection error threshold (px) — reject calibration if exceeded
MAX_REPROJ_ERROR_PX: float = 0.3


@dataclass
class CalibrationData:
    """All calibration artefacts needed by the measurement pipeline."""

    camera_name: str
    camera_matrix: np.ndarray        # shape (3, 3)
    dist_coeffs: np.ndarray          # shape (1, N) or (N,)
    homography: np.ndarray           # shape (3, 3) — pixel → mm (bird's-eye)
    map1: np.ndarray                 # undistortion remap x
    map2: np.ndarray                 # undistortion remap y
    image_size: Tuple[int, int]      # (width_px, height_px)
    reprojection_error: float        # px
    # FOV in mm that the warped output covers
    fov_width_mm: float = 1000.0
    fov_height_mm: float = 300.0
    # Warped output size in pixels
    warp_w_px: int = 2048
    warp_h_px: int = 256

    @property
    def mm_per_px_x(self) -> float:
        """Approximate mm per pixel in the horizontal direction after warp."""
        return self.fov_width_mm / self.warp_w_px

    @property
    def mm_per_px_y(self) -> float:
        return self.fov_height_mm / self.warp_h_px


# ── Internal helpers ────────────────────────────────────────────────────────

def _compute_reprojection_error(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    H: np.ndarray,
) -> float:
    """
    Project src_pts through H and compute mean distance to dst_pts.
    src_pts, dst_pts: (N, 2) float64 arrays.
    """
    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])  # (N, 3)
    proj = (H @ src_h.T).T                                    # (N, 3)
    proj_2d = proj[:, :2] / proj[:, 2:3]                      # (N, 2)
    diffs = proj_2d - dst_pts
    return float(np.mean(np.linalg.norm(diffs, axis=1)))


def _build_remap(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute undistortion maps for fast cv2.remap() calls."""
    w, h = image_size
    new_cam, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=0, newImgSize=(w, h)
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_cam, (w, h), cv2.CV_16SC2
    )
    return map1, map2


def _compute_homography(
    ref_px: np.ndarray,
    ref_mm: np.ndarray,
) -> np.ndarray:
    """Compute homography from pixel reference points to mm reference points."""
    H, mask = cv2.findHomography(ref_px, ref_mm, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("cv2.findHomography returned None — check reference points")
    inliers = int(mask.sum()) if mask is not None else len(ref_px)
    log.debug("Homography computed [inliers=%d/%d]", inliers, len(ref_px))
    return H


# ── Public API ──────────────────────────────────────────────────────────────

def load_calibration(
    calib_path: str,
    camera_name: str,
    warp_w_px: int = 2048,
    warp_h_px: int = 256,
    fov_width_mm: float = 1000.0,
    fov_height_mm: float = 300.0,
) -> CalibrationData:
    """
    Load calibration from JSON file and return a CalibrationData instance.

    Parameters
    ----------
    calib_path : str
        Path to the calibration JSON file.
    camera_name : str
        Human-readable name for logging ("left" / "right").
    warp_w_px, warp_h_px : int
        Desired output size of the warped bird's-eye image.
    fov_width_mm, fov_height_mm : float
        Real-world mm extent covered by the warped output.

    Raises
    ------
    FileNotFoundError
        If the calibration file does not exist.
    ValueError
        If reprojection error exceeds MAX_REPROJ_ERROR_PX.
    """
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(
            f"[{camera_name}] Calibration file not found: {calib_path}\n"
            "Run run_calibration.py first to generate it."
        )

    with open(calib_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
    image_size = (int(data["image_width_px"]), int(data["image_height_px"]))

    # ── Homography ─────────────────────────────────────────────────────────
    ref_px = np.array(data["ref_px_points"], dtype=np.float64)
    ref_mm = np.array(data["ref_mm_points"], dtype=np.float64)

    if "homography" in data:
        H = np.array(data["homography"], dtype=np.float64)
        log.debug("[%s] Loaded pre-computed homography from file.", camera_name)
    else:
        H = _compute_homography(ref_px, ref_mm)
        log.info("[%s] Homography computed from reference points.", camera_name)

    # ── Validate reprojection error ────────────────────────────────────────
    reproj_err = _compute_reprojection_error(ref_px, ref_mm, H)
    log.info(
        "[%s] Reprojection error: %.4f px (limit %.2f px)",
        camera_name, reproj_err, MAX_REPROJ_ERROR_PX,
    )
    if reproj_err > MAX_REPROJ_ERROR_PX:
        raise ValueError(
            f"[{camera_name}] Reprojection error {reproj_err:.4f} px exceeds "
            f"limit {MAX_REPROJ_ERROR_PX} px — recalibrate!"
        )

    # ── Undistortion remap maps ────────────────────────────────────────────
    map1, map2 = _build_remap(camera_matrix, dist_coeffs, image_size)
    log.info("[%s] Undistortion maps precomputed for %dx%d.", camera_name, *image_size)

    return CalibrationData(
        camera_name=camera_name,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        homography=H,
        map1=map1,
        map2=map2,
        image_size=image_size,
        reprojection_error=reproj_err,
        fov_width_mm=fov_width_mm,
        fov_height_mm=fov_height_mm,
        warp_w_px=warp_w_px,
        warp_h_px=warp_h_px,
    )


def make_synthetic_calibration(
    camera_name: str = "synthetic",
    image_width: int = 1920,
    image_height: int = 1080,
    fov_width_mm: float = 1000.0,
    fov_height_mm: float = 300.0,
    warp_w_px: int = 2048,
    warp_h_px: int = 256,
) -> CalibrationData:
    """
    Create a synthetic CalibrationData with identity-like mappings for dry-run
    and unit testing.  Pixels are linearly mapped to mm via a simple scale.
    """
    # Identity camera matrix scaled to image size
    fx = fy = float(image_width)
    cx, cy = image_width / 2.0, image_height / 2.0
    camera_matrix = np.array(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((1, 5), dtype=np.float64)

    # Build homography: map a region of the raw image to mm
    # Source: top-left quadrant of image (strip region)
    # Destination: [0, fov_width_mm] x [0, fov_height_mm]
    strip_y0 = (image_height - fov_height_mm * (image_height / fov_height_mm) * 0.1) / 2
    # Simple 4-corner mapping
    src = np.array([
        [0, 0],
        [image_width, 0],
        [image_width, image_height],
        [0, image_height],
    ], dtype=np.float64)
    dst = np.array([
        [0, 0],
        [fov_width_mm, 0],
        [fov_width_mm, fov_height_mm],
        [0, fov_height_mm],
    ], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst)

    map1, map2 = _build_remap(camera_matrix, dist_coeffs, (image_width, image_height))

    return CalibrationData(
        camera_name=camera_name,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        homography=H,
        map1=map1,
        map2=map2,
        image_size=(image_width, image_height),
        reprojection_error=0.0,
        fov_width_mm=fov_width_mm,
        fov_height_mm=fov_height_mm,
        warp_w_px=warp_w_px,
        warp_h_px=warp_h_px,
    )
