"""
debug_view.py
=============
Real-time visual debugger for the fabric width measurement system.

Opens an OpenCV window showing:
  - The warped (bird's-eye) image from either camera
  - A horizontal yellow line at the auto-detected laser row
  - Vertical GREEN lines at the left and right detected fabric edges
  - Text overlay: pixel columns, mm values, width, confidence

Usage
-----
  # Left camera (default)
  python debug_view.py

  # Right camera
  python debug_view.py --camera right

  # Custom RTSP source
  python debug_view.py --src rtsp://172.32.0.93:554/live/0

  # Use Sobel mode instead of laser
  python debug_view.py --mode sobel

Press  Q  or  Esc  to quit.
"""

from __future__ import annotations
import argparse
import sys
import time

import cv2
import numpy as np

from config import get_config, MeasurementConfig
from calibration import load_calibration, make_synthetic_calibration

# ── helpers ─────────────────────────────────────────────────────────────────

def _grab_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = cap.read()
    return frame if ok else None


def _detect_laser(gray: np.ndarray, cfg: MeasurementConfig):
    """
    Returns (laser_row, l_col, r_col, avg_row, confidence) or None.
    Same logic as measurement._measure_laser().
    """
    h, w = gray.shape
    row_means = gray.mean(axis=1)
    laser_row = int(np.argmax(row_means))

    r0 = max(0, laser_row - cfg.laser_row_band)
    r1 = min(h, laser_row + cfg.laser_row_band + 1)
    avg_row = gray[r0:r1].astype(np.float32).mean(axis=0)

    mask = avg_row >= cfg.laser_brightness_threshold
    bright_cols = np.where(mask)[0]

    if bright_cols.size == 0:
        return None, laser_row, avg_row

    l_col = int(bright_cols[0])
    r_col = int(bright_cols[-1])
    brightness = float(avg_row[bright_cols].mean())
    confidence = min(brightness / 255.0, 1.0)
    return (laser_row, l_col, r_col, confidence), laser_row, avg_row


def _detect_sobel(gray: np.ndarray, cfg: MeasurementConfig):
    """Returns (laser_row≈midpoint, l_col, r_col, confidence) or None."""
    h, w = gray.shape
    cy = h // 2
    half = cfg.strip_height_px // 2
    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    strip = gray[y0:y1]

    sobel = cv2.Sobel(strip, cv2.CV_64F, 1, 0, ksize=cfg.sobel_ksize)
    abs_sobel = np.abs(sobel)

    left_cols, right_cols, confs = [], [], []
    for row in abs_sobel:
        if row.max() < cfg.sobel_threshold:
            continue
        noise = float(np.median(row)) + 1e-6
        mid = w // 2
        lh, rh = row[:mid], row[mid:]
        if lh.size == 0 or rh.size == 0:
            continue
        li = int(np.argmax(lh))
        ri = int(np.argmax(rh)) + mid
        if lh[li] < cfg.sobel_threshold or rh[ri - mid] < cfg.sobel_threshold:
            continue
        snr_l = min(lh[li] / noise / 10.0, 1.0)
        snr_r = min(rh[ri - mid] / noise / 10.0, 1.0)
        left_cols.append(li)
        right_cols.append(ri)
        confs.append(float(np.sqrt(snr_l * snr_r)))

    if not left_cols:
        return None, cy, None

    l_col = int(np.mean(left_cols))
    r_col = int(np.mean(right_cols))
    conf  = float(np.mean(confs))
    return (cy, l_col, r_col, conf), cy, None


def _draw_overlay(
    display: np.ndarray,
    laser_result,
    laser_row: int,
    mm_per_px: float,
    mode: str,
    avg_row: np.ndarray | None,
    cfg: MeasurementConfig,
) -> np.ndarray:
    """Draw all annotations onto display (a BGR copy of the warped frame)."""
    h, w = display.shape[:2]

    # ── Yellow horizontal line at laser row ─────────────────────────────────
    cv2.line(display, (0, laser_row), (w - 1, laser_row), (0, 220, 220), 1)
    cv2.putText(
        display, f"laser row={laser_row}",
        (4, laser_row - 6 if laser_row > 20 else laser_row + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1, cv2.LINE_AA,
    )

    if laser_result is not None:
        _, l_col, r_col, confidence = laser_result
        width_px = r_col - l_col
        mm_per_px_val = mm_per_px
        left_mm  = l_col * mm_per_px_val
        right_mm = r_col * mm_per_px_val
        width_mm = right_mm - left_mm

        # ── Left edge — bright green ─────────────────────────────────────
        cv2.line(display, (l_col, 0), (l_col, h - 1), (0, 255, 0), 2)
        cv2.putText(
            display, f"L {l_col}px\n{left_mm:.1f}mm",
            (max(0, l_col - 60), h - 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA,
        )

        # ── Right edge — bright green ────────────────────────────────────
        cv2.line(display, (r_col, 0), (r_col, h - 1), (0, 255, 0), 2)
        cv2.putText(
            display, f"R {r_col}px\n{right_mm:.1f}mm",
            (min(w - 120, r_col + 4), h - 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA,
        )

        # ── Width span brackets ──────────────────────────────────────────
        cy_vis = laser_row
        cv2.arrowedLine(display, (l_col, cy_vis), (r_col, cy_vis), (0, 200, 255), 2, tipLength=0.02)
        cv2.arrowedLine(display, (r_col, cy_vis), (l_col, cy_vis), (0, 200, 255), 2, tipLength=0.02)

        # ── Top-left info box ────────────────────────────────────────────
        lines = [
            f"MODE: {mode.upper()}",
            f"WIDTH: {width_mm:.2f} mm  ({width_px} px)",
            f"LEFT:  {left_mm:.2f} mm  (col {l_col})",
            f"RIGHT: {right_mm:.2f} mm  (col {r_col})",
            f"CONF:  {confidence:.3f}",
        ]
        box_x, box_y = 8, 8
        lh_px = 22
        box_w = 320
        box_h = len(lines) * lh_px + 12
        overlay = display.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)
        for i, txt in enumerate(lines):
            color = (100, 255, 100) if i == 1 else (220, 220, 220)
            cv2.putText(
                display, txt,
                (box_x + 6, box_y + lh_px * (i + 1)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA,
            )
    else:
        # No detection — show warning
        cv2.putText(
            display, "NO EDGE DETECTED",
            (w // 2 - 110, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA,
        )
        if avg_row is not None:
            peak = float(avg_row.max())
            cv2.putText(
                display,
                f"Peak brightness: {peak:.0f}  threshold: {cfg.laser_brightness_threshold}",
                (w // 2 - 180, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 1, cv2.LINE_AA,
            )

    # ── Bottom timestamp ─────────────────────────────────────────────────
    ts_str = time.strftime("%H:%M:%S")
    cv2.putText(
        display, ts_str,
        (w - 80, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA,
    )

    return display


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time fabric edge debug viewer")
    parser.add_argument("--camera", choices=["left", "right"], default="left")
    parser.add_argument("--src", default=None, help="Override RTSP URL")
    parser.add_argument("--mode", choices=["laser", "sobel"], default="laser")
    parser.add_argument("--scale", type=float, default=0.6,
                        help="Display scale factor (default 0.6)")
    args = parser.parse_args()

    cfg = get_config()
    m_cfg = cfg.measurement

    # ── Select camera URL ────────────────────────────────────────────────────
    if args.src:
        rtsp_url = args.src
        cam_name = args.camera
    else:
        cam_cfg = cfg.left_camera if args.camera == "left" else cfg.right_camera
        rtsp_url = cam_cfg.rtsp_url
        cam_name = cam_cfg.name

    # ── Load calibration ─────────────────────────────────────────────────────
    try:
        from calibration import load_calibration
        calib_cfg = cfg.calibration
        calib_path = calib_cfg.left_path if args.camera == "left" else calib_cfg.right_path
        fov_mm = m_cfg.left_fov_mm_width if args.camera == "left" else m_cfg.right_fov_mm_width
        calib = load_calibration(
            calib_path=calib_path,
            camera_name=cam_name,
            warp_w_px=m_cfg.warp_output_width_px,
            warp_h_px=m_cfg.warp_output_height_px,
            fov_width_mm=fov_mm,
        )
        print(f"[DEBUG] Calibration loaded: {calib_path}")
    except Exception as e:
        print(f"[WARN] Could not load calibration ({e}), using synthetic.")
        from calibration import make_synthetic_calibration
        calib = make_synthetic_calibration(
            cam_name,
            fov_width_mm=m_cfg.left_fov_mm_width if args.camera == "left" else m_cfg.right_fov_mm_width,
            fov_height_mm=300.0,
            warp_w_px=m_cfg.warp_output_width_px,
            warp_h_px=m_cfg.warp_output_height_px,
        )

    warp_size = (calib.warp_w_px, calib.warp_h_px)
    mm_per_px = calib.fov_width_mm / calib.warp_w_px

    # ── Open camera ──────────────────────────────────────────────────────────
    print(f"[DEBUG] Connecting to: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("ERROR: Could not open camera stream.")
        return 1

    print("[DEBUG] Stream opened. Press Q or Esc to quit.")
    win_name = f"Edge Debug — {cam_name} [{args.mode.upper()}]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    fps_t = time.monotonic()
    fps_count = 0
    fps_display = 0.0

    while True:
        frame = _grab_frame(cap)
        if frame is None:
            print("[WARN] No frame — retrying …")
            time.sleep(0.05)
            continue

        # ── Pre-crop raw frame ───────────────────────────────────────────
        if m_cfg.raw_strip_height_px > 0:
            src_h = frame.shape[0]
            cy_raw = src_h // 2
            half_raw = m_cfg.raw_strip_height_px // 2
            frame = frame[max(0, cy_raw - half_raw): min(src_h, cy_raw + half_raw)]

        # ── Undistort + warp ─────────────────────────────────────────────
        undistorted = cv2.remap(frame, calib.map1, calib.map2, cv2.INTER_LINEAR)
        warped = cv2.warpPerspective(undistorted, calib.homography, warp_size, flags=cv2.INTER_LINEAR)

        # ── Grayscale ────────────────────────────────────────────────────
        if warped.ndim == 3:
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            gray = warped

        # Make a colour display copy (laser line is brighter in pseudo-colour)
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # ── Detect ──────────────────────────────────────────────────────
        if args.mode == "laser":
            laser_result, laser_row, avg_row = _detect_laser(gray, m_cfg)
        else:
            laser_result, laser_row, avg_row = _detect_sobel(gray, m_cfg)

        # ── Annotate ─────────────────────────────────────────────────────
        display = _draw_overlay(display, laser_result, laser_row, mm_per_px, args.mode, avg_row, m_cfg)

        # ── FPS counter ──────────────────────────────────────────────────
        fps_count += 1
        now = time.monotonic()
        if now - fps_t >= 1.0:
            fps_display = fps_count / (now - fps_t)
            fps_count = 0
            fps_t = now
        cv2.putText(
            display, f"{fps_display:.1f} fps",
            (8, display.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA,
        )

        # ── Show ─────────────────────────────────────────────────────────
        if args.scale != 1.0:
            h_d, w_d = display.shape[:2]
            display = cv2.resize(display, (int(w_d * args.scale), int(h_d * args.scale)))
        cv2.imshow(win_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # Q or Esc
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
