"""
run_calibration.py
==================
Standalone calibration capture utility for the Fabric Width Measurement System.

What it does
------------
1. Connect to each RTSP camera (or use a local video file / webcam)
2. Capture checkerboard frames for intrinsic calibration
3. Run cv2.calibrateCamera() per camera
4. Interactively collect 4 real-world reference points for homography
5. Save calibration_data/left_calib.json and right_calib.json

Usage
-----
  # Full calibration of both cameras (interactive)
  python run_calibration.py

  # Calibrate with a local video or image folder
  python run_calibration.py --left-src 0 --right-src 1   # webcams

  # Only load existing calib and recompute homography
  python run_calibration.py --homography-only

Checkerboard requirements
-------------------------
  Default: 9x6 inner corners, 25 mm square size.
  Override with --cols, --rows, --square-mm flags.

Output JSON format
------------------
  {
    "camera_matrix":   [[fx,0,cx],[0,fy,cy],[0,0,1]],
    "dist_coeffs":     [k1,k2,p1,p2,k3],
    "image_width_px":  1920,
    "image_height_px": 1080,
    "homography":      [[...],[...],[...]],
    "ref_px_points":   [[x,y],...],
    "ref_mm_points":   [[x,y],...],
    "reprojection_error": 0.12,
    "rms":             0.43
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("calibration_tool")

CALIB_DIR = "calibration_data"
os.makedirs(CALIB_DIR, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def open_source(src: str) -> cv2.VideoCapture:
    """Open a camera (int index), RTSP URL, or local video path."""
    try:
        idx = int(src)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {src}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def capture_checkerboard_frames(
    cap: cv2.VideoCapture,
    camera_name: str,
    board_cols: int,
    board_rows: int,
    num_frames: int = 30,
    skip_frames: int = 15,
) -> List[np.ndarray]:
    """
    Interactively capture checkerboard frames.

    Press SPACE to capture, Q to finish early.
    Returns list of captured frames containing detected checkerboards.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    board_size = (board_cols, board_rows)
    frames: List[np.ndarray] = []

    log.info(
        "[%s] Capturing %d checkerboard frames (%dx%d inner corners). "
        "Press SPACE to capture, Q to finish.",
        camera_name, num_frames, board_cols, board_rows,
    )

    frame_count = 0
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            log.error("[%s] Cannot read frame.", camera_name)
            break

        frame_count += 1
        if frame_count < skip_frames:
            continue  # let camera auto-expose

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if found:
            cv2.drawChessboardCorners(display, board_size, corners, found)
            cv2.putText(
                display,
                f"[{len(frames)+1}/{num_frames}] FOUND — SPACE to capture",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
        else:
            cv2.putText(
                display,
                "Checkerboard NOT found",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )

        cv2.imshow(f"Calibration — {camera_name}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and found:
            frames.append(frame)
            log.info("[%s] Captured frame %d/%d", camera_name, len(frames), num_frames)
            time.sleep(0.3)  # brief pause to allow physical movement of board
        elif key == ord("q"):
            log.info("[%s] User stopped capture early (%d frames).", camera_name, len(frames))
            break

    cv2.destroyAllWindows()
    return frames


def calibrate_intrinsics(
    frames: List[np.ndarray],
    board_cols: int,
    board_rows: int,
    square_mm: float,
    camera_name: str,
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    Compute intrinsic parameters from checkerboard frames.

    Returns (camera_matrix, dist_coeffs, rms_error, image_size).
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    board_size = (board_cols, board_rows)

    # 3D object points for one board position
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_mm

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    img_size: Optional[Tuple[int, int]] = None

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            continue

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        obj_points.append(objp)
        img_points.append(corners_refined)

    if len(obj_points) < 10:
        raise RuntimeError(
            f"[{camera_name}] Only {len(obj_points)} usable frames — need ≥10. "
            "Recapture with clearer checkerboard images."
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    log.info("[%s] Intrinsic calibration RMS: %.4f px", camera_name, rms)
    if rms > 1.0:
        log.warning("[%s] RMS %.4f px is high — consider recapturing.", camera_name, rms)

    return camera_matrix, dist_coeffs, rms, img_size  # type: ignore[return-value]


def collect_homography_points(
    cap: cv2.VideoCapture,
    camera_name: str,
    num_points: int = 4,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Interactively collect pixel ↔ real-world mm point pairs for homography.

    Click on the live frame to mark pixel points.  After each click, enter
    the corresponding real-world (x, y) in mm at the terminal prompt.
    """
    px_pts: List[List[float]] = []
    mm_pts: List[List[float]] = []
    click_buf: List[Tuple[int, int]] = []

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_buf.append((x, y))

    win = f"Homography — {camera_name} (click {num_points} reference points)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_cb)

    log.info(
        "[%s] Click %d reference points in the live frame.\n"
        "After each click, enter the real-world X Y in mm at the prompt.",
        camera_name, num_points,
    )

    ret, snapshot = cap.read()
    if not ret:
        raise RuntimeError(f"[{camera_name}] Cannot grab snapshot for homography.")

    display = snapshot.copy()

    while len(px_pts) < num_points:
        cv2.imshow(win, display)
        cv2.waitKey(1)

        if click_buf:
            px = click_buf.pop(0)
            cv2.circle(display, px, 6, (0, 255, 0), -1)
            cv2.putText(
                display,
                f"#{len(px_pts)+1}: ({px[0]},{px[1]})",
                (px[0]+8, px[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
            )
            print(f"  Point #{len(px_pts)+1} | pixel=({px[0]}, {px[1]})")
            try:
                raw = input("  Enter real-world mm (x y): ").strip().split()
                mm = [float(raw[0]), float(raw[1])]
            except (ValueError, IndexError):
                log.error("Invalid input — enter two numbers separated by space.")
                continue

            px_pts.append([float(px[0]), float(px[1])])
            mm_pts.append(mm)
            log.info("  Pair %d: px=%s mm=%s", len(px_pts), px, mm)

    cv2.destroyWindow(win)
    return px_pts, mm_pts


def compute_and_validate_homography(
    px_pts: List[List[float]],
    mm_pts: List[List[float]],
    camera_name: str,
    max_reproj_px: float = 0.3,
) -> Tuple[np.ndarray, float]:
    """Compute homography and validate reprojection error."""
    src = np.array(px_pts, dtype=np.float64)
    dst = np.array(mm_pts, dtype=np.float64)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError(f"[{camera_name}] Homography computation failed.")

    # Reprojection error
    src_h = np.hstack([src, np.ones((len(src), 1))])
    proj = (H @ src_h.T).T
    proj_2d = proj[:, :2] / proj[:, 2:3]
    error = float(np.mean(np.linalg.norm(proj_2d - dst, axis=1)))

    log.info("[%s] Homography reprojection error: %.4f px", camera_name, error)
    if error > max_reproj_px:
        log.warning(
            "[%s] Reprojection error %.4f px exceeds limit %.2f px — "
            "consider collecting more accurate reference points.",
            camera_name, error, max_reproj_px,
        )

    return H, error


def save_calibration(
    path: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rms: float,
    image_size: Tuple[int, int],
    H: np.ndarray,
    px_pts: List[List[float]],
    mm_pts: List[List[float]],
    reproj_error: float,
) -> None:
    data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "image_width_px": image_size[0],
        "image_height_px": image_size[1],
        "rms": round(rms, 6),
        "homography": H.tolist(),
        "ref_px_points": px_pts,
        "ref_mm_points": mm_pts,
        "reprojection_error": round(reproj_error, 6),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    log.info("Calibration saved → %s", path)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_camera_calibration(
    src: str,
    camera_name: str,
    out_path: str,
    board_cols: int,
    board_rows: int,
    square_mm: float,
    num_frames: int,
    homography_only: bool,
) -> None:
    cap = open_source(src)

    if homography_only:
        # Load existing intrinsics
        if not os.path.isfile(out_path):
            raise FileNotFoundError(f"No existing calibration at {out_path}")
        with open(out_path) as f:
            existing = json.load(f)
        camera_matrix = np.array(existing["camera_matrix"])
        dist_coeffs = np.array(existing["dist_coeffs"])
        rms = existing.get("rms", 0.0)
        image_size = (existing["image_width_px"], existing["image_height_px"])
        log.info("[%s] Loaded existing intrinsics — skipping checkerboard capture.", camera_name)
    else:
        # Full intrinsic calibration
        frames = capture_checkerboard_frames(
            cap, camera_name, board_cols, board_rows, num_frames
        )
        if not frames:
            raise RuntimeError(f"[{camera_name}] No checkerboard frames captured.")
        camera_matrix, dist_coeffs, rms, image_size = calibrate_intrinsics(
            frames, board_cols, board_rows, square_mm, camera_name
        )

    # Homography reference points
    px_pts, mm_pts = collect_homography_points(cap, camera_name)
    H, reproj_error = compute_and_validate_homography(px_pts, mm_pts, camera_name)

    save_calibration(
        path=out_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms=rms,
        image_size=image_size,
        H=H,
        px_pts=px_pts,
        mm_pts=mm_pts,
        reproj_error=reproj_error,
    )
    cap.release()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Camera Calibration Utility")
    p.add_argument("--left-src",  default="rtsp://admin:password@192.168.1.100:554/stream1",
                   help="Left camera source (RTSP URL, int index, or filename)")
    p.add_argument("--right-src", default="rtsp://admin:password@192.168.1.101:554/stream1",
                   help="Right camera source")
    p.add_argument("--cols", type=int, default=9, help="Checkerboard inner corner columns")
    p.add_argument("--rows", type=int, default=6, help="Checkerboard inner corner rows")
    p.add_argument("--square-mm", type=float, default=25.0, help="Checkerboard square size (mm)")
    p.add_argument("--num-frames", type=int, default=30, help="Number of checkerboard frames")
    p.add_argument("--homography-only", action="store_true",
                   help="Skip intrinsic calibration, only redo homography")
    p.add_argument("--camera", choices=["left", "right", "both"], default="both",
                   help="Which camera to calibrate")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        if args.camera in ("left", "both"):
            log.info("=== Calibrating LEFT camera ===")
            run_camera_calibration(
                src=args.left_src,
                camera_name="left",
                out_path=os.path.join(CALIB_DIR, "left_calib.json"),
                board_cols=args.cols,
                board_rows=args.rows,
                square_mm=args.square_mm,
                num_frames=args.num_frames,
                homography_only=args.homography_only,
            )
        if args.camera in ("right", "both"):
            log.info("=== Calibrating RIGHT camera ===")
            run_camera_calibration(
                src=args.right_src,
                camera_name="right",
                out_path=os.path.join(CALIB_DIR, "right_calib.json"),
                board_cols=args.cols,
                board_rows=args.rows,
                square_mm=args.square_mm,
                num_frames=args.num_frames,
                homography_only=args.homography_only,
            )
        log.info("Calibration complete. Files saved in '%s/'", CALIB_DIR)
    except Exception as exc:
        log.error("Calibration failed: %s", exc, exc_info=True)
        sys.exit(1)
