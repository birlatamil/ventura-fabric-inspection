"""
run_calibration.py
==================
Dual-mode calibration utility for the Fabric Width Measurement System.

MODES
-----
1. capture  — Open live camera streams, let you capture frames and save them to
              calibration_images/left/  and  calibration_images/right/.
              Press SPACE to capture a frame (when chessboard is detected),
              press Q to finish and move to the next camera.

2. calibrate — Read all saved images from calibration_images/left/ and
               calibration_images/right/, detect chessboard corners offline,
               run cv2.calibrateCamera(), collect homography reference points,
               and write calibration_data/left_calib.json + right_calib.json.

Usage
-----
  # Step 1 – capture frames for both cameras (uses webcam 0 & 1 by default)
  python run_calibration.py capture

  # Step 1 – capture using RTSP streams
  python run_calibration.py capture --left-src rtsp://... --right-src rtsp://...

  # Step 2 – calibrate from saved images
  python run_calibration.py calibrate

  # Step 2 – only redo homography (reuse existing intrinsics JSON)
  python run_calibration.py calibrate --homography-only

  # Capture or calibrate only one side
  python run_calibration.py capture   --camera left
  python run_calibration.py calibrate --camera right

Checkerboard requirements
-------------------------
  Default: 10x9 inner corners, 25 mm square size.
  Override with --cols, --rows, --square-mm flags.

Output JSON format 
------------------
  {
    "camera_matrix":      [[fx,0,cx],[0,fy,cy],[0,0,1]],
    "dist_coeffs":        [k1,k2,p1,p2,k3],
    "image_width_px":     1920,
    "image_height_px":    1080,
    "homography":         [[...],[...],[...]],
    "ref_px_points":      [[x,y],...],
    "ref_mm_points":      [[x,y],...],
    "reprojection_error": 0.12,
    "rms":                0.43
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("calibration_tool")

# ── Directory layout ─────────────────────────────────────────────────────────
CALIB_DIR   = "calibration_data"
CAPTURE_DIR = "calibration_images"   # sub-dirs: left/, right/

os.makedirs(CALIB_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def open_source(src: str) -> cv2.VideoCapture:
    """Open a camera (int index), RTSP URL, or local video path."""
    try:
        idx = int(src)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {src!r}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 – CAPTURE
# ══════════════════════════════════════════════════════════════════════════════

def capture_and_save_frames(
    cap: cv2.VideoCapture,
    camera_name: str,
    save_dir: str,
    board_cols: int,
    board_rows: int,
    num_frames: int = 30,
    skip_frames: int = 15,
) -> int:
    """
    Live capture: show the camera feed, detect chessboard in real-time,
    and save a frame to *save_dir* when the user presses SPACE (only when
    the chessboard is detected).

    Returns the number of frames saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    board_size = (board_cols, board_rows)

    log.info(
        "[%s] Capturing %d frames to '%s'.  "
        "Press SPACE (chessboard must be detected) to save.  "
        "Press Q to finish early.",
        camera_name, num_frames, save_dir,
    )

    saved      = 0
    frame_idx  = 0
    win_title  = f"Capture — {camera_name}  [0/{num_frames}]"

    while saved < num_frames:
        ret, frame = cap.read()
        if not ret:
            log.error("[%s] Cannot read frame from camera.", camera_name)
            break

        frame_idx += 1
        if frame_idx < skip_frames:
            continue   # let camera auto-expose

        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if found:
            cv2.drawChessboardCorners(display, board_size, corners, found)
            label = f"[{saved}/{num_frames}] FOUND — press SPACE to save"
            color = (0, 220, 0)
        else:
            label = "Chessboard NOT detected"
            color = (0, 0, 220)

        cv2.putText(display, label, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        cv2.imshow(win_title, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and found:
            fname = os.path.join(save_dir, f"{camera_name}_{saved:03d}.png")
            cv2.imwrite(fname, frame)
            saved += 1
            log.info("[%s] Saved %d/%d → %s", camera_name, saved, num_frames, fname)
            time.sleep(0.3)   # brief pause so user can move the board

        elif key == ord("q"):
            log.info("[%s] Stopped early — %d frames saved.", camera_name, saved)
            break

    cv2.destroyAllWindows()
    return saved


def run_capture(args: argparse.Namespace) -> None:
    """Entry-point for `capture` mode."""
    cameras = []
    if args.camera in ("left", "both"):
        cameras.append(("left",  args.left_src))
    if args.camera in ("right", "both"):
        cameras.append(("right", args.right_src))

    for name, src in cameras:
        log.info("=== Capture — %s camera (src=%s) ===", name.upper(), src)
        save_dir = os.path.join(CAPTURE_DIR, name)
        try:
            cap = open_source(src)
            n = capture_and_save_frames(
                cap, name, save_dir,
                board_cols=args.cols,
                board_rows=args.rows,
                num_frames=args.num_frames,
            )
            cap.release()
            log.info("[%s] %d frame(s) saved to '%s'.", name, n, save_dir)
        except Exception as exc:
            log.error("[%s] Capture failed: %s", name, exc, exc_info=True)

    log.info("Capture complete.  Run `python run_calibration.py calibrate` next.")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 – CALIBRATE (offline, from saved images)
# ══════════════════════════════════════════════════════════════════════════════

def load_images_from_folder(folder: str) -> List[np.ndarray]:
    """Load all PNG/JPG images from *folder*, sorted by filename."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(
        p for p in Path(folder).iterdir()
        if p.suffix.lower() in exts
    )
    if not paths:
        raise FileNotFoundError(f"No images found in '{folder}'.")
    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            images.append(img)
        else:
            log.warning("Could not read image: %s — skipped.", p)
    log.info("Loaded %d image(s) from '%s'.", len(images), folder)
    return images


def detect_and_calibrate(
    images: List[np.ndarray],
    board_cols: int,
    board_rows: int,
    square_mm: float,
    camera_name: str,
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    Detect chessboard corners in each saved image and run
    cv2.calibrateCamera().

    Returns (camera_matrix, dist_coeffs, rms, image_size).
    """
    criteria  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    board_size = (board_cols, board_rows)

    # 3-D object points for one board position
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_mm

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    img_size:   Optional[Tuple[int, int]] = None
    good = 0

    for i, frame in enumerate(images):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            log.warning("[%s] Image %d/%d — chessboard NOT detected, skipped.",
                        camera_name, i + 1, len(images))
            continue

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        obj_points.append(objp)
        img_points.append(corners_refined)
        good += 1
        log.info("[%s] Image %d/%d — chessboard detected (%d usable so far).",
                 camera_name, i + 1, len(images), good)

    if good < 10:
        raise RuntimeError(
            f"[{camera_name}] Only {good} usable frames — need ≥10.  "
            "Recapture more images with a clearly visible chessboard."
        )

    log.info("[%s] Running calibrateCamera() on %d frames …", camera_name, good)
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    log.info("[%s] Intrinsic calibration RMS: %.4f px", camera_name, rms)
    if rms > 1.0:
        log.warning("[%s] RMS %.4f px is high — consider recapturing.", camera_name, rms)

    return camera_matrix, dist_coeffs, rms, img_size  # type: ignore[return-value]


# ── Homography helpers (unchanged from original) ─────────────────────────────

def collect_homography_points(
    cap: cv2.VideoCapture,
    camera_name: str,
    num_points: int = 4,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Grab one live frame from *cap* and let the user click reference points.
    After each click, enter the real-world (x, y) in mm at the terminal prompt.
    """
    px_pts:    List[List[float]]    = []
    mm_pts:    List[List[float]]    = []
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
                (px[0] + 8, px[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
            )
            print(f"  Point #{len(px_pts)+1} | pixel=({px[0]}, {px[1]})")
            try:
                raw = input("  Enter real-world mm (x y): ").strip().split()
                mm  = [float(raw[0]), float(raw[1])]
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

    src_h  = np.hstack([src, np.ones((len(src), 1))])
    proj   = (H @ src_h.T).T
    proj2d = proj[:, :2] / proj[:, 2:3]
    error  = float(np.mean(np.linalg.norm(proj2d - dst, axis=1)))

    log.info("[%s] Homography reprojection error: %.4f mm", camera_name, error)
    if error > max_reproj_px:
        log.warning(
            "[%s] Reprojection error %.4f mm exceeds limit %.2f mm — "
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
        "camera_matrix":      camera_matrix.tolist(),
        "dist_coeffs":        dist_coeffs.flatten().tolist(),
        "image_width_px":     image_size[0],
        "image_height_px":    image_size[1],
        "rms":                round(rms, 6),
        "homography":         H.tolist(),
        "ref_px_points":      px_pts,
        "ref_mm_points":      mm_pts,
        "reprojection_error": round(reproj_error, 6),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    log.info("Calibration saved → %s", path)


def run_single_camera_calibration(
    camera_name: str,
    src: str,
    out_path: str,
    board_cols: int,
    board_rows: int,
    square_mm: float,
    homography_only: bool,
) -> None:
    """Calibrate one camera from saved images, then collect homography live."""

    if homography_only:
        # Reuse existing intrinsics
        if not os.path.isfile(out_path):
            raise FileNotFoundError(f"No existing calibration at {out_path!r}")
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
        camera_matrix = np.array(existing["camera_matrix"])
        dist_coeffs   = np.array(existing["dist_coeffs"])
        rms           = existing.get("rms", 0.0)
        image_size    = (existing["image_width_px"], existing["image_height_px"])
        log.info("[%s] Loaded existing intrinsics — skipping offline calibration.", camera_name)
    else:
        # ── Offline calibration from saved images ────────────────────────────
        img_folder = os.path.join(CAPTURE_DIR, camera_name)
        images = load_images_from_folder(img_folder)
        camera_matrix, dist_coeffs, rms, image_size = detect_and_calibrate(
            images, board_cols, board_rows, square_mm, camera_name
        )

    # ── Live homography point collection ─────────────────────────────────────
    cap = open_source(src)
    px_pts, mm_pts = collect_homography_points(cap, camera_name)
    cap.release()

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


def run_calibrate(args: argparse.Namespace) -> None:
    """Entry-point for `calibrate` mode."""
    cameras = []
    if args.camera in ("left", "both"):
        cameras.append(("left",  args.left_src,  os.path.join(CALIB_DIR, "left_calib.json")))
    if args.camera in ("right", "both"):
        cameras.append(("right", args.right_src, os.path.join(CALIB_DIR, "right_calib.json")))

    for name, src, out_path in cameras:
        log.info("=== Calibrating %s camera ===", name.upper())
        run_single_camera_calibration(
            camera_name=name,
            src=src,
            out_path=out_path,
            board_cols=args.cols,
            board_rows=args.rows,
            square_mm=args.square_mm,
            homography_only=args.homography_only,
        )

    log.info("Calibration complete.  Files saved in '%s/'.", CALIB_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Camera Calibration Utility — two-step: capture then calibrate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_calibration.py capture                     # webcams 0 & 1
  python run_calibration.py capture --left-src 0 --right-src 1
  python run_calibration.py capture --left-src rtsp://... --camera left
  python run_calibration.py calibrate
  python run_calibration.py calibrate --homography-only
  python run_calibration.py calibrate --camera right
""",
    )

    sub = p.add_subparsers(dest="mode", required=True)

    # ── shared arguments ──────────────────────────────────────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--left-src", default="0",
        help="Left camera source (int index, RTSP URL, or file path). Default: 0",
    )
    shared.add_argument(
        "--right-src", default="1",
        help="Right camera source. Default: 1",
    )
    shared.add_argument(
        "--cols", type=int, default=10,
        help="Chessboard inner corner columns (default: 10)",
    )
    shared.add_argument(
        "--rows", type=int, default=9,
        help="Chessboard inner corner rows (default: 9)",
    )
    shared.add_argument(
        "--camera", choices=["left", "right", "both"], default="both",
        help="Which camera to process (default: both)",
    )

    # ── capture sub-command ───────────────────────────────────────────────────
    cap_p = sub.add_parser(
        "capture",
        parents=[shared],
        help="Live capture: save frames to calibration_images/left/ and /right/",
    )
    cap_p.add_argument(
        "--num-frames", type=int, default=30,
        help="Number of frames to capture per camera (default: 30)",
    )

    # ── calibrate sub-command ─────────────────────────────────────────────────
    cal_p = sub.add_parser(
        "calibrate",
        parents=[shared],
        help="Offline calibration from saved images + live homography collection",
    )
    cal_p.add_argument(
        "--square-mm", type=float, default=25.0,
        help="Chessboard square size in mm (default: 25.0)",
    )
    cal_p.add_argument(
        "--homography-only", action="store_true",
        help="Skip intrinsic calibration and reuse existing JSON — only redo homography",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        if args.mode == "capture":
            run_capture(args)
        elif args.mode == "calibrate":
            run_calibrate(args)
    except Exception as exc:
        log.error("Operation failed: %s", exc, exc_info=True)
        sys.exit(1)
