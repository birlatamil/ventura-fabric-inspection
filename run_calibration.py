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
import concurrent.futures
import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("calibration_tool")

# ── GPU / OpenCL availability ────────────────────────────────────────────────
# cv2.UMat transparently offloads cvtColor, CLAHE, resize etc. to GPU via
# OpenCL when available — no special build required.
try:
    _test = cv2.UMat(np.zeros((2, 2), dtype=np.uint8))
    _USE_UMAT = True
    log.info("OpenCL/UMat acceleration available — GPU will be used.")
except Exception:
    _USE_UMAT = False
    log.info("OpenCL/UMat NOT available — falling back to CPU.")

# Force RTSP over TCP globally — eliminates UDP packet-loss and H264 decode
# errors.  We deliberately do NOT set fflags;nobuffer or flags;low_delay here
# because those cause H264 inter-frame reference errors when P-frames are
# skipped.  Latency is managed instead by keeping cap buffer=1 and draining
# stale frames inside the reader thread.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp",
)

# ── Directory layout ─────────────────────────────────────────────────────────
CALIB_DIR   = "calibration_data"
CAPTURE_DIR = "calibration_images"   # sub-dirs: left/, right/

os.makedirs(CALIB_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _robust_find_chessboard(
    img: np.ndarray,
    board_size: Tuple[int, int],
    *,
    use_downscale_precheck: bool = True,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Attempt to find chessboard corners with multiple pre-processing steps.
    Tries gray, CLAHE, and adaptive thresholding to maximize robustness.

    Speed optimisations
    -------------------
    * **Downscaled pre-check** — run FAST_CHECK on a half-resolution copy
      first; if no board is found there, skip the expensive full-res search.
    * **UMat (OpenCL GPU)** — cvtColor and CLAHE are offloaded to the GPU
      when available, freeing the CPU for other threads.
    """
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    # ── Downscaled pre-check (fast reject) ───────────────────────────────
    if use_downscale_precheck:
        h, w = img.shape[:2]
        small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        pre_found, _ = cv2.findChessboardCorners(small_gray, board_size, flags)
        if not pre_found:
            return False, None  # fast reject — no board visible

    # ── Full-resolution detection (UMat accelerated when possible) ───────
    if _USE_UMAT:
        u_img = cv2.UMat(img)
        u_gray = cv2.cvtColor(u_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.UMat.get(u_gray)  # download for findChessboardCorners
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Normal detection
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if found:
        return True, corners

    # 2. Try with CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if _USE_UMAT:
        clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) \
            if hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'createCLAHE') \
            else cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        try:
            gray_clahe = clahe.apply(u_gray)
            gray_clahe = cv2.UMat.get(gray_clahe)
        except Exception:
            clahe_cpu = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_clahe = clahe_cpu.apply(gray)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

    found, corners = cv2.findChessboardCorners(gray_clahe, board_size, flags)
    if found:
        return True, corners

    # 3. No board found with any preprocessing
    return False, None

def open_source(src: str) -> cv2.VideoCapture:
    """Open a camera (int index), RTSP URL, or local video path.

    For RTSP URLs, forces TCP transport via CAP_FFMPEG so that the
    OPENCV_FFMPEG_CAPTURE_OPTIONS env-var (set at the top of this file)
    is honoured, eliminating packet-loss and H264 decode / lag issues.
    """
    try:
        idx = int(src)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        # Use CAP_FFMPEG so the OPENCV_FFMPEG_CAPTURE_OPTIONS env var
        # defined above (tcp + nobuffer + low_delay) is applied.
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {src!r}")

    # Minimise internal buffer — keep only the latest frame
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Request hardware-accelerated decode if available
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

    # Drain a small number of frames that piled up during connection setup
    for _ in range(4):
        cap.grab()

    log.info(
        "Opened source '%s' — %dx%d @ %.1f fps",
        src,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        cap.get(cv2.CAP_PROP_FPS),
    )
    return cap


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 – PARALLEL CAPTURE  (both cameras live simultaneously)
# ══════════════════════════════════════════════════════════════════════════════

class _CameraWorker:
    """
    Per-camera worker: two threads, zero GUI calls.

    Thread 1 — reader: grab + retrieve frames from the RTSP stream at full
    camera rate.  Uses grab-then-retrieve (not cap.read) so we can drain one
    extra stale frame per cycle without breaking the H264 reference chain.

    Thread 2 — detector: runs cv2.findChessboardCorners on the most recent
    frame so the main/GUI thread never blocks on slow corner detection.

    The main thread picks up (display_frame, found, corners) via .get_display()
    and handles all imshow() / waitKey() / save logic.
    """

    def __init__(
        self,
        name: str,
        cap: cv2.VideoCapture,
        save_dir: str,
        board_size: Tuple[int, int],
        num_frames: int,
    ) -> None:
        self.name       = name
        self.save_dir   = save_dir
        self.board_size = board_size
        self.num_frames = num_frames
        self.saved      = 0
        self.done       = False   # set True when this camera is finished

        self._cap       = cap
        self._stop      = threading.Event()

        # Latest raw frame shared between reader → detector → main
        self._raw_lock  = threading.Lock()
        self._raw_frame: Optional[np.ndarray] = None
        self._raw_seq   = 0   # incremented every time a new raw frame arrives

        # Latest detection result shared between detector → main
        self._det_lock    = threading.Lock()
        self._det_display: Optional[np.ndarray] = None
        self._det_found   = False
        self._det_corners: Optional[np.ndarray] = None
        self._det_raw_ref: Optional[np.ndarray] = None  # raw frame that was detected

        # Save request: main → worker
        self._save_queue: queue.Queue = queue.Queue(maxsize=4)
        self._save_homog_ref = False  # True if we want to save next frame as homog ref

        os.makedirs(save_dir, exist_ok=True)

    # ── public (called from main thread) ─────────────────────────────────────

    def start(self) -> None:
        self._t_reader = threading.Thread(
            target=self._reader_loop, name=f"reader-{self.name}", daemon=True)
        self._t_detect = threading.Thread(
            target=self._detect_loop, name=f"detect-{self.name}", daemon=True)
        self._t_saver  = threading.Thread(
            target=self._saver_loop, name=f"saver-{self.name}",  daemon=True)
        self._t_reader.start()
        self._t_detect.start()
        self._t_saver.start()

    def stop(self) -> None:
        self._stop.set()
        self._save_queue.put(None)   # unblock saver
        self._t_reader.join(timeout=3)
        self._t_detect.join(timeout=3)
        self._t_saver.join(timeout=3)
        self._cap.release()

    def get_display(self) -> Tuple[Optional[np.ndarray], bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (display_frame, found, corners, raw_frame) — all may be None."""
        with self._det_lock:
            return (
                self._det_display,
                self._det_found,
                self._det_corners,
                self._det_raw_ref,
            )

    def request_save(self, raw_frame: np.ndarray, is_homog_ref: bool = False) -> None:
        """Ask the saver thread to write *raw_frame* to disk."""
        try:
            self._save_queue.put_nowait((raw_frame, is_homog_ref))
        except queue.Full:
            pass  # previous save still in progress — skip

    # ── reader thread ─────────────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        """
        Grab + retrieve at full camera speed.
        Uses grab() first so we can call grab() once more to skip one
        buffered frame and stay 1 frame ahead — without breaking H264 chains.
        """
        warm = 0
        while not self._stop.is_set():
            # Single extra grab to discard one buffered frame
            self._cap.grab()
            ok = self._cap.grab()
            if not ok:
                time.sleep(0.02)
                continue
            ret, frame = self._cap.retrieve()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            # Warm-up: let auto-exposure settle
            warm += 1
            if warm < 20:
                continue

            with self._raw_lock:
                self._raw_frame = frame
                self._raw_seq  += 1

    # ── detector thread ───────────────────────────────────────────────────────

    def _detect_loop(self) -> None:
        """
        Run chessboard detection on the latest raw frame.
        Runs at ~15 fps (67 ms sleep between detections) to avoid saturating
        a CPU core — detection is slow (~30–100 ms per frame).
        """
        last_seq = -1
        while not self._stop.is_set():
            with self._raw_lock:
                seq   = self._raw_seq
                frame = self._raw_frame

            if frame is None or seq == last_seq:
                time.sleep(0.02)
                continue
            last_seq = seq

            # Use downscaled pre-check for fast rejection in live mode
            found, corners = _robust_find_chessboard(
                frame, self.board_size, use_downscale_precheck=True,
            )

            # Avoid a full .copy() when not found — only copy when we overlay
            if found:
                display = frame.copy()
                cv2.drawChessboardCorners(display, self.board_size, corners, found)
                label = (
                    f"[{self.saved}/{self.num_frames}] "
                    f"FOUND — SPACE: save | H: homog ref"
                )
                color = (0, 220, 0)
            else:
                display = frame  # no overlay needed, skip expensive copy
                label = "Chessboard NOT detected"
                color = (0, 0, 220)

            cv2.putText(
                display, label, (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
            )
            cv2.putText(
                display,
                f"Press Q to finish {self.name} camera early",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
            )

            with self._det_lock:
                self._det_display  = display
                self._det_found    = found
                self._det_corners  = corners if found else None
                self._det_raw_ref  = frame

            # Reduced sleep: GPU + downscale pre-check are fast enough
            time.sleep(0.030)

    # ── saver thread ──────────────────────────────────────────────────────────

    def _saver_loop(self) -> None:
        """Write frames to disk in the background without blocking the GUI."""
        while True:
            item = self._save_queue.get()
            if item is None:
                break
            
            raw_frame, is_homog_ref = item
            
            if is_homog_ref:
                fname = os.path.join(self.save_dir, "homography_ref.png")
                cv2.imwrite(fname, raw_frame)
                log.info("[%s] Saved homography reference image → %s", self.name, fname)
            else:
                fname = os.path.join(
                    self.save_dir, f"{self.name}_{self.saved:03d}.png"
                )
                cv2.imwrite(fname, raw_frame)
                self.saved += 1
                log.info(
                    "[%s] Saved %d/%d → %s",
                    self.name, self.saved, self.num_frames, fname,
                )
                if self.saved >= self.num_frames:
                    self.done = True


def run_capture(args: argparse.Namespace) -> None:
    """
    Entry-point for `capture` mode.

    Both cameras start simultaneously.  SPACE saves a frame from the camera
    whose window is currently in focus.  Q marks that camera as done.
    The session ends once every requested camera has saved enough frames.
    """
    cam_specs: List[Tuple[str, str]] = []
    if args.camera in ("left", "both"):
        cam_specs.append(("left",  args.left_src))
    if args.camera in ("right", "both"):
        cam_specs.append(("right", args.right_src))

    board_size = (args.cols, args.rows)

    # Open captures and create workers
    workers: List[_CameraWorker] = []
    for name, src in cam_specs:
        log.info("=== Opening %s camera (src=%s) ===", name.upper(), src)
        try:
            cap = open_source(src)
        except Exception as exc:
            log.error("[%s] Cannot open camera: %s", name, exc)
            continue
        save_dir = os.path.join(CAPTURE_DIR, name)
        w = _CameraWorker(name, cap, save_dir, board_size, args.num_frames)
        workers.append(w)

    if not workers:
        log.error("No cameras could be opened.  Aborting.")
        return

    # Start all workers in parallel
    for w in workers:
        w.start()
        log.info(
            "[%s] capturing %d frames to '%s'  "
            "| SPACE = save  | Q = finish this camera early",
            w.name, w.num_frames, w.save_dir,
        )

    # ── Main / GUI loop (must run on main thread on Windows) ─────────────────
    # Track which camera window is "focused" (last key press).
    # Both windows are always shown; SPACE/Q apply to the focused one.
    focused: str = workers[0].name   # default focus
    manually_done: set = set()       # cameras the user pressed Q on

    WIN = {w.name: f"Capture — {w.name.upper()}" for w in workers}
    for w in workers:
        cv2.namedWindow(WIN[w.name], cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN[w.name], 960, 540)

    PLACEHOLDER = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(
        PLACEHOLDER, "Connecting …",
        (340, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2,
    )

    while True:
        # Check if all cameras are finished
        all_done = all(
            w.done or w.name in manually_done
            for w in workers
        )
        if all_done:
            break

        # Refresh each window
        for w in workers:
            if w.name in manually_done or w.done:
                continue
            display, found, corners, raw = w.get_display()
            frame_to_show = display if display is not None else PLACEHOLDER
            cv2.imshow(WIN[w.name], frame_to_show)

        # Single waitKey services all windows
        key = cv2.waitKey(1) & 0xFF

        # Detect which window is focused by checking which title was last clicked
        # OpenCV doesn't expose focus directly, so we use a simple heuristic:
        # iterate workers and check if their window exists — the first active one
        # with a fresh frame that received the last keypress gets the action.
        # For simplicity: if only one camera left, it owns the key.
        active = [w for w in workers if w.name not in manually_done and not w.done]
        if len(active) == 1:
            focused = active[0].name

        if key == ord("q") or key == ord("Q"):
            # Q finishes the focused camera
            target = next((w for w in workers if w.name == focused), None)
            if target:
                log.info(
                    "[%s] Stopped early — %d frame(s) saved.",
                    target.name, target.saved,
                )
                manually_done.add(target.name)

        elif key == ord(" "):
            # SPACE saves a frame from the focused camera if chessboard found
            target = next((w for w in workers if w.name == focused), None)
            if target and not target.done and target.name not in manually_done:
                _, found, corners, raw = target.get_display()
                if found and raw is not None:
                    target.request_save(raw.copy())
                else:
                    log.warning("[%s] Space pressed but chessboard not detected.",
                                target.name)

        elif key == ord("h") or key == ord("H"):
            # H saves a homography reference image
            target = next((w for w in workers if w.name == focused), None)
            if target and not target.done and target.name not in manually_done:
                _, found, corners, raw = target.get_display()
                if found and raw is not None:
                    target.request_save(raw.copy(), is_homog_ref=True)
                else:
                    log.warning("[%s] 'H' pressed but chessboard not detected.",
                                target.name)

        elif key in (ord("1"), ord("l")):
            focused = "left"
        elif key in (ord("2"), ord("r")):
            focused = "right"

    cv2.destroyAllWindows()

    # Stop all workers
    for w in workers:
        w.stop()
        log.info("[%s] %d frame(s) saved to '%s'.", w.name, w.saved, w.save_dir)

    log.info(
        "Capture complete.  "
        "Run `python run_calibration.py calibrate` next."
    )


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

    img_size: Optional[Tuple[int, int]] = None
    if images:
        img_size = (images[0].shape[1], images[0].shape[0])

    # ── Parallel chessboard detection ────────────────────────────────────
    # Detect corners across all images concurrently using a thread pool.
    # findChessboardCorners releases the GIL, so threads give a real
    # speedup on multi-core CPUs (and UMat/OpenCL parallelism).
    num_workers = min(len(images), os.cpu_count() or 4)
    log.info(
        "[%s] Detecting chessboard in %d images with %d parallel workers …",
        camera_name, len(images), num_workers,
    )

    def _detect_one(idx_frame):
        idx, frame = idx_frame
        found, corners = _robust_find_chessboard(
            frame, board_size, use_downscale_precheck=True,
        )
        return idx, found, corners, frame

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = pool.map(_detect_one, enumerate(images))
        results = list(futures)

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    good = 0

    for idx, found, corners, frame in results:
        if not found:
            log.warning("[%s] Image %d/%d — chessboard NOT detected, skipped.",
                        camera_name, idx + 1, len(images))
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        obj_points.append(objp)
        img_points.append(corners_refined)
        good += 1
        log.info("[%s] Image %d/%d — chessboard detected (%d usable so far).",
                 camera_name, idx + 1, len(images), good)

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
    source: cv2.VideoCapture | np.ndarray,
    camera_name: str,
    num_points: int = 4,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Grab reference points from either a live VideoCapture OR a static image.
    After each click, enter the real-world (x, y) in mm at the terminal prompt.
    """
    px_pts:    List[List[float]]    = []
    mm_pts:    List[List[float]]    = []
    click_buf: List[Tuple[int, int]] = []

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_buf.append((x, y))

    source_type = "Image" if isinstance(source, np.ndarray) else "Live"
    win = f"Homography ({source_type}) — {camera_name} (click {num_points} points)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_cb)

    if isinstance(source, np.ndarray):
        snapshot = source.copy()
    else:
        # Drain buffer and grab one live frame
        for _ in range(5):
            source.grab()
        ret, snapshot = source.read()
        if not ret:
            raise RuntimeError(f"[{camera_name}] Cannot grab snapshot for homography.")

    display = snapshot.copy()

    log.info(
        "[%s] %s mode: Click %d reference points.\n"
        "After each click, enter the real-world X Y in mm at the prompt.",
        camera_name, source_type, num_points,
    )

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
    img_folder = os.path.join(CAPTURE_DIR, camera_name)
    homography_ref_path = os.path.join(img_folder, "homography_ref.png")
    
    if os.path.exists(homography_ref_path):
        log.info("[%s] Found homography reference image: %s", camera_name, homography_ref_path)
        source = cv2.imread(homography_ref_path)
        if source is None:
             log.error("[%s] Could not read %s, falling back to live.", camera_name, homography_ref_path)
             cap = open_source(src)
             px_pts, mm_pts = collect_homography_points(cap, camera_name)
             cap.release()
        else:
             px_pts, mm_pts = collect_homography_points(source, camera_name)
    else:
        log.info("[%s] No homography reference image found. Opening live feed.", camera_name)
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
