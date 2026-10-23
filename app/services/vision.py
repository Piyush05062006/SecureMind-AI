import math
import os
import time
import threading
import logging
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

from app.agents.verifier import verify_incident


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================================================
# Configuration
# ==================================================

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")

CAMERA_SOURCE = os.getenv(
    "CAMERA_SOURCE",
    "http://10.176.135.255:8080/video",
)

# Custom ByteTrack settings (longer memory, fewer ghost IDs).
_TRACKER_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bytetrack_custom.yaml",
)
TRACKER_CONFIG = (
    _TRACKER_FILE if os.path.isfile(_TRACKER_FILE) else "bytetrack.yaml"
)

# Rolling buffer used for incident clips.
BUFFER_SECONDS = 15
BUFFER_FPS = 15
BUFFER_MAX_WIDTH = 960  # frames are shrunk before buffering to save RAM

# An object must be missing this long before an incident is created.
MISSING_THRESHOLD_SECONDS = 5

# A YOLO track must be seen continuously this long before it is "armed".
# Short-lived ghost IDs never become incidents.
STABLE_SECONDS = 3.0

# One incident at a time: further disappearances inside this window
# are treated as part of the same event.
INCIDENT_COOLDOWN_SECONDS = 30

# If a vanished YOLO ID overlaps another current box this much,
# it was an ID switch (object still there), not a disappearance.
ID_SWITCH_IOU = 0.3

# Manual tracker settings.
MANUAL_TRACKER_GRACE_FRAMES = 30
MANUAL_MATCH_THRESHOLD = float(os.getenv("MANUAL_MATCH_THRESHOLD", "0.5"))
LOW_TEXTURE_STD = 8.0

# If the loop stalls longer than this (ROI selection, camera outage),
# all timers are shifted so nothing looks "missing" because of the stall.
STALL_SHIFT_SECONDS = 2.0

RECONNECT_AFTER_FAILED_READS = 20


def _parse_watch_classes():
    """
    COCO classes to watch. Override with WATCH_CLASSES in .env:
    comma-separated class IDs, or the word 'all'.
    """
    default = [24, 26, 28, 39, 41, 63, 64, 66, 67, 73]
    # backpack, handbag, suitcase, bottle, cup, laptop, mouse,
    # keyboard, cell phone, book

    raw = os.getenv("WATCH_CLASSES")

    if raw is None or not raw.strip():
        return default

    if raw.strip().lower() == "all":
        return None

    try:
        return [int(part) for part in raw.split(",") if part.strip()]
    except ValueError:
        logger.warning("Invalid WATCH_CLASSES value; using defaults.")
        return default


WATCH_CLASSES = _parse_watch_classes()


# ==================================================
# Shared state
# ==================================================

_stop_event = threading.Event()
_state_lock = threading.Lock()
_running = False

model = YOLO(MODEL_PATH)


def is_vision_running():
    """Return whether the vision loop is running."""
    with _state_lock:
        return _running


def stop_vision_loop():
    """Request the vision loop to stop."""
    _stop_event.set()


# ==================================================
# Helpers
# ==================================================

def get_camera_source():
    """Convert numeric camera indexes to integers."""
    source = CAMERA_SOURCE.strip()

    if source.isdigit():
        return int(source)

    return source


def open_capture():
    """Open the camera. Returns None if it cannot be opened."""
    capture = cv2.VideoCapture(get_camera_source())

    if not capture.isOpened():
        capture.release()
        return None

    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    return capture


def create_tracker():
    """Create a compatible OpenCV object tracker."""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()

    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()

    raise RuntimeError(
        "No compatible OpenCV tracker found. "
        "Install opencv-contrib-python."
    )


def iou(a, b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def crop_region(img, x, y, w, h):
    """Safely crop a region; returns None if it is too small/outside."""
    height, width = img.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)

    if x2 - x1 < 5 or y2 - y1 < 5:
        return None

    return img[y1:y2, x1:x2]


def region_similarity(ref_gray, patch_bgr):
    """
    Compare the current tracked region with the reference snapshot taken
    when the object was selected. Returns roughly 0..1 (1 = same content).
    """
    if patch_bgr is None:
        return 0.0

    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (ref_gray.shape[1], ref_gray.shape[0]))

    # Flat, featureless reference: correlation is unreliable,
    # so compare pixel differences instead.
    if float(ref_gray.std()) < LOW_TEXTURE_STD:
        diff = float(
            np.mean(
                np.abs(
                    gray.astype(np.int16) - ref_gray.astype(np.int16)
                )
            )
        )
        return max(0.0, 1.0 - diff / 40.0)

    score = float(
        cv2.matchTemplate(gray, ref_gray, cv2.TM_CCOEFF_NORMED)[0][0]
    )

    return score if math.isfinite(score) else 0.0


def shrink_for_buffer(img):
    """Copy a frame, shrinking it if it is wider than BUFFER_MAX_WIDTH."""
    height, width = img.shape[:2]

    if width <= BUFFER_MAX_WIDTH:
        return img.copy()

    scale = BUFFER_MAX_WIDTH / width

    return cv2.resize(
        img,
        (BUFFER_MAX_WIDTH, max(1, int(height * scale))),
    )


def save_incident_clip(frames, item_id, fps):
    """Save buffered frames as an MP4 incident recording."""
    if not frames:
        raise ValueError("Cannot save an empty incident video.")

    output_dir = os.path.join("data", "incidents")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"incident_item_{item_id}_{timestamp}.mp4"

    output_path = os.path.abspath(os.path.join(output_dir, filename))

    height, width = frames[0].shape[:2]
    fps = max(1.0, min(float(fps), 30.0))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        writer.release()
        raise RuntimeError("Could not create incident video.")

    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)
    finally:
        writer.release()

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Incident video was not saved correctly.")

    return output_path


# ==================================================
# Main vision loop
# ==================================================

def run_vision_loop():
    """Run YOLO tracking, manual tracking, and incident detection."""

    global _running

    with _state_lock:
        if _running:
            logger.warning("Vision loop is already running.")
            return

        _running = True
        _stop_event.clear()

    capture = None
    window_name = "SecureMind AI - Surveillance"

    # id -> {"first", "last", "armed", "bbox", "label", "manual"}
    tracks = {}

    manual_trackers = {}
    manual_missed_frames = {}
    manual_refs = {}
    last_sim_log = {}
    next_manual_id = 900

    last_any_incident = None

    frame_buffer = deque()
    last_buffer_frame_time = 0.0
    buffer_interval = 1.0 / BUFFER_FPS

    failed_reads = 0

    try:
        # Forget old ByteTrack state from any previous run.
        try:
            model.predictor = None
        except Exception:
            pass

        capture = open_capture()

        if capture is None:
            raise RuntimeError(
                f"Could not open camera source: {CAMERA_SOURCE}. "
                "Check the URL, that the phone and PC are on the same "
                "network, and that the IP Webcam server is running."
            )

        logger.info(
            "Vision loop started using camera source: %s",
            CAMERA_SOURCE,
        )

        logger.info(
            "Tracker config: %s | watched classes: %s",
            TRACKER_CONFIG,
            "all" if WATCH_CLASSES is None else WATCH_CLASSES,
        )

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        last_good_read = time.monotonic()

        while not _stop_event.is_set():

            # ------------------------------------------
            # (Re)connect camera if needed
            # ------------------------------------------

            if capture is None:
                capture = open_capture()

                if capture is None:
                    logger.warning(
                        "Camera reconnect failed; retrying in 2 seconds."
                    )
                    time.sleep(2)
                    continue

                logger.info("Camera reconnected.")

            success, frame = capture.read()

            if not success or frame is None:
                failed_reads += 1

                if failed_reads >= RECONNECT_AFTER_FAILED_READS:
                    logger.warning("Camera stream lost; reconnecting.")
                    capture.release()
                    capture = None
                    failed_reads = 0

                time.sleep(0.2)
                continue

            failed_reads = 0

            now = time.monotonic()

            # ------------------------------------------
            # Stall compensation (ROI selection, outages)
            # ------------------------------------------

            gap = now - last_good_read

            if gap > STALL_SHIFT_SECONDS:
                for t in tracks.values():
                    t["first"] += gap
                    t["last"] += gap

                shifted = [(ts + gap, img) for ts, img in frame_buffer]
                frame_buffer.clear()
                frame_buffer.extend(shifted)

                last_buffer_frame_time += gap

                logger.info(
                    "Loop was paused for %.1f seconds; timers adjusted.",
                    gap,
                )

            last_good_read = now

            # raw: clean frame (no drawings) used for AI, trackers, clips
            # frame: display copy that gets drawn on
            raw = frame
            frame = raw.copy()

            current_boxes = {}

            # ------------------------------------------
            # Rolling frame buffer
            # ------------------------------------------

            if now - last_buffer_frame_time >= buffer_interval:
                frame_buffer.append((now, shrink_for_buffer(raw)))
                last_buffer_frame_time = now

            while frame_buffer and now - frame_buffer[0][0] > BUFFER_SECONDS:
                frame_buffer.popleft()

            # ------------------------------------------
            # YOLOv8 + ByteTrack
            # ------------------------------------------

            results = model.track(
                raw,
                persist=True,
                tracker=TRACKER_CONFIG,
                conf=0.3,
                iou=0.5,
                classes=WATCH_CLASSES,
                verbose=False,
            )

            if (
                results
                and results[0].boxes is not None
                and results[0].boxes.id is not None
            ):
                boxes = results[0].boxes

                ids = boxes.id.int().cpu().tolist()
                coordinates_list = boxes.xyxy.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()

                for track_id, coordinates, class_id in zip(
                    ids,
                    coordinates_list,
                    class_ids,
                ):
                    x1, y1, x2, y2 = coordinates

                    t = tracks.get(track_id)

                    if t is None:
                        t = {
                            "first": now,
                            "armed": False,
                            "manual": False,
                        }
                        tracks[track_id] = t

                    t["last"] = now
                    t["bbox"] = (x1, y1, x2, y2)
                    t["label"] = model.names.get(class_id, "object")

                    if not t["armed"] and now - t["first"] >= STABLE_SECONDS:
                        t["armed"] = True

                    current_boxes[track_id] = (x1, y1, x2, y2)

                    color = (0, 255, 0) if t["armed"] else (160, 160, 160)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(
                        frame,
                        f"{t['label']} #{track_id}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            # ------------------------------------------
            # Manual tracking (OpenCV) with appearance check
            # ------------------------------------------

            for manual_id, tracker in list(manual_trackers.items()):

                t = tracks.get(manual_id)

                if t is None:
                    manual_trackers.pop(manual_id, None)
                    manual_missed_frames.pop(manual_id, None)
                    continue

                tracked, bbox = tracker.update(raw)

                if not tracked:
                    manual_missed_frames[manual_id] = (
                        manual_missed_frames.get(manual_id, 0) + 1
                    )

                    if (
                        manual_missed_frames[manual_id]
                        >= MANUAL_TRACKER_GRACE_FRAMES
                    ):
                        logger.info(
                            "Manual tracker %s lost after %s frames.",
                            manual_id,
                            MANUAL_TRACKER_GRACE_FRAMES,
                        )

                        manual_trackers.pop(manual_id, None)
                        manual_missed_frames.pop(manual_id, None)

                    continue

                manual_missed_frames[manual_id] = 0

                x, y, w, h = [int(v) for v in bbox]

                similarity = region_similarity(
                    manual_refs[manual_id],
                    crop_region(raw, x, y, w, h),
                )

                if now - last_sim_log.get(manual_id, 0.0) >= 2.0:
                    logger.info(
                        "Manual %s similarity %.2f (threshold %.2f)",
                        manual_id,
                        similarity,
                        MANUAL_MATCH_THRESHOLD,
                    )
                    last_sim_log[manual_id] = now

                if similarity >= MANUAL_MATCH_THRESHOLD:
                    # The object is still there.
                    t["last"] = now
                    t["bbox"] = (x, y, x + w, y + h)
                    current_boxes[manual_id] = t["bbox"]

                    color = (255, 180, 0)
                    label = f"Manual ID: {manual_id}"

                else:
                    # Tracker still follows the box, but the content
                    # no longer matches: the object is probably gone.
                    # t["last"] is NOT refreshed, so the missing timer runs.
                    color = (0, 0, 255)
                    label = f"Manual ID: {manual_id} MISSING?"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                cv2.putText(
                    frame,
                    label,
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # ------------------------------------------
            # Detect objects that disappeared
            # ------------------------------------------

            for item_id, t in list(tracks.items()):

                if item_id in current_boxes:
                    continue

                if now - t["last"] < MISSING_THRESHOLD_SECONDS:
                    continue

                # Every ID is judged exactly once.
                tracks.pop(item_id, None)
                manual_trackers.pop(item_id, None)
                manual_missed_frames.pop(item_id, None)
                manual_refs.pop(item_id, None)
                last_sim_log.pop(item_id, None)

                # Short-lived ghost track: ignore.
                if not t["armed"]:
                    logger.info("Ignored unstable track %s.", item_id)
                    continue

                # YOLO ID switch: another box now covers the same spot.
                if not t["manual"] and any(
                    iou(t["bbox"], box) > ID_SWITCH_IOU
                    for box in current_boxes.values()
                ):
                    logger.info(
                        "Track %s looks like an ID switch; ignored.",
                        item_id,
                    )
                    continue

                # Same event already handled.
                if (
                    last_any_incident is not None
                    and now - last_any_incident < INCIDENT_COOLDOWN_SECONDS
                ):
                    logger.info(
                        "Item %s missing, but an incident was created "
                        "recently; skipped.",
                        item_id,
                    )
                    continue

                if not frame_buffer:
                    continue

                last_any_incident = now

                clip_frames = [img.copy() for _, img in frame_buffer]

                span = frame_buffer[-1][0] - frame_buffer[0][0]

                if span > 0 and len(clip_frames) > 1:
                    clip_fps = (len(clip_frames) - 1) / span
                else:
                    clip_fps = float(BUFFER_FPS)

                if span > 0:
                    vanish_fraction = (t["last"] - frame_buffer[0][0]) / span
                    vanish_fraction = min(max(vanish_fraction, 0.0), 1.0)
                else:
                    vanish_fraction = 0.66

                frame_h, frame_w = raw.shape[:2]
                bx1, by1, bx2, by2 = t["bbox"]

                bbox_norm = (
                    min(max(bx1 / frame_w, 0.0), 1.0),
                    min(max(by1 / frame_h, 0.0), 1.0),
                    min(max(bx2 / frame_w, 0.0), 1.0),
                    min(max(by2 / frame_h, 0.0), 1.0),
                )

                try:
                    video_path = save_incident_clip(
                        clip_frames,
                        item_id,
                        clip_fps,
                    )

                    logger.info(
                        "Incident clip created for item %s: %s",
                        item_id,
                        video_path,
                    )

                    threading.Thread(
                        target=verify_incident,
                        args=(
                            video_path,
                            item_id,
                            t["label"],
                            bbox_norm,
                            vanish_fraction,
                        ),
                        daemon=True,
                    ).start()

                except Exception:
                    logger.exception(
                        "Failed to create incident for item %s",
                        item_id,
                    )

            # ------------------------------------------
            # Display
            # ------------------------------------------

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            # ------------------------------------------
            # Manual object selection (press S)
            # ------------------------------------------

            if key == ord("s"):

                roi = cv2.selectROI(
                    window_name,
                    raw,
                    fromCenter=False,
                    showCrosshair=True,
                )

                x, y, w, h = [int(v) for v in roi]

                # Throw away frames that piled up while selecting.
                for _ in range(10):
                    capture.grab()

                if w > 0 and h > 0:
                    try:
                        reference = crop_region(raw, x, y, w, h)

                        if reference is None:
                            raise ValueError("Selected area is too small.")

                        while (
                            next_manual_id in tracks
                            or next_manual_id in manual_trackers
                        ):
                            next_manual_id += 1

                        tracker = create_tracker()
                        tracker.init(raw, (x, y, w, h))

                        selected_at = time.monotonic()

                        manual_trackers[next_manual_id] = tracker
                        manual_missed_frames[next_manual_id] = 0

                        manual_refs[next_manual_id] = cv2.cvtColor(
                            reference,
                            cv2.COLOR_BGR2GRAY,
                        )

                        tracks[next_manual_id] = {
                            "first": selected_at,
                            "last": selected_at,
                            "armed": True,
                            "bbox": (x, y, x + w, y + h),
                            "label": "manually selected object",
                            "manual": True,
                        }

                        logger.info(
                            "Manual tracking started with ID %s",
                            next_manual_id,
                        )

                        next_manual_id += 1

                    except Exception:
                        logger.exception("Could not start manual tracking.")

    except Exception:
        logger.exception("Vision loop failed.")

    finally:
        if capture is not None:
            capture.release()

        cv2.destroyAllWindows()

        with _state_lock:
            _running = False

        logger.info("Vision loop stopped.")