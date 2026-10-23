
import threading

from fastapi import APIRouter

from app.services.vision import run_vision_loop, stop_vision_loop, is_vision_running

router = APIRouter()

_camera_thread = None
_camera_lock = threading.Lock()


@router.post("/start")
def start_camera():
    global _camera_thread

    with _camera_lock:
        if is_vision_running():
            return {
                "status": "already_running",
                "message": "Camera processing is already running."
            }

        if _camera_thread is not None and _camera_thread.is_alive():
            return {
                "status": "already_running",
                "message": "Camera processing is starting or stopping."
            }

        _camera_thread = threading.Thread(
            target=run_vision_loop,
            daemon=True
        )
        _camera_thread.start()

    return {
        "status": "started",
        "message": "Camera processing has been started."
    }


@router.post("/stop")
def stop_camera():
    stop_vision_loop()

    return {
        "status": "stopping",
        "message": "A stop signal has been sent to the camera loop."
    }
