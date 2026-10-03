from fastapi import APIRouter
import threading
from app.services.vision import run_vision_loop, stop_vision_loop

router = APIRouter()

@router.post("/start")
def start_camera():
    thread = threading.Thread(target=run_vision_loop, daemon=True)
    thread.start()
    return {"status": "Camera started", "message": "Vision engine is now monitoring."}

@router.post("/stop")
def stop_camera():
    stop_vision_loop()
    return {"status": "Camera stopped"}