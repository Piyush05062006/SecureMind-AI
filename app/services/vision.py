import cv2
import collections
import time
import os
from ultralytics import YOLO


is_running = False

model = YOLO("yolov8n.pt") 

# Configuration
FPS = 30
BUFFER_SECONDS = 10
BUFFER_LENGTH = FPS * BUFFER_SECONDS
MISSING_THRESHOLD_SECONDS = 3.0  # Time before item is flagged as missing

def save_incident_clip(buffer, item_id):
    """Dumps the 10-second deque buffer into an mp4 file."""
    if not buffer:
        return None
    
    os.makedirs("data", exist_ok=True)
    
    timestamp = int(time.time())
    filename = f"data/incident_item_{item_id}_{timestamp}.mp4"
    
    height, width, _ = buffer[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, FPS, (width, height)) #out is object for saving the video
    #filename   → Where to save  , fourcc  → How to encode  ,FPS   → How fast to play  ,(width,height)  → Video resolution
    
    for frame in buffer:
        out.write(frame)
    out.release()
    
    print(f" Evidence saved: {filename}")
    return filename

def stop_vision_loop():
    """Called by /api/camera/stop to terminate the thread safely."""
    global is_running
    is_running = False

def run_vision_loop():
    """The main OpenCV loop running in the background thread."""
    global is_running
    is_running = True
    
    cap = cv2.VideoCapture(0)
    frame_buffer = collections.deque(maxlen=BUFFER_LENGTH)
    active_items = {}
    
    print(" Vision Engine Started. Monitoring feeds...")

    while is_running and cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera. Retrying...")
            time.sleep(1)
            continue
            
        #  Add current frame to our 10-second rolling buffer
        frame_buffer.append(frame.copy())
        
        #  Run YOLO Tracking (ByteTrack)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        current_time = time.time()
        
        #  Update Last-Seen Timestamps
        if results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().tolist()
            
            for tid in track_ids:
                active_items[tid] = current_time
                
        # 4. Check for Missing Items
        for tid, last_seen in list(active_items.items()):
            if (current_time - last_seen) > MISSING_THRESHOLD_SECONDS:
                print(f" ALERT: Item {tid} has disappeared!")
                
                video_path = save_incident_clip(frame_buffer, tid)
                
                # TODO: Trigger LangGraph Verification Agent here
                # from app.agents.verifier import verify_incident
                # verify_incident(video_path, tid)
                
                del active_items[tid]

       
        annotated_frame = results[0].plot()
        cv2.imshow("Security Feed - Server View", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            is_running = False

    cap.release()
    cv2.destroyAllWindows()
    print(" Vision Engine Shut Down.")