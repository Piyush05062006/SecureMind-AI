import cv2
import collections
import time
import os
from ultralytics import YOLO
import threading
from app.agents.verifier import verify_incident

is_running = False
model = YOLO("yolov8n.pt") 

# Configuration
FPS = 30
BUFFER_SECONDS = 15
BUFFER_LENGTH = FPS * BUFFER_SECONDS
MISSING_THRESHOLD_SECONDS = 5.0 

def save_incident_clip(buffer, item_id, buffer_start_time):
    """Dumps the 15-second deque buffer into an mp4 file using dynamic FPS."""
    if not buffer:
        return None
    os.makedirs("data", exist_ok=True)
    
    timestamp = int(time.time())
    filename = f"data/incident_item_{item_id}_{timestamp}.mp4"
    
    height, width, _ = buffer[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Dynamic FPS calculation to prevent the 15-second video distortion
    time_elapsed = time.time() - buffer_start_time
    actual_fps = len(buffer) / time_elapsed if time_elapsed > 0 else FPS
    
    out = cv2.VideoWriter(filename, fourcc, actual_fps, (width, height)) 
    
    for frame in buffer:
        out.write(frame)
    out.release()
    
    print(f" Evidence saved: {filename} (Recorded at {actual_fps:.1f} FPS)")
    return filename

def stop_vision_loop():
    """Called by /api/camera/stop to terminate the thread safely."""
    global is_running
    is_running = False

def run_vision_loop():
    """The main OpenCV loop running in the background thread."""
    global is_running
    is_running = True
    
    cap = cv2.VideoCapture("http://10.176.46.204:8080/video")
    frame_buffer = collections.deque(maxlen=BUFFER_LENGTH)
    
    active_items = {}
    reported_missing_ids = set() # Prevents duplicate video saving
    
    # Manual Tracker Setup
    manual_trackers = {}
    next_manual_id = 900 
    
    # Time tracking for dynamic FPS
    buffer_start_time = time.time()

    print(" Vision Engine Started. Monitoring feeds...")

    while is_running and cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera. Retrying...")
            time.sleep(1)
            continue

        frame = cv2.resize(frame, (1280, 720))
            
        # Reset buffer start time if it overflows the 15-second window
        if len(frame_buffer) == BUFFER_LENGTH:
            buffer_start_time = time.time() - BUFFER_SECONDS
            
        frame_buffer.append(frame.copy())
        current_time = time.time()
        
        # ==========================================
        # 1. YOLOv8 BYTETRACK (The AI Tracker)
        # ==========================================
        # FIXED: conf=0.75 and iou=0.3 to merge overlapping boxes and stop duplicate IDs
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.35, iou=0.3, verbose=False)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.int().tolist()
            
            for box, tid in zip(boxes, track_ids):
                # Update timestamp
                active_items[tid] = current_time
                # Remove from cooldown if it reappears
                reported_missing_ids.discard(tid) 
                
                # Draw custom GREEN boxes (bypassing results[0].plot)
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"AI ID: {tid}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ==========================================
        # 2. OPENCV KCF (The Manual Tracker)
        # ==========================================
        for m_id in list(manual_trackers.keys()):
            success, m_box = manual_trackers[m_id].update(frame)
            if success:
                # Update timestamp if the KCF tracker successfully finds the object
                active_items[m_id] = current_time
                reported_missing_ids.discard(m_id)
                
                x, y, w, h = [int(v) for v in m_box]
                # Draw custom BLUE boxes for manual items
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Manual ID: {m_id}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
        # ==========================================
        # 3. CHECK FOR MISSING ITEMS (AI & Manual)
        # ==========================================
        for tid, last_seen in list(active_items.items()):
            if (current_time - last_seen) > MISSING_THRESHOLD_SECONDS:
                if tid not in reported_missing_ids:
                    print(f" 🚨 ALERT: Item {tid} has disappeared! Saving video...")
                    video_path = save_incident_clip(frame_buffer, tid, buffer_start_time)
                    
                    # 2. Trigger LangGraph Agent in a Background Thread
                    if video_path:
                        print(f" 🧵 THREAD LAUNCHING: Sending {video_path} to Agent...")
                        agent_thread = threading.Thread(
                            target=verify_incident, 
                            args=(video_path, tid)
                        )
                        agent_thread.start()
                    else:
                        print(" ❌ ERROR: Video failed to save.")
                    
                    # 3. Add to cooldown to prevent duplicate triggers
                    reported_missing_ids.add(tid)
                    
                # Clean up memory
                del active_items[tid]
                if tid in manual_trackers:
                    del manual_trackers[tid]

        # ==========================================
        # 4. KEYBOARD INTERRUPTS
        # ==========================================
        cv2.imshow("Security Feed - Server View", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            is_running = False
            
        elif key == ord("s"):
            print("\n [PAUSED] Multi-Selection Mode.")
            print(" 1. Draw a tight box and press SPACE/ENTER to save it.")
            print(" 2. The box will turn blue. Immediately draw your next box.")
            print(" 3. When finished, DO NOT draw a box. Just press SPACE/ENTER to resume video.")
            
            pause_start = time.time()
            
            while True:
                # Trigger the ROI selector on the current frame
                box = cv2.selectROI("Security Feed - Server View", frame, fromCenter=False, showCrosshair=True)
                
                # If width or height is 0, the user pressed SPACE without drawing a box. Exit the loop.
                if box[2] == 0 or box[3] == 0:
                    break
                    
                # FIXED: Swapped CSRT for KCF to ensure the tracker breaks immediately upon item removal
                try:
                    tracker = cv2.TrackerKCF_create()
                except AttributeError:
                    # Fallback for newer OpenCV versions where legacy trackers are moved
                    tracker = cv2.legacy.TrackerKCF_create()
                    
                tracker.init(frame, box)
                
                manual_trackers[next_manual_id] = tracker
                active_items[next_manual_id] = time.time()
                print(f" ✅ Tracked ID {next_manual_id}")
                next_manual_id += 1
                
                # Permanently paint the box onto the frozen frame so it remains visible
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Manual ID: {next_manual_id-1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Cleanly unpause and forgive the lost time
            print(" ▶️ Resuming live feed...")
            pause_duration = time.time() - pause_start
            
            for item_id in active_items:
                active_items[item_id] += pause_duration
            buffer_start_time += pause_duration

    cap.release()
    cv2.destroyAllWindows()
    print(" Vision Engine Shut Down.")