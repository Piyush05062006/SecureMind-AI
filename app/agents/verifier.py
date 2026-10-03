import cv2
import base64
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from app.db.database import SessionLocal
from app.db.models import Incident


class SecurityState(TypedDict):
    item_id: int
    video_path: str
    frames_b64: List[str]
    is_false_alarm: bool
    reasoning: str

def extract_keyframes(state: SecurityState) -> dict:
    video_path = state["video_path"]
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        return {"frames_b64": []}

    indices = [0, total_frames // 4, total_frames // 2, (3 * total_frames) // 4, total_frames - 1] #Keyframes (Start, 25%, 50%, 75%, End)
    frames_b64 = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            # Resize frame down to 640px to reduce token consumption and speed up inference
            h, w = frame.shape[:2]
            scaled_frame = cv2.resize(frame, (640, int(h * (640 / w))))
            _, buffer = cv2.imencode(".jpg", scaled_frame)
            b64_str = base64.b64encode(buffer).decode("utf-8")
            frames_b64.append(b64_str)

    cap.release()
    return {"frames_b64": frames_b64}


def evaluate_displacement(state: SecurityState) -> dict:
    frames = state.get("frames_b64", [])
    item_id = state["item_id"]

    if not frames:
        return {"is_false_alarm": True, "reasoning": "Failed to extract video frames."}

    llm = ChatGroq(
        model="llama-3.2-90b-vision-preview", 
        temperature=0
    )

    # Build multimodal content payload
    content = [
        {
            "type": "text",
            "text": (
                f"You are a surveillance verification AI. Tracked object ID #{item_id} was reported missing. "
                "Attached are 5 sequential keyframes from the last 10 seconds of surveillance feed.\n\n"
                "Analyze the scene:\n"
                "1. Is the object genuinely removed/stolen/displaced?\n"
                "2. Or is this a false alarm (e.g., someone temporarily walked in front of it, "
                "camera glare, shadows, or the object is still clearly there)?\n\n"
                "Respond in this EXACT format:\n"
                "STATUS: [ALARM or FALSE_ALARM]\n"
                "REASONING: [1-2 concise sentences explaining why]"
            )
        }
    ]

    for b64 in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    response = llm.invoke([HumanMessage(content=content)])
    res_text = response.content

    is_false_alarm = "STATUS: FALSE_ALARM" in res_text
    reasoning = res_text.split("REASONING:")[-1].strip() if "REASONING:" in res_text else res_text
    return {"is_false_alarm": is_false_alarm, "reasoning": reasoning}


def decide_route(state: SecurityState) -> str:
    return "handle_false_alarm" if state["is_false_alarm"] else "trigger_verified_alarm"

#  Handle Verified Alarms (Log to MySQL + Send Alert)
def trigger_verified_alarm(state: SecurityState) -> dict:
    item_id = state["item_id"]
    video_path = state["video_path"]
    reasoning = state["reasoning"]

    print(f"\n [CONFIRMED INCIDENT] Item #{item_id} is missing!")
    print(f"Reasoning: {reasoning}")

    db = SessionLocal()
    try:
        incident = Incident(
            item_id=item_id,
            video_path=video_path,
            is_false_alarm=False,
            agent_reasoning=reasoning
        )
        db.add(incident)
        db.commit()
    finally:
        db.close()

    send_email_alert(item_id, video_path, reasoning)

    return {}


def handle_false_alarm(state: SecurityState) -> dict:
    print(f"\n [FALSE ALARM SUPPRESSED] Item #{state['item_id']} was just obscured.")
    print(f"Reasoning: {state['reasoning']}")

    # Optional: Log false alarm in MySQL for audit records
    db = SessionLocal()
    try:
        incident = Incident(
            item_id=state["item_id"],
            video_path=state["video_path"],
            is_false_alarm=True,
            agent_reasoning=state["reasoning"]
        )
        db.add(incident)
        db.commit()
    finally:
        db.close()

    return {}

def send_email_alert(item_id: int, video_path: str, reasoning: str):
    """Utility to send an incident notification via  Simple Mail Transfer Protocol."""
    sender_email = os.getenv("ALERT_EMAIL_SENDER", "security@example.com")
    receiver_email = os.getenv("ALERT_EMAIL_RECEIVER", "admin@example.com")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com") #Which mail server should I connect to to send the email? 
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_password = os.getenv("SMTP_PASSWORD", "")

    if not smtp_password:
        print("⚠️ SMTP credentials not set. Skipping email dispatch.")
        return

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = f"CRITICAL ALERT: Object #{item_id} Missing"

    body = f"Incident Detected!\n\nItem ID: {item_id}\nEvidence Clip: {video_path}\nAI Analysis: {reasoning}"
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, smtp_password)
            server.send_message(msg)
            print(" Alert email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

#  Compile the LangGraph
workflow = StateGraph(SecurityState)
workflow.add_node("extract_keyframes", extract_keyframes)
workflow.add_node("evaluate_displacement", evaluate_displacement)
workflow.add_node("trigger_verified_alarm", trigger_verified_alarm)
workflow.add_node("handle_false_alarm", handle_false_alarm)

workflow.set_entry_point("extract_keyframes")
workflow.add_edge("extract_keyframes", "evaluate_displacement")
workflow.add_conditional_edges(
    "evaluate_displacement",
    decide_route,
    {
        "trigger_verified_alarm": "trigger_verified_alarm",
        "handle_false_alarm": "handle_false_alarm"
    }
)
workflow.add_edge("trigger_verified_alarm", END)
workflow.add_edge("handle_false_alarm", END)

security_agent = workflow.compile()

def verify_incident(video_path: str, item_id: int):
    """Entry point called by the Vision Service."""
    initial_state = {
        "item_id": item_id,
        "video_path": video_path,
        "frames_b64": [],
        "is_false_alarm": False,
        "reasoning": ""
    }
    security_agent.invoke(initial_state)