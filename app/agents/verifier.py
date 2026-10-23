import os
import re
import cv2
import json
import time
import base64
import logging
import smtplib
import threading

from typing import TypedDict, List, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from app.db.database import SessionLocal
from app.db.models import Incident


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================================================
# Configuration
# ==================================================

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "").strip()
GEMINI_REQUEST_INTERVAL = 13.0

# Also email when the result is REVIEW (uncertain or Gemini unavailable).
ALERT_ON_REVIEW = os.getenv("ALERT_ON_REVIEW", "true").lower() == "true"

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024

_gemini_lock = threading.Lock()
_last_gemini_request = 0.0


class SecurityState(TypedDict):
    item_id: int
    video_path: str
    label: str
    bbox_norm: Optional[Tuple[float, float, float, float]]
    vanish_fraction: float
    frames_b64: List[str]
    frame_notes: List[str]
    clip_seconds: float
    last_seen_seconds: float
    status: str
    reasoning: str


# ==================================================
# Gemini helpers
# ==================================================

def invoke_gemini_with_rate_limit(llm, message):
    """
    Serialize Gemini calls and space request starts apart.
    Retry sleeping happens outside this lock (see invoke_with_retry).
    """
    global _last_gemini_request

    with _gemini_lock:
        elapsed = time.monotonic() - _last_gemini_request
        wait_time = GEMINI_REQUEST_INTERVAL - elapsed

        if wait_time > 0:
            logger.info(
                "Waiting %.1f seconds before the next Gemini request.",
                wait_time,
            )
            time.sleep(wait_time)

        _last_gemini_request = time.monotonic()

        return llm.invoke(message)


def is_transient_error(exc: Exception) -> bool:
    """Server-side or rate-limit problems worth retrying."""
    text = str(exc).upper()

    return any(
        marker in text
        for marker in (
            "503",
            "500",
            "429",
            "UNAVAILABLE",
            "RESOURCE_EXHAUSTED",
            "INTERNAL",
            "OVERLOADED",
            "DEADLINE",
            "TIMEOUT",
            "TIMED OUT",
        )
    )


def invoke_with_retry(api_key: str, message):
    """
    Try the main model with retries, then the fallback model.
    Raises the last error if everything fails.
    """
    model_names = [GEMINI_MODEL]

    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL:
        model_names.append(GEMINI_FALLBACK_MODEL)

    last_exc = None

    for model_name in model_names:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=api_key,
            max_retries=1,
        )

        for attempt in range(3):
            try:
                return invoke_gemini_with_rate_limit(llm, message)

            except Exception as exc:
                last_exc = exc

                logger.warning(
                    "Gemini model %s attempt %s failed: %s",
                    model_name,
                    attempt + 1,
                    exc,
                )

                # Wrong model name, bad key, etc.: do not keep retrying.
                if not is_transient_error(exc):
                    break

                if attempt < 2:
                    time.sleep(8 * (attempt + 1))

    raise last_exc


def extract_response_text(response) -> str:
    """Handle both string and block-list response formats."""
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, dict) and block.get("text"):
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)

        return "\n".join(text_parts).strip()

    return str(content).strip()


def _try_parse_json(text: str) -> Optional[dict]:
    """Parse a JSON object from plain, fenced, or embedded JSON text."""
    candidates = [text]

    embedded = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if embedded:
        candidates.append(embedded.group(0))

    for candidate in candidates:
        candidate = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            candidate.strip(),
            flags=re.IGNORECASE,
        ).strip()

        try:
            data = json.loads(candidate)

            if isinstance(data, dict):
                return data

        except (json.JSONDecodeError, TypeError):
            continue

    return None


def parse_gemini_response(response_text: str) -> Tuple[str, str]:
    """
    Parse the model response into ALARM, FALSE_ALARM, or REVIEW.
    Anything unclear becomes REVIEW, never ALARM.
    """
    cleaned = response_text.strip()

    data = _try_parse_json(cleaned)

    if data is not None:
        status = str(data.get("status", "REVIEW")).upper().strip()
        reasoning = str(data.get("reasoning", "")).strip()

        if status not in {"ALARM", "FALSE_ALARM", "REVIEW"}:
            status = "REVIEW"

        return status, reasoning or "The model did not provide reasoning."

    # Fallback: accept only an explicit "status: X" style statement.
    status_match = re.search(
        r'status["\']?\s*[:=]\s*["\']?(FALSE_ALARM|ALARM|REVIEW)\b',
        cleaned,
        flags=re.IGNORECASE,
    )

    if not status_match:
        return "REVIEW", cleaned or "Unable to interpret the model response."

    return status_match.group(1).upper(), cleaned


# ==================================================
# Graph nodes
# ==================================================

def extract_keyframes(state: SecurityState) -> dict:
    """
    Extract frames around the moment the object was last seen:
    some from before, some from after. A yellow rectangle marks
    the object's last known location.
    """
    video_path = state["video_path"]
    item_id = state["item_id"]
    bbox_norm = state.get("bbox_norm")

    vanish_fraction = state.get("vanish_fraction")
    if vanish_fraction is None:
        vanish_fraction = 0.66
    vanish_fraction = min(max(float(vanish_fraction), 0.0), 1.0)

    cap = cv2.VideoCapture(video_path)

    frames_b64: List[str] = []
    frame_notes: List[str] = []
    clip_seconds = 0.0
    last_seen_seconds = 0.0

    try:
        if not cap.isOpened():
            logger.error("Could not open incident video: %s", video_path)
            return {"frames_b64": [], "frame_notes": []}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

        if total_frames <= 0:
            logger.warning("Video contains no readable frames: %s", video_path)
            return {"frames_b64": [], "frame_notes": []}

        clip_seconds = total_frames / video_fps
        last_seen_seconds = vanish_fraction * clip_seconds

        f = vanish_fraction

        fractions = [
            0.0,
            f * 0.5,
            max(f - 0.03, 0.0),
            min(f + 0.05, 1.0),
            f + (1.0 - f) * 0.5,
            1.0,
        ]

        last_index = max(total_frames - 1, 0)

        indices = sorted(
            {
                min(last_index, max(0, int(round(fr * last_index))))
                for fr in fractions
            }
        )

        for number, idx in enumerate(indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()

            if not success or frame is None:
                continue

            height, width = frame.shape[:2]

            if width > 640:
                new_height = max(1, int(height * (640 / width)))
                frame = cv2.resize(frame, (640, new_height))
                height, width = frame.shape[:2]

            if bbox_norm:
                x1 = int(bbox_norm[0] * width)
                y1 = int(bbox_norm[1] * height)
                x2 = int(bbox_norm[2] * width)
                y2 = int(bbox_norm[3] * height)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )

            if not success:
                continue

            timestamp = idx / video_fps
            position = (
                "before" if timestamp <= last_seen_seconds + 0.2 else "after"
            )

            frames_b64.append(base64.b64encode(buffer).decode("utf-8"))
            frame_notes.append(
                f"Image {number} (t={timestamp:.1f}s, "
                f"{position} the object was last seen):"
            )

    except Exception:
        logger.exception("Failed to extract keyframes from %s", video_path)

    finally:
        cap.release()

    logger.info(
        "Extracted %s keyframes for item %s.",
        len(frames_b64),
        item_id,
    )

    return {
        "frames_b64": frames_b64,
        "frame_notes": frame_notes,
        "clip_seconds": clip_seconds,
        "last_seen_seconds": last_seen_seconds,
    }


def build_prompt(state: SecurityState) -> str:
    label = state.get("label") or "object"
    item_id = state["item_id"]
    last_seen = state.get("last_seen_seconds", 0.0)
    clip_seconds = state.get("clip_seconds", 0.0)

    return (
        "You are a surveillance incident verification AI.\n\n"
        f"A tracked {label} (ID #{item_id}) stopped being detected. "
        f"It was last seen at about t={last_seen:.1f}s of a "
        f"{clip_seconds:.1f}s clip, inside the yellow rectangle drawn on "
        "each image. The rectangle was added by the system and is not part "
        "of the scene. The images are in time order and each one is "
        "labelled with its timestamp.\n\n"
        "Compare the area inside the yellow rectangle in the images from "
        "before the object was last seen with the images from after it.\n\n"
        "Choose exactly one status:\n"
        "- ALARM: the object is clearly present in the rectangle in the "
        "earlier images and clearly absent in the later images while the "
        "rest of the scene is essentially unchanged, or you can see it "
        "being taken or moved.\n"
        "- FALSE_ALARM: the object is still present in the later images, "
        "or its disappearance is explained by something visible, such as "
        "the camera moving, blur, or a lighting change.\n"
        "- REVIEW: the images are too ambiguous, dark, blurry or "
        "inconsistent to decide.\n\n"
        "Describe only what is visible. Do not accuse anyone or claim "
        "theft. If unsure, choose REVIEW.\n\n"
        "Return ONLY valid JSON in this exact structure:\n"
        '{"status":"ALARM","reasoning":"One or two sentences."}\n'
        "The status must be exactly ALARM, FALSE_ALARM, or REVIEW."
    )


def evaluate_displacement(state: SecurityState) -> dict:
    """Ask Gemini whether the object really disappeared."""
    frames = state.get("frames_b64", [])
    notes = state.get("frame_notes", [])
    item_id = state["item_id"]

    if not frames:
        return {
            "status": "REVIEW",
            "reasoning": "No usable video frames were available for verification.",
        }

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        logger.error("GOOGLE_API_KEY is missing from the environment.")
        return {
            "status": "REVIEW",
            "reasoning": "Gemini API credentials are not configured.",
        }

    content = [{"type": "text", "text": build_prompt(state)}]

    for note, frame_b64 in zip(notes, frames):
        content.append({"type": "text", "text": note})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
            }
        )

    try:
        response = invoke_with_retry(api_key, [HumanMessage(content=content)])
        status, reasoning = parse_gemini_response(
            extract_response_text(response)
        )

    except Exception as exc:
        logger.exception("Gemini unavailable for item %s.", item_id)

        return {
            "status": "REVIEW",
            "reasoning": (
                "Automatic AI verification failed "
                f"({type(exc).__name__}). Please review the clip manually."
            ),
        }

    logger.info(
        "Gemini verification completed for item %s with status %s.",
        item_id,
        status,
    )

    return {"status": status, "reasoning": reasoning}


def save_incident(state: SecurityState) -> None:
    """Save the verification result to the database."""
    item_id = state["item_id"]
    status = state["status"]

    db = SessionLocal()

    try:
        incident_fields = {
            "item_id": item_id,
            "video_path": state["video_path"],
            "is_false_alarm": status == "FALSE_ALARM",
            "agent_reasoning": state["reasoning"][:1000],
        }

        if hasattr(Incident, "status"):
            incident_fields["status"] = status

        db.add(Incident(**incident_fields))
        db.commit()

        logger.info("Saved incident %s with status %s.", item_id, status)

    except Exception:
        db.rollback()
        logger.exception("Failed to save incident %s to the database.", item_id)
        raise

    finally:
        db.close()


def send_email_alert(
    item_id: int,
    video_path: str,
    reasoning: str,
    status: str = "ALARM",
) -> None:
    """Send an email with the incident clip attached when available."""
    sender_email = os.getenv("ALERT_EMAIL_SENDER")
    receiver_email = os.getenv("ALERT_EMAIL_RECEIVER")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([sender_email, receiver_email, smtp_password]):
        logger.error(
            "Email configuration is incomplete. "
            "Set ALERT_EMAIL_SENDER, ALERT_EMAIL_RECEIVER, and SMTP_PASSWORD."
        )
        return

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = f"SECURITY ALERT ({status}): Object #{item_id}"

    attachment_note = ""

    if os.path.isfile(video_path):
        size = os.path.getsize(video_path)

        if size > MAX_ATTACHMENT_BYTES:
            attachment_note = (
                f"\nThe clip is too large to attach ({size // (1024 * 1024)} MB). "
                f"It is saved at: {video_path}"
            )
        else:
            try:
                with open(video_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())

                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f'attachment; filename="{os.path.basename(video_path)}"',
                )
                message.attach(part)

            except Exception:
                logger.exception("Could not attach incident video: %s", video_path)
                attachment_note = f"\nThe clip could not be attached: {video_path}"
    else:
        logger.warning("Incident video not found: %s", video_path)
        attachment_note = "\nThe incident video file was not found."

    body = (
        "A potential security incident was detected.\n\n"
        f"Item ID: {item_id}\n"
        f"Verification status: {status}\n"
        f"AI reasoning: {reasoning}\n"
        f"{attachment_note}\n"
        "Please review the incident video."
    )

    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(sender_email, smtp_password)
            server.send_message(message)

        logger.info("Alert email sent for item %s.", item_id)

    except Exception:
        logger.exception("Failed to send alert email for item %s.", item_id)


def dispatch_result(state: SecurityState) -> dict:
    """Save every result; email on ALARM (and on REVIEW if enabled)."""
    item_id = state["item_id"]
    video_path = state["video_path"]
    status = state["status"]
    reasoning = state["reasoning"]

    if status == "ALARM":
        logger.warning("CONFIRMED INCIDENT | Item %s | %s", item_id, reasoning)
    elif status == "FALSE_ALARM":
        logger.info("FALSE ALARM | Item %s | %s", item_id, reasoning)
    else:
        logger.warning("INCIDENT NEEDS REVIEW | Item %s | %s", item_id, reasoning)

    # A database problem must never block the alert.
    try:
        save_incident(state)
    except Exception:
        logger.error(
            "Incident %s was not stored in the database; continuing to alert.",
            item_id,
        )

    if status == "ALARM" or (status == "REVIEW" and ALERT_ON_REVIEW):
        send_email_alert(item_id, video_path, reasoning, status)

    return {}


# ==================================================
# Graph
# ==================================================

workflow = StateGraph(SecurityState)

workflow.add_node("extract_keyframes", extract_keyframes)
workflow.add_node("evaluate_displacement", evaluate_displacement)
workflow.add_node("dispatch_result", dispatch_result)

workflow.set_entry_point("extract_keyframes")

workflow.add_edge("extract_keyframes", "evaluate_displacement")
workflow.add_edge("evaluate_displacement", "dispatch_result")
workflow.add_edge("dispatch_result", END)

security_agent = workflow.compile()


def verify_incident(
    video_path: str,
    item_id: int,
    label: str = "object",
    bbox_norm: Optional[Tuple[float, float, float, float]] = None,
    vanish_fraction: Optional[float] = None,
) -> None:
    """Run the incident verification workflow."""
    logger.info("Starting LangGraph verification for item %s.", item_id)

    initial_state: SecurityState = {
        "item_id": item_id,
        "video_path": video_path,
        "label": label,
        "bbox_norm": bbox_norm,
        "vanish_fraction": 0.66 if vanish_fraction is None else vanish_fraction,
        "frames_b64": [],
        "frame_notes": [],
        "clip_seconds": 0.0,
        "last_seen_seconds": 0.0,
        "status": "REVIEW",
        "reasoning": "",
    }

    try:
        security_agent.invoke(initial_state)
        logger.info("Verification workflow finished for item %s.", item_id)

    except Exception:
        logger.exception(
            "Verification failed for item %s. "
            "The incident clip has been retained for review.",
            item_id,
        )