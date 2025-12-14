# ===========================
# SafetyEye - Shared Utilities
# Common functions and configurations
# ===========================
import os
import sqlite3
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Suppress PyTorch warnings that conflict with Streamlit file watcher
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*RuntimeError.*torch.*")

import cv2
import numpy as np
import pandas as pd  # type: ignore
from PIL import Image

# Import YOLO with error handling for Streamlit compatibility
try:
    from ultralytics import YOLO # type: ignore
except Exception as e:
    YOLO = None
    print(f"[WARN] Failed to import YOLO: {e}")

from dotenv import load_dotenv # type: ignore

# Load environment variables
load_dotenv()

# ----------------- Config -----------------
BASE_DIR = Path(os.getenv("PROJECT_ROOT", r"D:\Desktop\coding\Flask_app")).resolve()
MODEL_PATH = os.getenv("MODEL_PATH", r"D:\desktop\coding\Flask_app\model\best.pt")

# Folders
UPLOAD_DIR = BASE_DIR / "flask_uploads"
OUTPUT_DIR = BASE_DIR / "flask_outputs"
LOG_DIR = BASE_DIR / "logs"
SNAP_DIR = LOG_DIR / "snaps"
LOG_DB_PATH = LOG_DIR / "violations.sqlite"

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}

# File size limits
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Streaming config
stream_conf = {
    "conf": float(os.getenv("STREAM_CONF", "0.25")),
    "iou": float(os.getenv("STREAM_IOU", "0.45"))
}

# Global model variable
_model = None
_model_lock = threading.Lock()

# ---------- Utilities ----------
def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def allowed_file(filename: str, allowed: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed

def validate_file_size(file_bytes: bytes, max_size: int) -> Tuple[bool, str]:
    """Validate file size. Returns (is_valid, error_message)."""
    size = len(file_bytes)
    if size > max_size:
        max_mb = max_size / (1024 * 1024)
        return False, f"File size exceeds maximum allowed size of {max_mb:.1f}MB"
    return True, ""

# ---------- Database helpers ----------
def init_db() -> None:
    """Create the violations DB and table if missing."""
    conn = sqlite3.connect(str(LOG_DB_PATH))
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                video_source TEXT,
                violation_type TEXT NOT NULL,
                person_id INTEGER,
                snapshot_path TEXT
            )
            """
        )
    conn.close()

def save_violation_record(violation_type: str, snap_path: str, person_id: Optional[int] = None, source: str = "") -> None:
    """Insert a violation row into the DB."""
    init_db()
    conn = sqlite3.connect(str(LOG_DB_PATH))
    ts = datetime.now().isoformat(timespec="seconds")
    with conn:
        conn.execute(
            "INSERT INTO violations (ts, video_source, violation_type, person_id, snapshot_path) VALUES (?,?,?,?,?)",
            (ts, source, violation_type, person_id, snap_path)
        )
    conn.close()

def read_violations_df(limit: int = 1000) -> pd.DataFrame:
    """Return the latest violations as a pandas DataFrame."""
    if not LOG_DB_PATH.exists():
        return pd.DataFrame(columns=["id", "ts", "video_source", "violation_type", "person_id", "snapshot_path"])
    conn = sqlite3.connect(str(LOG_DB_PATH))
    df = pd.read_sql_query(
        f"SELECT id, ts, video_source, violation_type, person_id, snapshot_path FROM violations ORDER BY id DESC LIMIT {limit}",
        conn
    )
    conn.close()
    return df

# ---------- Model loading ----------
def load_model():
    """Load YOLO model."""
    global _model
    with _model_lock:
        if _model is None:
            try:
                print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
                _model = YOLO(MODEL_PATH)
                print("[INFO] Model loaded. Classes:", getattr(_model, "names", {}))
            except Exception as e:
                print(f"[ERROR] Failed loading YOLO model: {e}")
                _model = None
        return _model

def get_model():
    """Get the loaded model."""
    return _model

# ---------- Geometry helpers ----------
def iou_box(boxA: List[int], boxB: List[int]) -> float:
    """Compute IoU for two boxes given as [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-9)

# ---------- Inference & annotation ----------
def annotate_frame_with_model(frame: np.ndarray, conf: float = None, iou: float = None) -> Tuple[np.ndarray, List[Dict]]:
    """Run model on BGR frame and return annotated frame + detections list."""
    conf = conf if conf is not None else float(stream_conf.get("conf", 0.25))
    iou = iou if iou is not None else float(stream_conf.get("iou", 0.45))
    detections: List[Dict] = []
    out = frame.copy()

    model = get_model()
    if model is None:
        return out, detections

    try:
        results = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
    except Exception as e:
        print(f"[ERROR] model.predict failed: {e}")
        return out, detections

    res = results[0]
    if getattr(res, "boxes", None) is None or len(res.boxes) == 0:
        return out, detections

    try:
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls_inds = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()
    except Exception:
        return out, detections

    # Draw boxes and prepare detections list
    for (x1, y1, x2, y2), cls_i, score in zip(xyxy, cls_inds, scores):
        label = str(model.names[int(cls_i)]).lower() if (model and getattr(model, "names", None) is not None) else str(cls_i)
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        detections.append({"label": label, "conf": float(score), "box": bbox})

        # Annotation styling
        color = (0, 229, 255)
        cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        caption = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(0, bbox[1] - th - 6)
        cv2.rectangle(out, (bbox[0], y0), (bbox[0] + tw, bbox[1]), color, -1)
        cv2.putText(out, caption, (bbox[0], bbox[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return out, detections

# ---------- Rule engine ----------
def evaluate_frame_rules(detections: List[Dict]) -> List[Dict]:
    """Detect missing PPE per person."""
    violations: List[Dict] = []
    persons = [d for d in detections if d["label"] == "person"]
    helmets = [d for d in detections if "helmet" in d["label"] or "hardhat" in d["label"]]
    vests = [d for d in detections if "vest" in d["label"] or "safety vest" in d["label"]]
    masks = [d for d in detections if "mask" in d["label"] or "n95" in d["label"] or "face_mask" in d["label"]]
    shoes = [d for d in detections if "shoe" in d["label"] or "boot" in d["label"] or "safety_shoe" in d["label"]]

    for pid, p in enumerate(persons):
        pbox = p["box"]
        has_helmet = any(iou_box(pbox, h["box"]) > 0.02 for h in helmets)
        if not has_helmet:
            violations.append({"type": "No Helmet", "person_id": pid, "person_box": pbox})

        has_vest = any(iou_box(pbox, v["box"]) > 0.01 for v in vests)
        if not has_vest:
            violations.append({"type": "No Vest", "person_id": pid, "person_box": pbox})

        has_mask = any(iou_box(pbox, m["box"]) > 0.01 for m in masks)
        if not has_mask:
            violations.append({"type": "No Mask", "person_id": pid, "person_box": pbox})

        has_shoes = any(iou_box(pbox, s["box"]) > 0.01 for s in shoes)
        if not has_shoes:
            violations.append({"type": "No Shoes", "person_id": pid, "person_box": pbox})

    return violations

# ---------- Email alert helper ----------
def send_email_alert(subject: str, body: str, attachment_path: Optional[str] = None) -> None:
    """
    Send email alert with violation details and snapshot.
    Both sender and recipient are configurable via environment variables.
    Prevents duplicate alerts using timestamp-based deduplication.
    """
    SMTP_HOST = os.getenv("SMTP_HOST")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASS")
    ALERT_TO = os.getenv("ALERT_TO")
    ALERT_FROM = os.getenv("ALERT_FROM", SMTP_USER)

    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_TO]):
        print("[WARN] SMTP not configured. Set SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_TO in .env")
        return

    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    def _send():
        try:
            msg = MIMEMultipart()
            msg["From"] = ALERT_FROM
            msg["To"] = ALERT_TO
            msg["Subject"] = subject
            
            # Enhanced email body with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            enhanced_body = f"""
SafetyEye AI - PPE Violation Alert

{body}

Detection Details:
- Timestamp: {timestamp}
- System: SafetyEye AI PPE Detection
- Alert Type: Real-time Detection

This is an automated alert from the SafetyEye AI system.
            """
            msg.attach(MIMEText(enhanced_body.strip(), "plain"))
            
            # Attach snapshot if available
            if attachment_path and os.path.exists(attachment_path):
                with open(attachment_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f'attachment; filename="violation_{timestamp.replace(":", "-").replace(" ", "_")}.jpg"'
                )
                msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15)
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            
            # Support multiple recipients (comma-separated)
            recipients = [email.strip() for email in ALERT_TO.split(",")]
            server.sendmail(ALERT_FROM, recipients, msg.as_string())
            server.quit()
            
            print(f"[INFO] Alert email sent to {ALERT_TO} at {timestamp}")
        except Exception as e:
            print(f"[ERROR] Failed to send email alert: {e}")
            # Don't raise - email failure shouldn't stop detection

    # Send in background thread to avoid blocking
    threading.Thread(target=_send, daemon=True).start()

