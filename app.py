import os
import io
import time
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_PUBLIC = os.path.join(BASE_DIR, "storage", "public")
STORAGE_VAULT = os.path.join(BASE_DIR, "storage", "vault")
AUDIT_LOG_FILE = os.path.join(BASE_DIR, "storage", "audit_trail.jsonl")
MODEL_DIR = os.path.join(BASE_DIR, "models")
YUNET_PATH = os.path.join(MODEL_DIR, "yunet.onnx")

os.makedirs(STORAGE_PUBLIC, exist_ok=True)
os.makedirs(STORAGE_VAULT, exist_ok=True)
os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = FastAPI(
    title="Municipal Operations & Alert Dispatch Platform",
    description="Semi-Automated Hazard Detection (Water Clogging, Tree Debris, Road Potholes, Electricity & Power) & Multi-Department Alert Dispatch Engine",
    version="4.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OFFICIAL_MUNICIPAL_SECRET = "MUNICIPAL_CORP_OFFICER_2026"
PHONE_REGEX = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}')

# ---------------------------------------------------------------------------
# 1. CORE IMPORTANT MUNICIPAL DEPARTMENTS (INCL. ELECTRICITY)
# ---------------------------------------------------------------------------
IMPORTANT_MUNICIPAL_DEPARTMENTS = {
    "DEPT_ELECTRICITY": {
        "id": "DEPT_ELECTRICITY",
        "name": "Electricity & Street Lighting Department",
        "code": "ELEC",
        "in_charge": "Superintending Engineer (Power Distribution & Street Lighting)",
        "handles": "Live snapped wires, sparking transformers, fallen electric poles, electric shock hazards in water, dark streetlights, open junction boxes",
        "alert_channel": "electricity-emergency@municipal.gov.in | Control Room VHF Ch-3 | Emergency Hot Line 1912",
        "sla_hours": {"CRITICAL": 1, "HIGH": 3, "MEDIUM": 6, "LOW": 24},
        "keywords": ["wire", "electric", "current", "pole", "transformer", "spark", "cable", "shock", "streetlight", "power", "blackout", "lamp"]
    },
    "DEPT_DRAINAGE": {
        "id": "DEPT_DRAINAGE",
        "name": "Stormwater Drainage & Flood Mitigation Department",
        "code": "SWD",
        "in_charge": "Executive Engineer (Stormwater Drainage & Flood Control)",
        "handles": "Water clogging, flooded roads, blocked culverts, open storm drain manholes, dewatering pump operations",
        "alert_channel": "drainage-emergency@municipal.gov.in | Control Room VHF Ch-4 | Flood Cell Line 1077",
        "sla_hours": {"CRITICAL": 2, "HIGH": 4, "MEDIUM": 8, "LOW": 24},
        "keywords": ["water", "clogging", "waterlogging", "flood", "drain", "gutter", "manhole", "culvert", "overflow", "stagnant", "submerged"]
    },
    "DEPT_ROADS": {
        "id": "DEPT_ROADS",
        "name": "Roads & Infrastructure Department",
        "code": "ROADS",
        "in_charge": "Executive Engineer (Road Maintenance & Pothole Repair)",
        "handles": "Road potholes, asphalt craters, road cave-ins, broken paver blocks, damaged footpaths, flyover expansion joints",
        "alert_channel": "roads-repair@municipal.gov.in | Control Room VHF Ch-2 | Helpline 1916",
        "sla_hours": {"CRITICAL": 3, "HIGH": 6, "MEDIUM": 12, "LOW": 48},
        "keywords": ["pothole", "crater", "road", "asphalt", "cave-in", "pavement", "footpath", "tar", "divider", "damaged road"]
    },
    "DEPT_HORTICULTURE": {
        "id": "DEPT_HORTICULTURE",
        "name": "Horticulture & Tree Authority",
        "code": "HORT",
        "in_charge": "Superintendent of Gardens & Tree Officer",
        "handles": "Tree debris, fallen trees, uprooted trunks, hanging dangerous branches blocking transit or touching cables",
        "alert_channel": "trees-emergency@municipal.gov.in | Control Room VHF Ch-5 | Hotline 1099",
        "sla_hours": {"CRITICAL": 2, "HIGH": 5, "MEDIUM": 12, "LOW": 36},
        "keywords": ["tree", "debris", "branch", "uprooted", "trunk", "fallen tree", "foliage", "bough", "timber", "garden"]
    },
    "DEPT_EMERGENCY_CELL": {
        "id": "DEPT_EMERGENCY_CELL",
        "name": "Municipal Commissioner's Emergency Cell",
        "code": "COMM-CTRL",
        "in_charge": "Additional Municipal Commissioner (Disaster Operations)",
        "handles": "Boundary road disputes, multi-department deadlocks, severe emergency escalations",
        "alert_channel": "commissioner-control@municipal.gov.in | Emergency Hot Line 101",
        "sla_hours": {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 4, "LOW": 12},
        "keywords": ["deadlock", "boundary dispute", "emergency", "crisis", "evacuation"]
    }
}
LEAN_MUNICIPAL_DEPARTMENTS = IMPORTANT_MUNICIPAL_DEPARTMENTS

TICKETS_DB: Dict[str, dict] = {}
ALERTS_SENT_LOG: List[dict] = []

class SendAlertRequest(BaseModel):
    ticket_id: str
    department_id: str
    admin_id: str = "ADMIN_DISPATCHER_1"
    priority: str = "HIGH"
    admin_notes: Optional[str] = "Immediate field inspection and emergency response mandated."
    override_sla_hours: Optional[int] = None

class ReassignAlertRequest(BaseModel):
    ticket_id: str
    from_department: str
    to_department: str
    reassigned_by: str
    reason: str

# ---------------------------------------------------------------------------
# 2. PRIVACY & COMPUTER VISION PIPELINE (WIDER-FACE REDACTION)
# ---------------------------------------------------------------------------
def strip_exif(pil_img: Image.Image) -> Image.Image:
    clean = Image.new(pil_img.mode, pil_img.size)
    clean.paste(pil_img)
    return clean

def redact_and_sanitize(raw_bytes: bytes) -> tuple:
    pil_raw = Image.open(io.BytesIO(raw_bytes))
    clean_pil = strip_exif(pil_raw)
    cv_img = cv2.cvtColor(np.array(clean_pil), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    targets = []
    # Face detection
    if os.path.exists(YUNET_PATH):
        proc_w = 1280
        scale = proc_w / float(w)
        scaled = cv2.resize(cv_img, (proc_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        det = cv2.FaceDetectorYN.create(YUNET_PATH, '', (proc_w, int(h * scale)), score_threshold=0.30, nms_threshold=0.3)
        _, faces = det.detect(scaled)
        if faces is not None:
            for f in faces:
                bx, by, bw, bh = f[0:4].astype(int)
                conf = float(f[-1])
                targets.append({
                    "type": "Citizen Face",
                    "confidence": round(conf, 3),
                    "box": [int(bx / scale), int(by / scale), int(bw / scale), int(bh / scale)]
                })

    # License plate detection
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array([15, 60, 80]), np.array([38, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 40, 255]))
    mask = cv2.bitwise_or(yellow, white)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    p_cnt = 0
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if 1.5 <= (cw / float(ch if ch > 0 else 1)) <= 4.8 and 400 < (cw * ch) < 40000:
            targets.append({"type": "Vehicle License Plate", "confidence": 0.90, "box": [cx, cy, cw, ch]})
            p_cnt += 1
            if p_cnt >= 5:
                break

    # Apply minimal tight blackbox
    public_cv = cv_img.copy()
    for t in targets:
        bx, by, bw, bh = t["box"]
        cv2.rectangle(public_cv, (bx, by), (bx + bw, by + bh), (0, 0, 0), -1)

    _, pub_bytes = cv2.imencode(".jpg", public_cv, [cv2.IMWRITE_JPEG_QUALITY, 88])
    _, vault_bytes = cv2.imencode(".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return pub_bytes.tobytes(), vault_bytes.tobytes(), targets

# ---------------------------------------------------------------------------
# 3. SEMI-AUTOMATED PROBLEM CLASSIFIER (INCL. ELECTRICITY)
# ---------------------------------------------------------------------------
def classify_municipal_problem(hazard_type: str, description: str) -> dict:
    """
    Detects core municipal complaints:
    1. Electricity & Power / Live Wire Hazards
    2. Water Clogging & Drainage
    3. Tree Debris & Obstructions
    4. Road Potholes & Infrastructure
    """
    text = f"{hazard_type} {description}".lower()
    
    scores = {
        "DEPT_ELECTRICITY": 0,
        "DEPT_DRAINAGE": 0,
        "DEPT_ROADS": 0,
        "DEPT_HORTICULTURE": 0
    }
    
    for dept_id, info in IMPORTANT_MUNICIPAL_DEPARTMENTS.items():
        if dept_id == "DEPT_EMERGENCY_CELL":
            continue
        scores[dept_id] = sum(1 for kw in info["keywords"] if kw in text)

    best_dept = max(scores, key=scores.get)
    max_matches = scores[best_dept]
    
    detected_problem = "Municipal Civic Hazard"
    if best_dept == "DEPT_ELECTRICITY":
        detected_problem = "Live Wire / Electrical Hazard & Street Lighting"
    elif best_dept == "DEPT_DRAINAGE":
        detected_problem = "Water Clogging & Drainage Issue"
    elif best_dept == "DEPT_ROADS":
        detected_problem = "Road Potholes & Infrastructure Damage"
    elif best_dept == "DEPT_HORTICULTURE":
        detected_problem = "Tree Debris & Transit Obstruction"

    conf = min(0.99, max(0.70, 0.60 + (max_matches * 0.15)))
    dept_info = IMPORTANT_MUNICIPAL_DEPARTMENTS[best_dept]
    
    return {
        "detected_problem": detected_problem,
        "recommended_department_id": best_dept,
        "recommended_department_name": dept_info["name"],
        "code": dept_info["code"],
        "in_charge": dept_info["in_charge"],
        "confidence": round(conf, 2)
    }

def calculate_sla(dept_id: str, priority: str, custom_hours: Optional[int] = None) -> dict:
    dept = IMPORTANT_MUNICIPAL_DEPARTMENTS.get(dept_id, IMPORTANT_MUNICIPAL_DEPARTMENTS["DEPT_DRAINAGE"])
    hrs = custom_hours if custom_hours else dept["sla_hours"].get(priority, 6)
    target = datetime.now(timezone.utc) + timedelta(hours=hrs)
    min_hrs = max(1, int(hrs * 0.75))
    max_hrs = int(hrs * 1.5)
    return {
        "sla_hours_limit": hrs,
        "sla_target_utc": target.isoformat(),
        "probabilistic_eta_range": f"{min_hrs} to {max_hrs} hours (subject to weather severity)"
    }

# ---------------------------------------------------------------------------
# 4. API ENDPOINTS: INTAKE, ALERT DISPATCH WITH FULL DETAILS & MONITORING
# ---------------------------------------------------------------------------

@app.post("/api/v1/reports")
async def create_civic_report(
    text: str = Form(...),
    hazard_type: str = Form(...),
    lat: float = Form(...),
    lng: float = Form(...),
    ward: str = Form("Central Ward"),
    is_ward_boundary: bool = Form(False),
    image: UploadFile = File(...)
):
    ticket_id = f"TICK-{uuid.uuid4().hex[:6].upper()}"
    raw_bytes = await image.read()
    
    pub_bytes, vault_bytes, pii_targets = redact_and_sanitize(raw_bytes)
    
    pub_path = os.path.join(STORAGE_PUBLIC, f"{ticket_id}_redacted.jpg")
    vault_path = os.path.join(STORAGE_VAULT, f"{ticket_id}_raw.jpg")
    with open(pub_path, "wb") as f:
        f.write(pub_bytes)
    with open(vault_path, "wb") as f:
        f.write(vault_bytes)

    # Classify problem into important department (including electricity)
    classification = classify_municipal_problem(hazard_type, text)
    
    # Calculate urgency risk score
    risk_score = 75
    if "wire" in text.lower() or "shock" in text.lower() or "spark" in text.lower():
        risk_score = 98  # Electricity is super-critical
    elif "manhole" in text.lower() or "severe" in text.lower():
        risk_score = 94
    elif "crater" in text.lower() or "uprooted" in text.lower():
        risk_score = 88

    priority = "CRITICAL" if risk_score >= 90 else "HIGH"

    ticket = {
        "ticket_id": ticket_id,
        "raw_description": text,
        "sanitized_description": PHONE_REGEX.sub("[REDACTED_PHONE]", text),
        "exact_coordinates": {"latitude": lat, "longitude": lng},
        "coarse_coordinates": {"latitude": round(lat, 3), "longitude": round(lng, 3)},
        "ward": ward,
        "is_ward_boundary": is_ward_boundary,
        "risk_score": risk_score,
        "suggested_priority": priority,
        "status": "PENDING_ADMIN_ALERT",
        "detected_problem": classification["detected_problem"],
        "pre_selected_department_id": classification["recommended_department_id"],
        "pre_selected_department_name": classification["recommended_department_name"],
        "classification_confidence": classification["confidence"],
        "assigned_department": None,
        "alert_dispatched": False,
        "alert_details": None,
        "public_image_url": f"/api/v1/public/media/{ticket_id}_redacted.jpg",
        "official_vault_url": f"/api/v1/official/reports/{ticket_id}/raw-evidence",
        "pii_masked_count": len(pii_targets),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reassignment_hops": [],
        "history": [{"action": "INGESTED", "timestamp": datetime.now(timezone.utc).isoformat()}]
    }
    TICKETS_DB[ticket_id] = ticket

    return {
        "status": "RECEIVED",
        "ticket_id": ticket_id,
        "detected_problem": classification["detected_problem"],
        "recommended_department": classification["recommended_department_name"],
        "risk_score": risk_score,
        "priority": priority,
        "admin_action_required": "Review details and trigger semi-automated alert dispatch"
    }

@app.get("/api/v1/admin/unassigned-tickets")
async def get_admin_inbox():
    """Admin Inbox: Shows complaints waiting for semi-automated Alert dispatch with pre-filled department."""
    pending = [t for t in TICKETS_DB.values() if t.get("status") == "PENDING_ADMIN_ALERT"]
    pending.sort(key=lambda x: x.get("risk_score", 50), reverse=True)
    return {
        "total_pending_alerts": len(pending),
        "tickets": pending
    }

@app.get("/api/v1/admin/departments")
async def get_important_departments():
    """Returns the core important municipal departments (including Electricity)."""
    depts = []
    for d_id, d in IMPORTANT_MUNICIPAL_DEPARTMENTS.items():
        active = sum(1 for t in TICKETS_DB.values() if t.get("assigned_department") == d_id and t.get("status") in ["ALERT_DISPATCHED", "IN_PROGRESS"])
        depts.append({
            "department_id": d["id"],
            "department_name": d["name"],
            "code": d["code"],
            "in_charge": d["in_charge"],
            "handles": d["handles"],
            "alert_channel": d["alert_channel"],
            "active_tickets_count": active,
            "sla_policy": d["sla_hours"]
        })
    return {"departments": depts}

@app.post("/api/v1/admin/send-alert")
async def send_alert_with_all_details(req: SendAlertRequest):
    """
    SEMI-AUTOMATED CORE DISPATCH:
    Sends complete, rich alert dispatch request with ALL operational details
    to the concerned department (including exact coordinates, photos, and SLA clock).
    """
    if req.ticket_id not in TICKETS_DB:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if req.department_id not in IMPORTANT_MUNICIPAL_DEPARTMENTS:
        raise HTTPException(status_code=400, detail="Invalid Department ID")

    ticket = TICKETS_DB[req.ticket_id]
    dept = IMPORTANT_MUNICIPAL_DEPARTMENTS[req.department_id]
    sla = calculate_sla(req.department_id, req.priority, req.override_sla_hours)
    now_str = datetime.now(timezone.utc).isoformat()
    
    # Complete, comprehensive alert dispatch payload
    alert_payload = {
        "alert_id": f"ALT-{uuid.uuid4().hex[:6].upper()}",
        "ticket_id": req.ticket_id,
        "dispatched_at_utc": now_str,
        "dispatched_by_admin": req.admin_id,
        "concerned_department": {
            "department_id": dept["id"],
            "department_name": dept["name"],
            "department_code": dept["code"],
            "in_charge_officer": dept["in_charge"],
            "alert_channel": dept["alert_channel"]
        },
        "incident_details": {
            "detected_problem": ticket["detected_problem"],
            "priority_level": req.priority,
            "risk_score": ticket["risk_score"],
            "ward": ticket["ward"],
            "is_ward_boundary": ticket["is_ward_boundary"],
            "exact_gps_coordinates": ticket["exact_coordinates"],
            "coarse_public_coordinates": ticket["coarse_coordinates"],
            "sanitized_description": ticket["sanitized_description"],
            "raw_caller_description": ticket["raw_description"],
            "admin_dispatch_notes": req.admin_notes
        },
        "evidence_media": {
            "public_redacted_image_url": ticket["public_image_url"],
            "official_raw_vault_url": ticket["official_vault_url"],
            "pii_redacted_count": ticket["pii_masked_count"]
        },
        "sla_enforcement": {
            "allowed_hours_limit": sla["sla_hours_limit"],
            "sla_target_deadline_utc": sla["sla_target_utc"],
            "probabilistic_eta_range": sla["probabilistic_eta_range"]
        }
    }

    ticket["assigned_department"] = req.department_id
    ticket["assigned_department_name"] = dept["name"]
    ticket["priority"] = req.priority
    ticket["status"] = "ALERT_DISPATCHED"
    ticket["alert_dispatched"] = True
    ticket["alert_details"] = alert_payload
    ticket["sla"] = sla
    ticket["history"].append({
        "action": "FULL_ALERT_DISPATCHED",
        "admin_id": req.admin_id,
        "department": dept["name"],
        "priority": req.priority,
        "timestamp": now_str
    })
    
    ALERTS_SENT_LOG.append(alert_payload)

    return {
        "status": "ALERT_SENT_SUCCESSFULLY",
        "ticket_id": req.ticket_id,
        "alert": alert_payload,
        "alert_summary": {
            "alert_id": alert_payload["alert_id"],
            "target_department": dept["name"],
            "target_department_code": dept["code"],
            "in_charge_officer": dept["in_charge"],
            "alert_channel": dept["alert_channel"],
            "detected_problem": ticket["detected_problem"],
            "probabilistic_eta": sla["probabilistic_eta_range"]
        }
    }

@app.post("/api/v1/admin/reassign-alert")
async def reassign_alert_with_anti_loop(req: ReassignAlertRequest):
    if req.ticket_id not in TICKETS_DB:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket = TICKETS_DB[req.ticket_id]
    now_str = datetime.now(timezone.utc).isoformat()
    hops = ticket.get("reassignment_hops", [])
    hops.append({
        "from": req.from_department,
        "to": req.to_department,
        "by": req.reassigned_by,
        "reason": req.reason,
        "timestamp": now_str
    })
    ticket["reassignment_hops"] = hops

    # PING-PONG LOOP INTERCEPTION
    if len(hops) >= 2 and hops[-2]["from"] == req.to_department and hops[-2]["to"] == req.from_department:
        deadlock_dept = "DEPT_EMERGENCY_CELL"
        dept_info = IMPORTANT_MUNICIPAL_DEPARTMENTS[deadlock_dept]
        ticket["assigned_department"] = deadlock_dept
        ticket["assigned_department_name"] = dept_info["name"]
        ticket["status"] = "DEADLOCK_ESCALATED"
        ticket["is_deadlocked"] = True
        
        sla = calculate_sla(deadlock_dept, "CRITICAL", custom_hours=1)
        ticket["sla"] = sla

        ticket["history"].append({
            "action": "PING_PONG_LOOP_INTERCEPTED",
            "alert": "Jurisdictional dispute detected. Both departments declined responsibility.",
            "escalated_to": dept_info["name"],
            "timestamp": now_str
        })

        return {
            "status": "DEADLOCK_TRIGGERED",
            "message": "Ping-pong loop intercepted! Alert auto-escalated directly to Municipal Commissioner.",
            "escalated_owner": dept_info["name"],
            "emergency_sla": sla
        }

    new_dept = IMPORTANT_MUNICIPAL_DEPARTMENTS[req.to_department]
    ticket["assigned_department"] = req.to_department
    ticket["assigned_department_name"] = new_dept["name"]
    ticket["status"] = "ALERT_REASSIGNED"
    sla = calculate_sla(req.to_department, ticket.get("priority", "HIGH"))
    ticket["sla"] = sla

    ticket["history"].append({
        "action": "ALERT_REASSIGNED",
        "from": req.from_department,
        "to": req.to_department,
        "reason": req.reason,
        "timestamp": now_str
    })

    return {"status": "ALERT_REASSIGNED", "new_department": new_dept["name"], "sla": sla}

@app.get("/api/v1/public/reports/{ticket_id}")
async def get_public_ticket(ticket_id: str):
    if ticket_id not in TICKETS_DB:
        raise HTTPException(status_code=404, detail="Ticket not found")
    t = TICKETS_DB[ticket_id]
    return {
        "view_type": "PUBLIC_CITIZEN_VIEW",
        "ticket_id": t["ticket_id"],
        "detected_problem": t["detected_problem"],
        "description": t["sanitized_description"],
        "approximate_location": t["coarse_coordinates"],
        "assigned_department": t.get("assigned_department_name", "Municipal Review In-Progress"),
        "probabilistic_eta": t["sla"]["probabilistic_eta_range"] if t.get("sla") else "Pending Dispatch",
        "public_image_url": t["public_image_url"]
    }

@app.get("/api/v1/public/media/{filename}")
async def get_public_media(filename: str):
    path = os.path.join(STORAGE_PUBLIC, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="image/jpeg")

@app.get("/api/v1/official/reports/{ticket_id}/raw-evidence")
async def get_raw_evidence(ticket_id: str, x_officer_token: Optional[str] = Header(None)):
    if not x_officer_token or x_officer_token != OFFICIAL_MUNICIPAL_SECRET:
        raise HTTPException(status_code=403, detail="Access Denied: Official municipal credentials required.")
    if ticket_id not in TICKETS_DB:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return FileResponse(os.path.join(STORAGE_VAULT, f"{ticket_id}_raw.jpg"), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
