"""
FastAPI Microservice for Automated Privacy Redaction (Faces & License Plates)
Provides high-speed REST API endpoints for image redaction, bounding box detection, and EXIF stripping.
"""

import base64
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Query, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redactor import PrivacyRedactor

app = FastAPI(
    title="Privacy Redaction Engine API",
    description="Automated AI engine for identifying and redacting human faces and vehicle license plates.",
    version="1.0.0"
)

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Faces-Redacted", "X-Plates-Redacted", "X-Latency-Ms", "X-Total-Redactions"]
)

# Initialize the Privacy Redactor singleton
redactor = PrivacyRedactor()

@app.get("/")
@app.get("/health")
def health_check():
    """Service status and engine info."""
    return {
        "status": "healthy",
        "service": "Privacy Redaction Engine",
        "version": "1.0.0",
        "model": "YuNet ONNX + Multi-Palette Plate Detector",
        "tiling_enabled": redactor.enable_tiling
    }

@app.post("/v1/redact")
async def redact_image_file(
    file: UploadFile = File(...),
    style: str = Query("blackbox", description="Redaction style: 'blackbox' or 'blur'"),
    quality: int = Query(90, ge=10, le=100, description="Output JPEG quality")
):
    """
    Direct File Redaction Endpoint:
    Accepts an image file upload and returns the sanitized, redacted image directly as JPEG bytes.
    Includes detection metrics in HTTP response headers.
    """
    try:
        raw_bytes = await file.read()
        redacted_bytes, meta = redactor.redact(raw_bytes, style=style, quality=quality)

        headers = {
            "X-Faces-Redacted": str(meta["faces_redacted"]),
            "X-Plates-Redacted": str(meta["plates_redacted"]),
            "X-Total-Redactions": str(meta["total_redactions"]),
            "X-Latency-Ms": str(meta["latency_ms"])
        }

        return Response(content=redacted_bytes, media_type="image/jpeg", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

@app.post("/v1/redact-json")
async def redact_image_json(
    file: UploadFile = File(...),
    style: str = Query("blackbox", description="Redaction style: 'blackbox' or 'blur'"),
    quality: int = Query(90, ge=10, le=100)
):
    """
    JSON Redaction Endpoint:
    Accepts an image file and returns structured JSON with bounding box coordinates,
    confidence metrics, and a base64-encoded redacted image string.
    """
    try:
        raw_bytes = await file.read()
        redacted_bytes, meta = redactor.redact(raw_bytes, style=style, quality=quality)
        b64_str = base64.b64encode(redacted_bytes).decode("utf-8")

        return {
            "status": "success",
            "metadata": meta,
            "redacted_image_base64": f"data:image/jpeg;base64,{b64_str}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing failed: {str(e)}")

@app.post("/v1/detect")
async def detect_pii(file: UploadFile = File(...)):
    """
    Inspection / Detection Endpoint:
    Returns bounding box coordinates, entity types ('Face' or 'License Plate'),
    and confidence scores without modifying the original image.
    """
    try:
        raw_bytes = await file.read()
        cv_img, targets = redactor.detect_all(raw_bytes)
        h, w = cv_img.shape[:2]

        return {
            "status": "success",
            "image_dimensions": {"width": w, "height": h},
            "total_detected": len(targets),
            "faces_count": sum(1 for t in targets if t["type"] == "Face"),
            "plates_count": sum(1 for t in targets if t["type"] == "License Plate"),
            "detections": targets
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Detection failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
