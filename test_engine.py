"""
Unit Tests for PrivacyRedactor Engine and REST Endpoints
"""

import os
import io
from PIL import Image
from fastapi.testclient import TestClient
from redactor import PrivacyRedactor
from app import app

client = TestClient(app)

def create_synthetic_image(width=800, height=600):
    img = Image.new("RGB", (width, height), color=(180, 200, 220))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_engine_initialization():
    redactor = PrivacyRedactor()
    assert os.path.exists(redactor.model_path)
    assert redactor.face_score_threshold == 0.30

def test_exif_stripping():
    img = Image.new("RGB", (100, 100))
    clean = PrivacyRedactor.strip_exif(img)
    assert clean.size == (100, 100)
    assert not clean.getexif()

def test_redact_execution():
    redactor = PrivacyRedactor()
    raw = create_synthetic_image()
    redacted_bytes, meta = redactor.redact(raw, style="blackbox")
    assert isinstance(redacted_bytes, bytes)
    assert len(redacted_bytes) > 0
    assert "latency_ms" in meta
    assert "total_redactions" in meta

def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_api_redact_file_endpoint():
    raw = create_synthetic_image()
    res = client.post(
        "/v1/redact",
        files={"file": ("test.jpg", raw, "image/jpeg")}
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert "x-faces-redacted" in res.headers

def test_api_detect_endpoint():
    raw = create_synthetic_image()
    res = client.post(
        "/v1/detect",
        files={"file": ("test.jpg", raw, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "total_detected" in data

if __name__ == "__main__":
    print("Running engine unit tests...")
    test_engine_initialization()
    test_exif_stripping()
    test_redact_execution()
    test_api_health()
    test_api_redact_file_endpoint()
    test_api_detect_endpoint()
    print("All unit tests passed successfully!")
