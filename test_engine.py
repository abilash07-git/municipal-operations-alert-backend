"""
Unit Tests for PrivacyRedactor Engine and REST Endpoints
Validates face landmark geometry verification, license plate precision,
and privacy sanitization against false positives.
"""

import os
import io
import numpy as np
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
    assert redactor.face_score_threshold == 0.48

def test_exif_stripping():
    img = Image.new("RGB", (100, 100))
    clean = PrivacyRedactor.strip_exif(img)
    assert clean.size == (100, 100)
    assert not clean.getexif()

def test_landmark_geometric_validation():
    # Valid face landmark configuration
    valid_face = np.array([
        100, 100, 50, 60,      # bx, by, bw, bh
        115, 120,               # right eye (x, y)
        135, 120,               # left eye (x, y)
        125, 135,               # nose tip (x, y)
        118, 148,               # right mouth (x, y)
        132, 148,               # left mouth (x, y)
        0.85                    # score
    ], dtype=np.float32)
    assert PrivacyRedactor.validate_face_landmarks(valid_face) is True

    # Upside-down / distorted false positive (eyes below mouth)
    inverted_fake = np.array([
        100, 100, 50, 60,
        115, 150,               # eyes below mouth
        135, 150,
        125, 135,
        118, 110,
        132, 110,
        0.42
    ], dtype=np.float32)
    assert PrivacyRedactor.validate_face_landmarks(inverted_fake) is False

    # Extreme non-human aspect ratio
    collapsed_fake = np.array([
        100, 100, 120, 20,      # AR = 6.0
        115, 110,
        185, 110,
        150, 115,
        120, 118,
        180, 118,
        0.55
    ], dtype=np.float32)
    assert PrivacyRedactor.validate_face_landmarks(collapsed_fake) is False

def test_plate_color_filtering():
    # Pure white plate ROI
    white_plate = np.full((30, 80, 3), 220, dtype=np.uint8)
    # Add dark text strokes
    white_plate[8:22, 15:65] = 20
    assert PrivacyRedactor._check_plate_color_validity(white_plate) is True

    # Dusty road / asphalt patch (low contrast, grey-brown)
    asphalt_patch = np.full((30, 80, 3), [135, 130, 120], dtype=np.uint8)
    assert PrivacyRedactor._check_plate_color_validity(asphalt_patch) is False

def test_redact_execution():
    redactor = PrivacyRedactor()
    raw = create_synthetic_image()
    redacted_bytes, meta = redactor.redact(raw, style="blackbox")
    assert isinstance(redacted_bytes, bytes)
    assert len(redacted_bytes) > 0
    assert "latency_ms" in meta
    assert "total_redactions" in meta
    assert meta["faces_redacted"] == 0
    assert meta["plates_redacted"] == 0

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
    assert data["total_detected"] == 0

if __name__ == "__main__":
    print("Running engine unit tests...")
    test_engine_initialization()
    test_exif_stripping()
    test_landmark_geometric_validation()
    test_plate_color_filtering()
    test_redact_execution()
    test_api_health()
    test_api_redact_file_endpoint()
    test_api_detect_endpoint()
    print("All unit tests passed successfully!")
