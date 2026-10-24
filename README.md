# AutoPrivacy — Face & Vehicle License Plate Redaction Engine

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Microservice-009688.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-YuNet%20DNN-5C3EE8.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An ultra-fast, high-accuracy, production-ready computer vision engine for **automatically detecting and redacting human faces and vehicle license plates** in photos.

Designed as a **modular drop-in tool** that other developers can easily integrate into their own backend architectures, crowd-sourced apps, municipal systems, and media upload pipelines to ensure privacy compliance (DPDP Act, GDPR).

---

## 🚀 Key Features

* **WIDER-FACE YuNet Deep Neural Network**:
  * Detects faces at extreme angles, side profiles, partially occluded faces, and faces wearing helmets.
  * Native 232 KB ONNX model running on OpenCV DNN — **no PyTorch, TensorFlow, or CUDA required**.
* **Multi-Scale Pyramid Tiling**:
  * Automatically scans high-resolution images using adaptive sliding windows so distant, small background faces in crowded scenes are reliably caught.
* **Dual-Palette Vehicle License Plate Detection**:
  * Accurately detects both **private white number plates** and **commercial transport yellow plates** (auto-rickshaws, cabs, trucks, buses) via calibrated HSV segmentation and aspect-ratio contour filters.
* **Phone Number & Background Digit Masking (DBNet)**:
  * Detects contact numbers, phone numbers, and digit sequences on background signboards, banners, posters, and vehicles via deep DBNet text detection.
* **Camera EXIF Metadata Scrubbing**:
  * Automatically strips camera hardware serial numbers, device model signatures, and embedded GPS geolocation coordinates.
* **Flexible Redaction Styles**:
  * Supports minimal, tight **`blackbox`** solid masking or smooth **`blur`** (Gaussian blur).
* **Dual Integration Modes**:
  1. **Python Library / SDK (`redactor.py`)**: Direct function call inside your existing Python backend.
  2. **REST API Microservice (`app.py`)**: High-performance HTTP server for any frontend, mobile app, or language (Node.js, Go, Java, PHP, etc.).

---

## 📂 Repository Structure

```text
├── redactor.py         # Core standalone Python engine class (PrivacyRedactor)
├── app.py              # FastAPI microservice exposing REST redaction endpoints
├── models/
│   ├── yunet.onnx      # Deep learning face detection model weights (232 KB)
│   └── text_det.onnx   # DBNet deep text & phone number detection model (2.4 MB)
├── example_usage.py    # Runnable example script demonstrating SDK and HTTP usage
├── test_engine.py      # Automated unit tests for engine and API endpoints
├── requirements.txt    # Lightweight Python dependencies
└── README.md           # Full documentation and integration guide
```

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/abilash07-git/municipal-operations-alert-backend.git
cd municipal-operations-alert-backend

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 💡 How to Use in Your Own Engine

### Option 1: As a Python Library (Drop-In SDK)

Import `PrivacyRedactor` directly into your existing Python backend:

```python
from redactor import PrivacyRedactor

# 1. Initialize the engine
redactor = PrivacyRedactor(
    face_score_threshold=0.30,  # Confidence cutoff [0.0 - 1.0]
    enable_tiling=True          # Multi-scale pyramid for high-res images
)

# 2. Load your image bytes
with open("crowd_scene.jpg", "rb") as f:
    raw_bytes = f.read()

# 3. Redact (returns sanitized JPEG bytes + metadata)
redacted_bytes, meta = redactor.redact(raw_bytes, style="blackbox")

print(f"Faces redacted: {meta['faces_redacted']}")
print(f"Plates redacted: {meta['plates_redacted']}")
print(f"Latency: {meta['latency_ms']} ms")

# 4. Save sanitized image
with open("sanitized_public.jpg", "wb") as f:
    f.write(redacted_bytes)
```

#### Gaussian Blur Alternative:
```python
# To blur instead of solid blackbox:
blurred_bytes, meta = redactor.redact(raw_bytes, style="blur", blur_kernel_size=51)
```

---

### Option 2: As a REST API Microservice

Run the included FastAPI server:

```bash
python app.py
```
* **Base URL:** `http://localhost:8000`
* **Interactive Swagger UI:** `http://localhost:8000/docs`

#### Microservice Endpoints Reference:

| Method | Endpoint | Description | Response Type |
|---|---|---|---|
| `POST` | `/v1/redact` | Uploads image and returns sanitized image file directly | `image/jpeg` |
| `POST` | `/v1/redact-json` | Uploads image and returns bounding boxes, metrics & base64 image | `application/json` |
| `POST` | `/v1/detect` | Returns bounding boxes and confidence scores without modifying image | `application/json` |
| `GET` | `/health` | Health check and engine version | `application/json` |

---

## 💻 Client Integration Examples

### 1. cURL (Command Line)
```bash
# Redact an image and save directly:
curl -X POST "http://localhost:8000/v1/redact?style=blackbox" \
     -F "file=@input_photo.jpg" \
     --output "sanitized_photo.jpg"
```

### 2. JavaScript / TypeScript (Frontend & Node.js)
```javascript
// Upload an image and get back the sanitized image blob
async function redactPhoto(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://localhost:8000/v1/redact?style=blackbox", {
    method: "POST",
    body: formData
  });

  const blob = await response.blob();
  const sanitizedUrl = URL.createObjectURL(blob);
  
  // Directly bind to HTML <img> element
  document.getElementById("preview").src = sanitizedUrl;
  console.log("Faces redacted:", response.headers.get("X-Faces-Redacted"));
  console.log("Plates redacted:", response.headers.get("X-Plates-Redacted"));
}
```

### 3. Python (`requests` / `httpx`)
```python
import requests

with open("street_view.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/redact",
        params={"style": "blackbox", "quality": 92},
        files={"file": f}
    )

with open("street_view_redacted.jpg", "wb") as f:
    f.write(resp.content)

print(f"Faces masked: {resp.headers.get('X-Faces-Redacted')}")
print(f"Plates masked: {resp.headers.get('X-Plates-Redacted')}")
print(f"Engine latency: {resp.headers.get('X-Latency-Ms')} ms")
```

---

## 📊 Benchmark & Accuracy Metrics (Hard Real-World Datasets)

| Test Scenario | Image Resolution | Faces Redacted | Plates Redacted | Phone Numbers Redacted | Total PII Redacted | CPU Latency |
|---|---|---|---|---|---|---|
| **Dense Urban Traffic (Varanasi Street)** | 3494 × 2336 | **34 faces** | **12 plates** | **4 numbers** | **50 targets** | ~3.6s |
| **Monsoon Night Glare & Helmet (Chennai Flyover)** | 3264 × 2448 | **4 faces** (in-helmet) | **12 plates** | 0 numbers | **16 targets** | ~3.6s |
| **Extreme Density Market Crowd (T. Nagar Hub)** | 3264 × 2448 | **155 faces** | **12 plates** | **16 numbers** | **183 targets** | ~3.1s |
| **Standard Urban Pedestrians (Argyle Street)** | 2698 × 1799 | **3 faces** | **8 plates** | 0 numbers | **11 targets** | ~1.2s |

---

## 🧪 Running Unit Tests

Run the included automated test suite:

```bash
python test_engine.py
```

Expected output:
```text
Running engine unit tests...
All unit tests passed successfully!
```

---

## 📄 License

This project is licensed under the MIT License — feel free to use and integrate it into your own commercial or open-source projects.
