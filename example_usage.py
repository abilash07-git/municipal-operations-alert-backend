"""
Example Usage: Privacy Redaction Engine
Demonstrates both Python Library usage and REST API client calls.
"""

import sys
import os
from redactor import PrivacyRedactor

def demo_python_library():
    print("=" * 60)
    print("1. Using PrivacyRedactor as a Python Library")
    print("=" * 60)

    # Initialize the engine
    redactor = PrivacyRedactor(face_score_threshold=0.30)

    # Create or load sample image
    from PIL import Image, ImageDraw
    sample = Image.new("RGB", (600, 400), color=(200, 220, 240))
    draw = ImageDraw.Draw(sample)
    draw.rectangle([200, 100, 300, 200], fill=(230, 190, 150)) # Mock face
    draw.rectangle([180, 280, 420, 340], fill=(255, 255, 255)) # Mock plate
    draw.text((220, 300), "DL 01 AB 1234", fill=(0, 0, 0))

    import io
    buf = io.BytesIO()
    sample.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    # Redact using blackbox style
    redacted_bytes, meta = redactor.redact(raw_bytes, style="blackbox")
    print(f"Redaction completed in {meta['latency_ms']} ms")
    print(f"Faces redacted: {meta['faces_redacted']}")
    print(f"Plates redacted: {meta['plates_redacted']}")
    print(f"Output JPEG size: {len(redacted_bytes)} bytes")

    # Or redact using gaussian blur style
    blurred_bytes, _ = redactor.redact(raw_bytes, style="blur", blur_kernel_size=45)
    print("Gaussian blur redaction also available!")

def demo_rest_api_client():
    print("\n" + "=" * 60)
    print("2. Calling the Privacy Redaction REST API")
    print("=" * 60)
    print("""
# Using cURL:
curl -X POST "http://localhost:8000/v1/redact?style=blackbox" \\
     -F "file=@your_photo.jpg" \\
     --output "redacted_photo.jpg"

# Using Python requests:
import requests
with open("your_photo.jpg", "rb") as f:
    resp = requests.post("http://localhost:8000/v1/redact", files={"file": f})
with open("redacted_photo.jpg", "wb") as f:
    f.write(resp.content)

# Using JavaScript (Fetch API):
const formData = new FormData();
formData.append("file", fileInput.files[0]);
const res = await fetch("http://localhost:8000/v1/redact", {
  method: "POST",
  body: formData
});
const blob = await res.blob();
const imgUrl = URL.createObjectURL(blob);
    """)

if __name__ == "__main__":
    demo_python_library()
    demo_rest_api_client()
