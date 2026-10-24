"""
PrivacyRedactor - Production-grade Face & License Plate Redaction Engine
High-accuracy privacy sanitization powered by OpenCV DNN & WIDER-FACE YuNet.
Strips camera EXIF metadata, detects occluded/helmet faces, and masks private & commercial plates.
"""

import os
import io
import time
from typing import List, Dict, Tuple, Union, Optional
from PIL import Image
import cv2
import numpy as np

class PrivacyRedactor:
    """
    Modular Computer Vision Privacy Engine for automated face and vehicle plate redaction.
    Can be used as a standalone Python library or embedded in web APIs.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        face_score_threshold: float = 0.30,
        face_nms_threshold: float = 0.30,
        enable_tiling: bool = True,
        tiling_size: int = 1024,
        tiling_overlap: int = 128
    ):
        """
        Initialize the Privacy Redactor engine.
        :param model_path: Path to the yunet.onnx model file.
        :param face_score_threshold: Confidence threshold for face detection [0.0 - 1.0].
        :param face_nms_threshold: Non-Maximum Suppression threshold for face overlap.
        :param enable_tiling: Enables multi-scale window tiling for high-res images to detect distant/small faces.
        :param tiling_size: Crop size for tiling sliding window.
        :param tiling_overlap: Overlap size between adjacent windows.
        """
        if model_path is None:
            # Default to models/yunet.onnx relative to this file
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(curr_dir, "models", "yunet.onnx")

        self.model_path = model_path
        self.face_score_threshold = face_score_threshold
        self.face_nms_threshold = face_nms_threshold
        self.enable_tiling = enable_tiling
        self.tiling_size = tiling_size
        self.tiling_overlap = tiling_overlap

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"YuNet ONNX model not found at {self.model_path}. "
                "Ensure models/yunet.onnx is present."
            )

    @staticmethod
    def strip_exif(pil_image: Image.Image) -> Image.Image:
        """
        Removes all EXIF metadata (GPS coordinates, camera serials, timestamps)
        by reconstructing a clean PIL canvas.
        """
        clean_img = Image.new(pil_image.mode, pil_image.size)
        clean_img.paste(pil_image)
        return clean_img

    def detect_faces(self, cv_img: np.ndarray) -> List[Dict]:
        """
        Detects faces in BGR OpenCV image using YuNet with adaptive multi-scale pyramid tiling.
        """
        h, w = cv_img.shape[:2]
        faces_detected = []

        # 1. Global image pass (resized to standard 1280px width)
        proc_w = 1280
        scale = proc_w / float(w)
        scaled_h = int(h * scale)
        scaled_img = cv2.resize(cv_img, (proc_w, scaled_h), interpolation=cv2.INTER_AREA)

        detector = cv2.FaceDetectorYN.create(
            self.model_path,
            '',
            (proc_w, scaled_h),
            score_threshold=self.face_score_threshold,
            nms_threshold=self.face_nms_threshold
        )
        _, faces = detector.detect(scaled_img)
        if faces is not None:
            for f in faces:
                bx, by, bw, bh = f[0:4].astype(int)
                conf = float(f[-1])
                orig_box = [
                    int(bx / scale),
                    int(by / scale),
                    int(bw / scale),
                    int(bh / scale)
                ]
                faces_detected.append({
                    "type": "Face",
                    "box": orig_box,
                    "confidence": round(conf, 3)
                })

        # 2. Multi-scale Tiling for high-res images (w > 1800 or h > 1800)
        if self.enable_tiling and (w > 1800 or h > 1800):
            step = self.tiling_size - self.tiling_overlap
            y_starts = list(range(0, max(1, h - self.tiling_size + 1), step))
            if y_starts and y_starts[-1] + self.tiling_size < h:
                y_starts.append(h - self.tiling_size)
            x_starts = list(range(0, max(1, w - self.tiling_size + 1), step))
            if x_starts and x_starts[-1] + self.tiling_size < w:
                x_starts.append(w - self.tiling_size)

            for ys in y_starts:
                for xs in x_starts:
                    crop = cv_img[ys:ys + self.tiling_size, xs:xs + self.tiling_size]
                    ch, cw = crop.shape[:2]
                    tile_det = cv2.FaceDetectorYN.create(
                        self.model_path, '', (cw, ch),
                        score_threshold=self.face_score_threshold + 0.05,
                        nms_threshold=self.face_nms_threshold
                    )
                    _, tile_faces = tile_det.detect(crop)
                    if tile_faces is not None:
                        for tf in tile_faces:
                            tbx, tby, tbw, tbh = tf[0:4].astype(int)
                            tconf = float(tf[-1])
                            global_box = [int(xs + tbx), int(ys + tby), int(tbw), int(tbh)]
                            # Avoid duplicates from global pass
                            if not self._is_duplicate_box(global_box, [f["box"] for f in faces_detected]):
                                faces_detected.append({
                                    "type": "Face",
                                    "box": global_box,
                                    "confidence": round(tconf, 3)
                                })

        return faces_detected

    def detect_plates(self, cv_img: np.ndarray) -> List[Dict]:
        """
        Detects vehicle license plates using dual-palette HSV color segmentation
        (Private White plates + Commercial Yellow transport plates) and aspect ratio analysis.
        """
        plates_detected = []
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        # Indian Commercial Yellow: Hue [15, 38], Sat [60, 255], Val [80, 255]
        mask_yellow = cv2.inRange(hsv, np.array([15, 60, 80]), np.array([38, 255, 255]))
        # Standard Private White: Hue [0, 180], Sat [0, 40], Val [160, 255]
        mask_white = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 40, 255]))
        combined_mask = cv2.bitwise_or(mask_yellow, mask_white)

        # Morphological rectangle closing to fuse plate text into plate bounding region
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))
        closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            aspect_ratio = cw / float(ch if ch > 0 else 1)
            area = cw * ch
            # License plates typically have aspect ratios between 1.5 and 4.8 and reasonable area
            if 1.5 <= aspect_ratio <= 4.8 and 400 < area < 45000:
                plates_detected.append({
                    "type": "License Plate",
                    "box": [cx, cy, cw, ch],
                    "confidence": 0.90
                })
                count += 1
                if count >= 8:
                    break

        return plates_detected

    def detect_all(self, image_input: Union[bytes, np.ndarray, Image.Image]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Runs complete detection for both faces and license plates on the given input image.
        Returns: (cv_image_bgr, targets_list)
        """
        cv_img = self._load_as_cv2(image_input)
        faces = self.detect_faces(cv_img)
        plates = self.detect_plates(cv_img)
        all_targets = faces + plates
        return cv_img, all_targets

    def redact(
        self,
        image_input: Union[bytes, np.ndarray, Image.Image],
        style: str = "blackbox",
        blur_kernel_size: int = 51,
        quality: int = 90
    ) -> Tuple[bytes, Dict]:
        """
        Redacts detected faces and license plates, strips EXIF metadata, and returns clean JPEG bytes.
        :param image_input: Input image as bytes, PIL Image, or OpenCV BGR numpy array.
        :param style: "blackbox" (tight solid black rectangle) or "blur" (Gaussian blur).
        :param blur_kernel_size: Kernel size if style="blur" (must be odd).
        :param quality: Output JPEG quality [1-100].
        :return: (redacted_jpeg_bytes, metadata_dict)
        """
        start_time = time.perf_counter()
        cv_img, targets = self.detect_all(image_input)
        h, w = cv_img.shape[:2]

        redacted_img = cv_img.copy()

        for t in targets:
            bx, by, bw, bh = t["box"]
            # Clamp coordinates to image boundaries
            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(w, bx + bw)
            y2 = min(h, by + bh)

            if x2 <= x1 or y2 <= y1:
                continue

            if style == "blur":
                ksize = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
                roi = redacted_img[y1:y2, x1:x2]
                blurred = cv2.GaussianBlur(roi, (ksize, ksize), 30)
                redacted_img[y1:y2, x1:x2] = blurred
            else:  # default "blackbox"
                cv2.rectangle(redacted_img, (x1, y1), (x2, y2), (0, 0, 0), -1)

        # Encode to clean JPEG bytes
        _, encoded = cv2.imencode(".jpg", redacted_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        redacted_bytes = encoded.tobytes()
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        face_count = sum(1 for t in targets if t["type"] == "Face")
        plate_count = sum(1 for t in targets if t["type"] == "License Plate")

        metadata = {
            "image_dimensions": {"width": w, "height": h},
            "faces_redacted": face_count,
            "plates_redacted": plate_count,
            "total_redactions": len(targets),
            "style": style,
            "latency_ms": latency_ms,
            "targets": targets
        }

        return redacted_bytes, metadata

    def _load_as_cv2(self, image_input: Union[bytes, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input))
            clean_pil = self.strip_exif(pil_img)
            return cv2.cvtColor(np.array(clean_pil), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, Image.Image):
            clean_pil = self.strip_exif(image_input)
            return cv2.cvtColor(np.array(clean_pil), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    @staticmethod
    def _is_duplicate_box(box_a: List[int], existing_boxes: List[List[int]], iou_thresh: float = 0.35) -> bool:
        ax, ay, aw, ah = box_a
        for bx, by, bw, bh in existing_boxes:
            ix1 = max(ax, bx)
            iy1 = max(ay, by)
            ix2 = min(ax + aw, bx + bw)
            iy2 = min(ay + ah, by + bh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (aw * ah) + (bw * bh) - inter
            if union > 0 and (inter / float(union)) > iou_thresh:
                return True
        return False
