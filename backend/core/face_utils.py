"""
core/face_utils.py
Low-level computer-vision helpers:
  - Haar-cascade loading (with auto-download fallback)
  - Base64 image decode / encode
  - Face detection
  - Face-region encoding (flatten to 1-D feature vector)
"""

import base64
import os
import shutil
import urllib.request

import cv2
import numpy as np

from config import CASCADE_PATH, FACE_SIZE, MIN_FACE_PX, SCALE_FACTOR, MIN_NEIGHBORS

# ── Haar cascade ───────────────────────────────────────────────────────────────
_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "data/haarcascades/haarcascade_frontalface_default.xml"
)

_face_detector: cv2.CascadeClassifier | None = None


def _ensure_cascade() -> None:
    """Download or copy the Haar cascade XML if it isn't on disk yet."""
    if os.path.exists(CASCADE_PATH):
        return
    try:
        urllib.request.urlretrieve(_CASCADE_URL, CASCADE_PATH)
        return
    except Exception:
        pass
    # Fallback: copy from OpenCV's own data bundle
    bundled = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    if os.path.exists(bundled):
        shutil.copy(bundled, CASCADE_PATH)


def get_face_detector() -> cv2.CascadeClassifier:
    """Return a singleton CascadeClassifier, initialising on first call."""
    global _face_detector
    if _face_detector is None:
        _ensure_cascade()
        src = CASCADE_PATH if os.path.exists(CASCADE_PATH) else os.path.join(
            cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'
        )
        _face_detector = cv2.CascadeClassifier(src)
    return _face_detector


# ── Image codec helpers ────────────────────────────────────────────────────────

def decode_base64_image(data_url: str) -> np.ndarray | None:
    """Decode a base64 data-URL string into a BGR cv2 image."""
    try:
        _, data = data_url.split(',', 1) if ',' in data_url else ('', data_url)
        img_bytes = base64.b64decode(data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def encode_image_base64(img: np.ndarray) -> str:
    """Encode a BGR cv2 image to a base64 JPEG string (no data-URL prefix)."""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


# ── Detection & encoding ───────────────────────────────────────────────────────

def detect_faces(img: np.ndarray) -> list[tuple]:
    """
    Run Haar-cascade on *img* and return a list of (x, y, w, h) tuples.
    Returns an empty list when no faces are found.
    """
    fd = get_face_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = fd.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=(MIN_FACE_PX, MIN_FACE_PX),
    )
    return list(faces) if len(faces) > 0 else []


def extract_face_encoding(img: np.ndarray, face_rect: tuple) -> np.ndarray:
    """
    Crop *face_rect* from *img*, resize to FACE_SIZE, and flatten to 1-D.
    The resulting vector has FACE_SIZE[0] * FACE_SIZE[1] * 3 elements (7 500
    for the default 50×50 RGB setting).
    """
    x, y, w, h = face_rect
    face_crop = img[y:y + h, x:x + w]
    face_resized = cv2.resize(face_crop, FACE_SIZE)
    return face_resized.flatten()
