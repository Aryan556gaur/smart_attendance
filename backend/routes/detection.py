"""
routes/detection.py
Real-time face detection and attendance marking.

  POST /api/detect           – identify faces in a webcam frame
  POST /api/attendance/mark  – write identified faces to DB + CSV
"""

import csv
import os
from datetime import datetime

import cv2
from flask import Blueprint, jsonify, request

from config import ATTENDANCE_DIR, DETECT_MIN_CONF
from core.database import get_conn
from core.face_store import get_knn_model
from core.face_utils import decode_base64_image, encode_image_base64, detect_faces, extract_face_encoding

detection_bp = Blueprint('detection', __name__)


# ── Detect ─────────────────────────────────────────────────────────────────────

@detection_bp.post('/api/detect')
def detect_attendance():
    """
    Accept a base64 webcam frame, run KNN recognition on every detected
    face, and return annotated image + structured detections list.
    """
    data = request.json or {}
    image_data = data.get('image')
    if not image_data:
        return jsonify({'error': 'No image provided'}), 400

    img = decode_base64_image(image_data)
    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    knn, _ = get_knn_model()
    if knn is None:
        return jsonify({'error': 'No students enrolled yet. Please enroll students first.'}), 400

    faces = detect_faces(img)
    if not faces:
        return jsonify({'status': 'no_face', 'detections': [], 'annotated_image': None})

    detections = []
    annotated  = img.copy()

    for (x, y, w, h) in faces:
        encoding       = extract_face_encoding(img, (x, y, w, h)).reshape(1, -1)
        predicted_name = knn.predict(encoding)[0]
        confidence     = float(max(knn.predict_proba(encoding)[0])) * 100

        color = (0, 255, 100) if confidence > 60 else (0, 165, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(annotated, (x, y - 45), (x + w, y), color, -1)
        cv2.putText(annotated, predicted_name, (x + 4, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, f'{confidence:.0f}%', (x + 4, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        detections.append({
            'name':       predicted_name,
            'confidence': round(confidence, 1),
            'bbox':       [int(x), int(y), int(w), int(h)],
        })

    return jsonify({
        'status':          'detected',
        'detections':      detections,
        'face_count':      len(faces),
        'annotated_image': f'data:image/jpeg;base64,{encode_image_base64(annotated)}',
    })


# ── Mark attendance ────────────────────────────────────────────────────────────

@detection_bp.post('/api/attendance/mark')
def mark_attendance():
    """
    Persist attendance records to SQLite and the daily CSV file.
    Skips detections below DETECT_MIN_CONF or already marked today.
    """
    data = request.json or {}
    detections = data.get('detections', [])
    if not detections:
        return jsonify({'error': 'No detections to mark'}), 400

    now      = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    marked:         list[dict] = []
    already_marked: list[str]  = []

    conn = get_conn()
    c    = conn.cursor()

    for det in detections:
        name       = det['name']
        confidence = det.get('confidence', 0)
        if confidence < DETECT_MIN_CONF:
            continue

        c.execute('SELECT student_id FROM students WHERE name = ?', (name,))
        row = c.fetchone()
        if not row:
            continue
        student_id = row['student_id']

        try:
            c.execute(
                '''INSERT INTO attendance
                   (student_id, student_name, date, time, confidence)
                   VALUES (?, ?, ?, ?, ?)''',
                (student_id, name, date_str, time_str, confidence),
            )
            marked.append({'name': name, 'time': time_str, 'confidence': confidence})
        except Exception:
            already_marked.append(name)

    conn.commit()
    conn.close()

    # Append to daily CSV
    if marked:
        _append_csv(date_str, marked)

    return jsonify({
        'marked':         marked,
        'already_marked': already_marked,
        'date':           date_str,
    })


# ── Private helper ─────────────────────────────────────────────────────────────

def _append_csv(date_str: str, records: list[dict]) -> None:
    """Create or append to the daily attendance CSV file."""
    csv_path   = os.path.join(ATTENDANCE_DIR, f'Attendance_{date_str}.csv')
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['NAME', 'DATE', 'TIME', 'CONFIDENCE', 'STATUS'])
        for rec in records:
            writer.writerow([
                rec['name'], date_str, rec['time'],
                f"{rec['confidence']:.1f}%", 'Present',
            ])
