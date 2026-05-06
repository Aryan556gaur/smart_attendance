"""
routes/enrollment.py
Three-step enrollment flow:

  POST /api/enroll/start    – open a session
  POST /api/enroll/capture  – submit one webcam frame
  POST /api/enroll/complete – finalise and persist
"""

import os
from datetime import datetime

import cv2
from flask import Blueprint, jsonify, request

from config import PHOTOS_DIR, ENROLL_MIN_SAMPLES
from core.database import get_conn
from core.face_store import save_face_data
from core.face_utils import decode_base64_image, encode_image_base64, detect_faces, extract_face_encoding

enrollment_bp = Blueprint('enrollment', __name__)

# In-memory store: session_id → session dict
_sessions: dict[str, dict] = {}


# ── Start ──────────────────────────────────────────────────────────────────────

@enrollment_bp.post('/api/enroll/start')
def start_enrollment():
    """Validate input, check for duplicate ID, open a new session."""
    data = request.json or {}
    for field in ('student_id', 'name'):
        if not data.get(field):
            return jsonify({'error': f'{field} is required'}), 400

    student_id = data['student_id'].strip()

    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT id FROM students WHERE student_id = ?', (student_id,))
    exists = c.fetchone()
    conn.close()
    if exists:
        return jsonify({'error': 'Student ID already exists'}), 409

    _sessions[student_id] = {
        'student_id': student_id,
        'name':        data['name'].strip(),
        'roll_number': data.get('roll_number', ''),
        'class_name':  data.get('class_name', ''),
        'email':       data.get('email', ''),
        'face_samples': [],
        'started_at':  datetime.now().isoformat(),
    }
    return jsonify({
        'message':        'Enrollment session started',
        'session_id':     student_id,
        'samples_needed': 10,
    })


# ── Capture ────────────────────────────────────────────────────────────────────

@enrollment_bp.post('/api/enroll/capture')
def capture_face():
    """Accept one webcam frame, detect a face, and store its encoding."""
    data = request.json or {}
    session_id = data.get('session_id')
    image_data  = data.get('image')

    if not session_id or session_id not in _sessions:
        return jsonify({'error': 'Invalid or expired session'}), 400
    if not image_data:
        return jsonify({'error': 'No image provided'}), 400

    session = _sessions[session_id]
    img = decode_base64_image(image_data)
    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    faces = detect_faces(img)

    if len(faces) == 0:
        return jsonify({
            'status': 'no_face',
            'message': 'No face detected',
            'samples': len(session['face_samples']),
        })

    if len(faces) > 1:
        return jsonify({
            'status':  'multiple_faces',
            'message': 'Multiple faces detected. Please ensure only one face is visible.',
            'samples': len(session['face_samples']),
        })

    face_rect = faces[0]
    encoding  = extract_face_encoding(img, face_rect)
    session['face_samples'].append(encoding)

    # Visual feedback
    x, y, w, h = face_rect
    count = len(session['face_samples'])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 100), 2)
    cv2.putText(img, f'Sample {count}/10', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

    return jsonify({
        'status':          'captured',
        'samples':         count,
        'samples_needed':  10,
        'annotated_image': f'data:image/jpeg;base64,{encode_image_base64(img)}',
    })


# ── Complete ───────────────────────────────────────────────────────────────────

@enrollment_bp.post('/api/enroll/complete')
def complete_enrollment():
    """Persist face encodings, save profile photo, insert DB record."""
    data = request.json or {}
    session_id = data.get('session_id')
    photo_data  = data.get('photo')

    if not session_id or session_id not in _sessions:
        return jsonify({'error': 'Invalid or expired session'}), 400

    session = _sessions[session_id]
    sample_count = len(session['face_samples'])

    if sample_count < ENROLL_MIN_SAMPLES:
        return jsonify({
            'error': f'Need at least {ENROLL_MIN_SAMPLES} face samples. Got {sample_count}.'
        }), 400

    # Save profile photo
    photo_filename = ''
    if photo_data:
        img = decode_base64_image(photo_data)
        if img is not None:
            photo_filename = f"{session['student_id']}.jpg"
            cv2.imwrite(os.path.join(PHOTOS_DIR, photo_filename), img)

    # Persist face encodings
    save_face_data(session['name'], session['face_samples'])

    # Insert student record
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute(
            '''INSERT INTO students
               (student_id, name, roll_number, class_name, email,
                photo_path, enrolled_at, face_samples)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (session['student_id'], session['name'], session['roll_number'],
             session['class_name'], session['email'], photo_filename,
             datetime.now().isoformat(), sample_count),
        )
        conn.commit()
    except Exception as exc:
        conn.close()
        return jsonify({'error': str(exc)}), 409
    conn.close()

    del _sessions[session_id]
    return jsonify({
        'message':         f"Student {session['name']} enrolled successfully!",
        'student_id':      session_id,
        'samples_captured': sample_count,
    })
