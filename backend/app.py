"""
Smart Attendance System - Production Backend
Flask REST API with two subsystems:
  1. Enrollment System: Register students via webcam (captures face + stores in DB)
  2. Detection System: Identify students via webcam and mark attendance
"""

from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS
import cv2
import numpy as np
import pickle
import os
import csv
import json
import base64
import time
import sqlite3
import threading
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Data')
PHOTOS_DIR = os.path.join(BASE_DIR, '..', 'student_photos')
ATTENDANCE_DIR = os.path.join(BASE_DIR, '..', 'Attendance_Records')
DB_PATH = os.path.join(DATA_DIR, 'attendance.db')
CASCADE_PATH = os.path.join(DATA_DIR, 'haarcascade_frontalface_default.xml')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# ─── Download Haar Cascade if not present ─────────────────────────────────────
def ensure_cascade():
    if not os.path.exists(CASCADE_PATH):
        import urllib.request
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            urllib.request.urlretrieve(url, CASCADE_PATH)
        except Exception as e:
            # Try cv2 data path
            cv2_data = cv2.data.haarcascades
            src = os.path.join(cv2_data, 'haarcascade_frontalface_default.xml')
            if os.path.exists(src):
                import shutil
                shutil.copy(src, CASCADE_PATH)

ensure_cascade()

# ─── Database Setup ───────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        roll_number TEXT,
        class_name TEXT,
        email TEXT,
        photo_path TEXT,
        enrolled_at TEXT,
        face_samples INTEGER DEFAULT 0
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        student_name TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        confidence REAL,
        status TEXT DEFAULT 'Present',
        UNIQUE(student_id, date)
    )''')
    conn.commit()
    conn.close()

init_db()

# ─── Face Utilities ───────────────────────────────────────────────────────────
face_detector = None

def get_face_detector():
    global face_detector
    if face_detector is None:
        if os.path.exists(CASCADE_PATH):
            face_detector = cv2.CascadeClassifier(CASCADE_PATH)
        else:
            # fallback
            cv2_data = cv2.data.haarcascades
            face_detector = cv2.CascadeClassifier(
                os.path.join(cv2_data, 'haarcascade_frontalface_default.xml'))
    return face_detector

def decode_base64_image(data_url):
    """Decode base64 image from data URL."""
    if ',' in data_url:
        header, data = data_url.split(',', 1)
    else:
        data = data_url
    img_bytes = base64.b64decode(data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def encode_image_base64(img):
    """Encode cv2 image to base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def detect_faces(img):
    """Detect faces in image, return list of (x,y,w,h)."""
    fd = get_face_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = fd.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    return faces if len(faces) > 0 else []

def extract_face_encoding(img, face_rect):
    """Extract 50x50 flattened face encoding."""
    x, y, w, h = face_rect
    face = img[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (50, 50))
    return face_resized.flatten()

def load_face_data():
    """Load pickled face data."""
    names_path = os.path.join(DATA_DIR, 'names.pkl')
    faces_path = os.path.join(DATA_DIR, 'faces_data.pkl')
    if not os.path.exists(names_path) or not os.path.exists(faces_path):
        return None, None
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    with open(faces_path, 'rb') as f:
        faces = pickle.load(f)
    return names, faces

def save_face_data(name, face_samples):
    """Append new face data to pickle files."""
    names_path = os.path.join(DATA_DIR, 'names.pkl')
    faces_path = os.path.join(DATA_DIR, 'faces_data.pkl')
    
    new_faces = np.array(face_samples)  # shape: (N, 7500)
    new_names = [name] * len(face_samples)
    
    if os.path.exists(names_path) and os.path.exists(faces_path):
        with open(names_path, 'rb') as f:
            existing_names = pickle.load(f)
        with open(faces_path, 'rb') as f:
            existing_faces = pickle.load(f)
        existing_names += new_names
        existing_faces = np.vstack([existing_faces, new_faces])
    else:
        existing_names = new_names
        existing_faces = new_faces
    
    with open(names_path, 'wb') as f:
        pickle.dump(existing_names, f)
    with open(faces_path, 'wb') as f:
        pickle.dump(existing_faces, f)

def get_knn_model():
    """Load and return trained KNN model."""
    names, faces = load_face_data()
    if names is None or len(names) == 0:
        return None, None
    k = min(5, len(set(names)))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(faces, names)
    return knn, names

# ─── Enrollment System ────────────────────────────────────────────────────────
# In-memory enrollment sessions
enrollment_sessions = {}

@app.route('/')
def index():
    # This serves your index.html file to the browser
    return send_file('E:\\Computer Vision\\smart_attendance\\frontend\\index.html')

@app.route('/api/students', methods=['GET'])
def get_students():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM students ORDER BY enrolled_at DESC')
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({'students': rows, 'count': len(rows)})

@app.route('/api/students/<student_id>', methods=['GET'])
def get_student(student_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM students WHERE student_id = ?', (student_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({'error': 'Student not found'}), 404
    return jsonify(dict(row))

@app.route('/api/students/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    """Delete student and rebuild face data without them."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT name FROM students WHERE student_id = ?', (student_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Student not found'}), 404
    student_name = row[0]
    c.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
    conn.commit()
    conn.close()

    # Remove from face data
    names_path = os.path.join(DATA_DIR, 'names.pkl')
    faces_path = os.path.join(DATA_DIR, 'faces_data.pkl')
    if os.path.exists(names_path) and os.path.exists(faces_path):
        with open(names_path, 'rb') as f:
            names = pickle.load(f)
        with open(faces_path, 'rb') as f:
            faces = pickle.load(f)
        mask = [n != student_name for n in names]
        names = [n for n, m in zip(names, mask) if m]
        faces = faces[mask]
        with open(names_path, 'wb') as f:
            pickle.dump(names, f)
        with open(faces_path, 'wb') as f:
            pickle.dump(faces, f)

    return jsonify({'message': 'Student deleted successfully'})

@app.route('/api/enroll/start', methods=['POST'])
def start_enrollment():
    """Initialize an enrollment session."""
    data = request.json
    required = ['student_id', 'name']
    for field in required:
        if not data.get(field):
            return jsonify({'error': f'{field} is required'}), 400

    student_id = data['student_id'].strip()
    
    # Check duplicate
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id FROM students WHERE student_id = ?', (student_id,))
    if c.fetchone():
        conn.close()
        return jsonify({'error': 'Student ID already exists'}), 409
    conn.close()

    enrollment_sessions[student_id] = {
        'student_id': student_id,
        'name': data['name'].strip(),
        'roll_number': data.get('roll_number', ''),
        'class_name': data.get('class_name', ''),
        'email': data.get('email', ''),
        'face_samples': [],
        'started_at': datetime.now().isoformat()
    }
    return jsonify({
        'message': 'Enrollment session started',
        'session_id': student_id,
        'samples_needed': 10
    })

@app.route('/api/enroll/capture', methods=['POST'])
def capture_face():
    """Capture a single face frame during enrollment."""
    data = request.json
    session_id = data.get('session_id')
    image_data = data.get('image')

    if not session_id or session_id not in enrollment_sessions:
        return jsonify({'error': 'Invalid or expired session'}), 400
    if not image_data:
        return jsonify({'error': 'No image provided'}), 400

    session = enrollment_sessions[session_id]
    img = decode_base64_image(image_data)
    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    faces = detect_faces(img)
    if len(faces) == 0:
        return jsonify({'status': 'no_face', 'message': 'No face detected', 'samples': len(session['face_samples'])})

    if len(faces) > 1:
        return jsonify({'status': 'multiple_faces', 'message': 'Multiple faces detected. Please ensure only one face is visible.', 'samples': len(session['face_samples'])})

    face = faces[0]
    encoding = extract_face_encoding(img, face)
    session['face_samples'].append(encoding)

    # Draw feedback on image
    x, y, w, h = face
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 100), 2)
    cv2.putText(img, f"Sample {len(session['face_samples'])}/10", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

    annotated = encode_image_base64(img)
    return jsonify({
        'status': 'captured',
        'samples': len(session['face_samples']),
        'samples_needed': 10,
        'annotated_image': f'data:image/jpeg;base64,{annotated}'
    })

@app.route('/api/enroll/complete', methods=['POST'])
def complete_enrollment():
    """Finalize enrollment - save face data and student record."""
    data = request.json
    session_id = data.get('session_id')
    photo_data = data.get('photo')  # best quality photo for display

    if not session_id or session_id not in enrollment_sessions:
        return jsonify({'error': 'Invalid or expired session'}), 400

    session = enrollment_sessions[session_id]
    if len(session['face_samples']) < 5:
        return jsonify({'error': f'Need at least 5 face samples. Got {len(session["face_samples"])}'}), 400

    # Save photo
    photo_path = ''
    if photo_data:
        img = decode_base64_image(photo_data)
        if img is not None:
            photo_filename = f"{session['student_id']}.jpg"
            photo_path = os.path.join(PHOTOS_DIR, photo_filename)
            cv2.imwrite(photo_path, img)
            photo_path = photo_filename

    # Save face data to pickle
    save_face_data(session['name'], session['face_samples'])

    # Save to database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('''INSERT INTO students 
            (student_id, name, roll_number, class_name, email, photo_path, enrolled_at, face_samples)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (session['student_id'], session['name'], session['roll_number'],
             session['class_name'], session['email'], photo_path,
             datetime.now().isoformat(), len(session['face_samples'])))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Student ID already exists'}), 409
    conn.close()

    del enrollment_sessions[session_id]
    return jsonify({
        'message': f'Student {session["name"]} enrolled successfully!',
        'student_id': session_id,
        'samples_captured': len(session['face_samples'])
    })

# ─── Detection System ──────────────────────────────────────────────────────────
@app.route('/api/detect', methods=['POST'])
def detect_attendance():
    """Detect faces in an image and identify registered students."""
    data = request.json
    image_data = data.get('image')
    if not image_data:
        return jsonify({'error': 'No image provided'}), 400

    img = decode_base64_image(image_data)
    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    knn, all_names = get_knn_model()
    if knn is None:
        return jsonify({'error': 'No students enrolled yet. Please enroll students first.'}), 400

    faces = detect_faces(img)
    if len(faces) == 0:
        return jsonify({'status': 'no_face', 'detections': [], 'annotated_image': None})

    detections = []
    annotated = img.copy()

    for (x, y, w, h) in faces:
        encoding = extract_face_encoding(img, (x, y, w, h)).reshape(1, -1)
        predicted_name = knn.predict(encoding)[0]
        probas = knn.predict_proba(encoding)[0]
        confidence = float(max(probas)) * 100

        # Draw on image
        color = (0, 255, 100) if confidence > 60 else (0, 165, 255)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(annotated, (x, y-45), (x+w, y), color, -1)
        cv2.putText(annotated, predicted_name, (x+4, y-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, f"{confidence:.0f}%", (x+4, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        detections.append({
            'name': predicted_name,
            'confidence': round(confidence, 1),
            'bbox': [int(x), int(y), int(w), int(h)]
        })

    annotated_b64 = encode_image_base64(annotated)
    return jsonify({
        'status': 'detected',
        'detections': detections,
        'face_count': len(faces),
        'annotated_image': f'data:image/jpeg;base64,{annotated_b64}'
    })

@app.route('/api/attendance/mark', methods=['POST'])
def mark_attendance():
    """Mark attendance for identified students."""
    data = request.json
    detections = data.get('detections', [])
    if not detections:
        return jsonify({'error': 'No detections to mark'}), 400

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    marked = []
    already_marked = []

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for det in detections:
        name = det['name']
        confidence = det.get('confidence', 0)
        if confidence < 40:
            continue

        # Find student_id by name
        c.execute('SELECT student_id FROM students WHERE name = ?', (name,))
        row = c.fetchone()
        if not row:
            continue
        student_id = row[0]

        try:
            c.execute('''INSERT INTO attendance (student_id, student_name, date, time, confidence)
                         VALUES (?, ?, ?, ?, ?)''',
                      (student_id, name, date_str, time_str, confidence))
            marked.append({'name': name, 'time': time_str, 'confidence': confidence})
        except sqlite3.IntegrityError:
            already_marked.append(name)

    conn.commit()

    # Also write CSV
    if marked:
        csv_path = os.path.join(ATTENDANCE_DIR, f'Attendance_{date_str}.csv')
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['NAME', 'DATE', 'TIME', 'CONFIDENCE', 'STATUS'])
            for m in marked:
                writer.writerow([m['name'], date_str, m['time'], f"{m['confidence']:.1f}%", 'Present'])

    conn.close()
    return jsonify({
        'marked': marked,
        'already_marked': already_marked,
        'date': date_str
    })

# ─── Attendance Records ────────────────────────────────────────────────────────
@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    date = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM attendance WHERE date = ? ORDER BY time DESC', (date,))
    records = [dict(r) for r in c.fetchall()]
    
    # Stats
    c.execute('SELECT COUNT(DISTINCT student_id) FROM students')
    total_students = c.fetchone()[0]
    conn.close()
    
    return jsonify({
        'date': date,
        'records': records,
        'present': len(records),
        'total': total_students,
        'absent': max(0, total_students - len(records))
    })

@app.route('/api/attendance/dates', methods=['GET'])
def get_attendance_dates():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT DISTINCT date FROM attendance ORDER BY date DESC LIMIT 30')
    dates = [row[0] for row in c.fetchall()]
    conn.close()
    return jsonify({'dates': dates})

@app.route('/api/attendance/download', methods=['GET'])
def download_attendance():
    date = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    csv_path = os.path.join(ATTENDANCE_DIR, f'Attendance_{date}.csv')
    if not os.path.exists(csv_path):
        return jsonify({'error': 'No attendance file found for this date'}), 404
    return send_file(csv_path, as_attachment=True, download_name=f'Attendance_{date}.csv')

@app.route('/api/student_photo/<filename>')
def get_photo(filename):
    photo_path = os.path.join(PHOTOS_DIR, filename)
    if os.path.exists(photo_path):
        return send_file(photo_path)
    return jsonify({'error': 'Photo not found'}), 404

@app.route('/api/stats', methods=['GET'])
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM students')
    total_students = c.fetchone()[0]
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute('SELECT COUNT(*) FROM attendance WHERE date = ?', (today,))
    today_present = c.fetchone()[0]
    c.execute('SELECT COUNT(DISTINCT date) FROM attendance')
    total_sessions = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM attendance')
    total_records = c.fetchone()[0]
    conn.close()
    return jsonify({
        'total_students': total_students,
        'today_present': today_present,
        'total_sessions': total_sessions,
        'total_records': total_records,
        'today': today
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("Smart Attendance System Backend running on http://localhost:5000")
    app.run(debug=True, port=5000, threaded=True)
