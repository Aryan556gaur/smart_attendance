"""
config.py
All paths, directory setup, and application-level constants.
"""

import os

# ── Root dirs ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR       = os.path.join(PROJECT_DIR, 'Data')
PHOTOS_DIR     = os.path.join(PROJECT_DIR, 'student_photos')
ATTENDANCE_DIR = os.path.join(PROJECT_DIR, 'Attendance_Records')

# ── File paths ─────────────────────────────────────────────────────────────────
DB_PATH      = os.path.join(DATA_DIR, 'attendance.db')
CASCADE_PATH = os.path.join(DATA_DIR, 'haarcascade_frontalface_default.xml')
NAMES_PKL    = os.path.join(DATA_DIR, 'names.pkl')
FACES_PKL    = os.path.join(DATA_DIR, 'faces_data.pkl')

FRONTEND_INDEX = os.path.join(PROJECT_DIR, 'frontend', 'index.html')

# ── Tuning knobs ───────────────────────────────────────────────────────────────
FACE_SIZE         = (50, 50)          # resize before flattening → 7500 features
MIN_FACE_PX       = 30                # haarcascade minSize
SCALE_FACTOR      = 1.3
MIN_NEIGHBORS     = 5
KNN_MAX_K         = 5                 # capped so k ≤ unique students
ENROLL_MIN_SAMPLES = 5                # refuse complete() below this
DETECT_MIN_CONF   = 40.0              # skip marking below this %

# ── Ensure all directories exist ───────────────────────────────────────────────
for _d in (DATA_DIR, PHOTOS_DIR, ATTENDANCE_DIR):
    os.makedirs(_d, exist_ok=True)
