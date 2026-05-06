"""
app.py
Flask application factory and entry point.

Run with:
    python app.py
or via start.sh / start.bat at the project root.
"""

import sys

from flask import Flask, send_file
from flask_cors import CORS

from config import FRONTEND_INDEX
from core.database import init_db
from core.face_utils import get_face_detector   # warm up cascade on startup

from routes.students   import students_bp
from routes.enrollment import enrollment_bp
from routes.detection  import detection_bp
from routes.attendance import attendance_bp


def create_app() -> Flask:
    """Initialise and return the configured Flask application."""
    app = Flask(__name__)
    CORS(app)

    # ── Bootstrap ──────────────────────────────────────────────────────────────
    init_db()
    get_face_detector()   # load cascade once; cached for subsequent requests

    # ── Register blueprints ────────────────────────────────────────────────────
    for bp in (students_bp, enrollment_bp, detection_bp, attendance_bp):
        app.register_blueprint(bp)

    # ── Frontend convenience route ─────────────────────────────────────────────
    @app.get('/')
    def index():
        return send_file(FRONTEND_INDEX)

    return app


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print('Smart Attendance System — http://localhost:5000')
    create_app().run(debug=True, port=5000, threaded=True)
