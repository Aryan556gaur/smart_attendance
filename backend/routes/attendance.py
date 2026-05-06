"""
routes/attendance.py
Read-only attendance data endpoints.

  GET /api/attendance           – records for a given date
  GET /api/attendance/dates     – list of dates that have records
  GET /api/attendance/download  – download daily CSV
  GET /api/stats                – dashboard summary numbers
  GET /api/health               – liveness probe
"""

import os
from datetime import datetime

from flask import Blueprint, jsonify, request, send_file

from config import ATTENDANCE_DIR
from core.database import get_conn

attendance_bp = Blueprint('attendance', __name__)


@attendance_bp.get('/api/attendance')
def get_attendance():
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    conn = get_conn()
    c    = conn.cursor()

    c.execute('SELECT * FROM attendance WHERE date = ? ORDER BY time DESC', (date,))
    records = [dict(r) for r in c.fetchall()]

    c.execute('SELECT COUNT(DISTINCT student_id) FROM students')
    total_students = c.fetchone()[0]
    conn.close()

    return jsonify({
        'date':    date,
        'records': records,
        'present': len(records),
        'total':   total_students,
        'absent':  max(0, total_students - len(records)),
    })


@attendance_bp.get('/api/attendance/dates')
def get_attendance_dates():
    conn = get_conn()
    c    = conn.cursor()
    c.execute('SELECT DISTINCT date FROM attendance ORDER BY date DESC LIMIT 30')
    dates = [row[0] for row in c.fetchall()]
    conn.close()
    return jsonify({'dates': dates})


@attendance_bp.get('/api/attendance/download')
def download_attendance():
    date     = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    csv_path = os.path.join(ATTENDANCE_DIR, f'Attendance_{date}.csv')
    if not os.path.exists(csv_path):
        return jsonify({'error': 'No attendance file found for this date'}), 404
    return send_file(csv_path, as_attachment=True, download_name=f'Attendance_{date}.csv')


@attendance_bp.get('/api/stats')
def get_stats():
    today = datetime.now().strftime('%Y-%m-%d')
    conn  = get_conn()
    c     = conn.cursor()

    c.execute('SELECT COUNT(*) FROM students')
    total_students = c.fetchone()[0]

    c.execute('SELECT COUNT(*) FROM attendance WHERE date = ?', (today,))
    today_present = c.fetchone()[0]

    c.execute('SELECT COUNT(DISTINCT date) FROM attendance')
    total_sessions = c.fetchone()[0]

    c.execute('SELECT COUNT(*) FROM attendance')
    total_records = c.fetchone()[0]

    conn.close()
    return jsonify({
        'total_students': total_students,
        'today_present':  today_present,
        'total_sessions': total_sessions,
        'total_records':  total_records,
        'today':          today,
    })


@attendance_bp.get('/api/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})
