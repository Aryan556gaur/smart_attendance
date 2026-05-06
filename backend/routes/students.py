"""
routes/students.py
CRUD endpoints for student records.

  GET    /api/students          – list all students
  GET    /api/students/<id>     – get one student
  DELETE /api/students/<id>     – remove student + face data
  GET    /api/student_photo/<f> – serve profile photo
"""

import os

from flask import Blueprint, jsonify, request, send_file

from config import PHOTOS_DIR
from core.database import get_conn
from core.face_store import delete_face_data_for

students_bp = Blueprint('students', __name__)


@students_bp.get('/api/students')
def list_students():
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT * FROM students ORDER BY enrolled_at DESC')
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({'students': rows, 'count': len(rows)})


@students_bp.get('/api/students/<student_id>')
def get_student(student_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT * FROM students WHERE student_id = ?', (student_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({'error': 'Student not found'}), 404
    return jsonify(dict(row))


@students_bp.delete('/api/students/<student_id>')
def delete_student(student_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT name FROM students WHERE student_id = ?', (student_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Student not found'}), 404

    student_name = row['name']
    c.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
    conn.commit()
    conn.close()

    delete_face_data_for(student_name)
    return jsonify({'message': 'Student deleted successfully'})


@students_bp.get('/api/student_photo/<filename>')
def get_photo(filename: str):
    photo_path = os.path.join(PHOTOS_DIR, filename)
    if os.path.exists(photo_path):
        return send_file(photo_path)
    return jsonify({'error': 'Photo not found'}), 404
