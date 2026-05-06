# 🧠 NeuralAttend — Smart Face Recognition Attendance System

A production-grade upgrade of the [Smart_Attendence_System](https://github.com/Aryan556gaur/Smart_Attendence_System) with a full REST API backend and beautiful web dashboard frontend.

---

## 🏗️ Architecture

```
smart_attendance/
├── backend/
│   ├── app.py              ← Flask REST API (enrollment + detection)
│   └── requirements.txt
├── frontend/
│   └── index.html          ← Single-page web dashboard
├── Data/                   ← Face data (PKL files) + DB + Haar cascade
├── student_photos/         ← Student profile photos
├── Attendance_Records/     ← CSV attendance files per date
├── start.sh               ← Quick startup script
└── README.md
```

---

## ⚙️ Two Core Systems

### 1. 📸 Enrollment System (`/api/enroll/*`)
- Admin fills in student details (ID, name, roll no., class, email)
- Live webcam opens in browser
- System captures **10 face samples** per student
- Face encodings (50×50 px flattened = 7500 features) stored in `Data/faces_data.pkl`
- Student profile photo saved in `student_photos/`
- Student record saved in SQLite database

### 2. 🔍 Detection System (`/api/detect` + `/api/attendance/mark`)
- Opens browser webcam
- Sends frames to backend
- Backend uses **KNN classifier (k=5)** to identify faces
- Returns annotated image with name + confidence %
- Admin clicks "Mark Attendance" to log identified students
- Records saved to SQLite DB + CSV file per date

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam
- Modern web browser (Chrome/Firefox recommended)

### Installation

```bash
# 1. Install backend dependencies
pip install flask flask-cors opencv-python-headless scikit-learn numpy pandas Pillow

# 2. Start the backend
bash start.sh
# OR directly:
python3 backend/app.py
```

### Open the Frontend
Open `frontend/index.html` in your browser — **no build step needed**.

> The frontend connects to `http://localhost:5000` by default.

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Backend health check |
| GET | `/api/stats` | Dashboard statistics |
| GET | `/api/students` | List all students |
| DELETE | `/api/students/:id` | Delete a student |
| POST | `/api/enroll/start` | Start enrollment session |
| POST | `/api/enroll/capture` | Capture one face sample |
| POST | `/api/enroll/complete` | Finalize enrollment |
| POST | `/api/detect` | Detect & identify faces in image |
| POST | `/api/attendance/mark` | Record attendance |
| GET | `/api/attendance` | Get attendance by date |
| GET | `/api/attendance/download` | Download CSV |

---

## 🎨 Frontend Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Stats overview, today's attendance, quick actions |
| **Enroll Student** | 3-step wizard: details → face capture → confirmation |
| **Take Attendance** | Live webcam detection + one-click attendance marking |
| **Students** | Grid view of all enrolled students, searchable, deletable |
| **Attendance Log** | Date-filtered attendance table with CSV download |

---

## 🔧 Tech Stack

**Backend**
- Python 3 + Flask
- OpenCV (Haar Cascade face detection)
- scikit-learn KNN classifier
- SQLite (student + attendance database)
- Pickle (face encoding storage)
- CORS enabled for browser access

**Frontend**
- Vanilla HTML/CSS/JS (zero dependencies)
- Real-time webcam via `getUserMedia` API
- Base64 image streaming to backend
- Responsive dark theme dashboard

---