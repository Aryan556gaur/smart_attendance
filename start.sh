#!/bin/bash
# Smart Attendance System - Backend Startup Script

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  NeuralAttend — Smart Attendance System"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 not found. Please install Python 3.8+"
  exit 1
fi

# Go to project root
cd "$(dirname "$0")"

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r backend/requirements.txt -q --break-system-packages 2>/dev/null || \
pip3 install -r backend/requirements.txt -q

# Copy Haar cascade if needed
HAAR_SRC=$(python3 -c "import cv2; import os; print(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))" 2>/dev/null)
HAAR_DEST="Data/haarcascade_frontalface_default.xml"
if [ -f "$HAAR_SRC" ] && [ ! -f "$HAAR_DEST" ]; then
  echo "📁 Copying Haar cascade..."
  cp "$HAAR_SRC" "$HAAR_DEST"
fi

echo ""
echo "🚀 Starting backend on http://localhost:5000"
echo "🌐 Open frontend: frontend/index.html in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start backend
python3 backend/app.py
