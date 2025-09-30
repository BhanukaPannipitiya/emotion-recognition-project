#!/bin/bash

# Emotion Detection Web App Startup Script
echo "🎭 Starting Emotion Detection Web App"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "env" ]; then
    echo "❌ Virtual environment not found. Please run the setup first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source env/bin/activate

# Check if model exists
if [ ! -f "model1_best.h5" ]; then
    echo "❌ Model file not found: model1_best.h5"
    echo "Please make sure the model file is in the current directory"
    exit 1
fi

# Check if Flask is installed
echo "🔍 Checking dependencies..."
python -c "import flask, tensorflow, cv2, PIL, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing missing dependencies..."
    pip install flask opencv-python pillow
fi

echo "✅ All dependencies ready"
echo ""
echo "🚀 Starting Flask server..."
echo "📱 Open your browser and go to: http://localhost:8080"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Start the Flask app
python app.py
