#!/usr/bin/env python3
"""
Startup script for the Emotion Detection Web App
This script will start the Flask server and open the web app in your browser.
"""

import subprocess
import sys
import webbrowser
import time
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import tensorflow
        import cv2
        import PIL
        import numpy
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_model():
    """Check if the model file exists"""
    model_path = Path("model1_best.h5")
    if model_path.exists():
        print("✅ Model file found: model1_best.h5")
        return True
    else:
        print("❌ Model file not found: model1_best.h5")
        print("Please make sure the model file is in the current directory")
        return False

def main():
    print("🎭 Emotion Detection Web App")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check model
    if not check_model():
        sys.exit(1)
    
    print("\n🚀 Starting the web application...")
    print("📱 The app will open in your browser at: http://localhost:8080")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 40)
    
    # Start the Flask app
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
