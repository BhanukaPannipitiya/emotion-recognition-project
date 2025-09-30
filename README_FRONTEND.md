# 🎭 Emotion Detection Web App

A real-time emotion detection web application powered by your trained `model1_best.h5` model.

## 🚀 Quick Start

### Option 1: Using the startup script (Recommended)
```bash
python run_app.py
```

### Option 2: Manual start
```bash
# Install requirements (if not already installed)
pip install -r requirements.txt

# Start the Flask server
python app.py
```

Then open your browser and go to: **http://localhost:5000**

## 📁 Project Structure

```
emotion-recognition-project/
├── app.py                 # Flask backend server
├── run_app.py            # Easy startup script
├── model1_best.h5        # Your trained Keras model
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Frontend HTML
├── static/
│   └── styles.css        # Frontend CSS
└── README_FRONTEND.md    # This file
```

## 🎯 Features

- **Real-time emotion detection** using your webcam
- **Face detection** with bounding boxes
- **Live confidence scores** for all emotions
- **Beautiful, responsive UI** that works on desktop and mobile
- **Three emotion classes**: Angry, Happy, Neutral
- **Capture photos** for analysis
- **Model status indicators**

## 🎮 How to Use

1. **Start the app** using one of the methods above
2. **Click "Start Camera"** to begin real-time detection
3. **Look at the camera** - the app will detect your face and predict emotions
4. **Use "Capture Photo"** to take a snapshot for analysis
5. **Click "Stop Camera"** when done

## 🔧 Technical Details

### Backend (Flask)
- Serves the Keras model via REST API
- Handles image preprocessing (48x48 grayscale)
- Face detection using OpenCV
- Returns emotion predictions with confidence scores

### Frontend (HTML/CSS/JavaScript)
- Real-time video capture from webcam
- Sends frames to backend for analysis
- Displays results with animated progress bars
- Responsive design for all screen sizes

### Model Integration
- Loads `model1_best.h5` on startup
- Preprocesses images to match training format
- Returns predictions for: angry, happy, neutral

## 🛠️ Troubleshooting

### Camera Issues
- **Permission denied**: Allow camera access in your browser
- **No video**: Check if another app is using the camera
- **Poor detection**: Ensure good lighting and face visibility

### Model Issues
- **Model not loading**: Ensure `model1_best.h5` is in the project root
- **Prediction errors**: Check that the model was trained on the same classes

### Server Issues
- **Port 5000 in use**: Change the port in `app.py` (line 95)
- **Dependencies missing**: Run `pip install -r requirements.txt`

## 📱 Browser Compatibility

- ✅ Chrome (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge
- ❌ Internet Explorer

## 🔒 Privacy & Security

- All processing happens locally on your machine
- No data is sent to external servers
- Camera access is only used for emotion detection
- Images are not stored permanently

## 🎨 Customization

### Changing the Model
Replace `model1_best.h5` with your own trained model and update the classes in `app.py` (line 15).

### Styling
Modify `static/styles.css` to change the appearance of the web app.

### Adding New Emotions
1. Retrain your model with new classes
2. Update the `classes` list in `app.py`
3. Update the frontend in `templates/index.html`

## 📊 Performance

- **Model loading**: ~2-3 seconds on startup
- **Prediction speed**: ~50-100ms per frame
- **Memory usage**: ~200-300MB (including TensorFlow)
- **CPU usage**: Moderate (depends on your hardware)

## 🆘 Support

If you encounter any issues:
1. Check the console output for error messages
2. Ensure all dependencies are installed
3. Verify the model file exists and is valid
4. Try refreshing the browser page

---

**Enjoy your emotion detection app! 🎉**
