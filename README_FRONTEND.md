### Emotion Detection Web App — Frontend Guide

This guide explains how to use and customize the UI (`templates/index.html`, `static/styles.css`) for the Flask-based emotion detection app.

---

### Quick Start
Start the server using one of the following and open the printed URL.

Option A:
```bash
python run_app.py
```

Option B:
```bash
pip install -r requirements.txt
python app.py    # respects PORT env; defaults to 8080
```

Option C:
```bash
bash start_app.sh
```

Default URLs:
- `run_app.py`: http://localhost:5000
- `app.py` default: http://localhost:8080 (or your `PORT`)

---

### UI Overview
- Tabs: Upload and Webcam
- Results: predicted emotion, confidence bar, all class probabilities
- Face box overlay on uploaded images when detection succeeds

Supported classes (backend): `angry`, `happy`, `neutral`.

---

### How to Use
Upload tab:
1. Drag-and-drop or click to select an image (JPG/PNG/WEBP/GIF, < 5MB)
2. Click "Analyze Emotion"
3. View prediction, confidence, probabilities, and face box

Webcam tab:
1. Click "Start Webcam" and allow camera access
2. Click "Capture & Analyze" to send a snapshot to the backend
3. View prediction and probabilities

---

### Frontend–Backend Contract
Endpoints used by the UI:
- `POST /upload` — multipart form; field `file`
- `POST /predict` — JSON `{ image: "data:image/jpeg;base64,..." }`

Response shape (simplified):
```json
{
  "success": true,
  "results": [
    {
      "emotion": "happy",
      "confidence": 0.92,
      "all_predictions": { "angry": 0.01, "happy": 0.92, "neutral": 0.07 },
      "face_box": [x, y, w, h]
    }
  ]
}
```

If no face is detected, the backend analyzes a centered crop and still returns a prediction.

---

### Customization
- Change classes: update `classes` in `app.py` and the UI labels/icons in `templates/index.html`.
- Styling: edit `static/styles.css` for colors, layout, animations.
- Icons: Font Awesome is loaded via CDN in `index.html`.
- Canvas overlay: bounding box drawing uses `previewCanvas` in `index.html` script.

---

### Troubleshooting
- Camera permission denied: allow access in the browser.
- No video: ensure no other app uses the camera; try another browser.
- Large images: the backend resizes internally; ensure file < 5MB.
- Missing model: place `model1_best.h5` in the project root.

Browser support: Chrome, Firefox, Safari, Edge.

---

### Notes for Developers
- Preprocessing: CLAHE + resize to 48×48 grayscale + normalization (handled server-side).
- Detection: MediaPipe first, Haar cascade fallback.
- Performance: CPU-only TF with limited threads; OpenCV threads set to 1.

For deeper backend details, see `README.md` and `app.py`.
