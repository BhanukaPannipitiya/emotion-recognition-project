### Emotion Recognition Web App

An end-to-end emotion recognition web application built with Flask, TensorFlow/Keras, OpenCV, and MediaPipe. It supports image uploads and live webcam capture, detects faces, and predicts emotions using a trained model (`model1_best.h5`).

---

### Features
- Predicts emotions: angry, happy, neutral
- Two input modes: image upload and webcam capture
- Face detection via MediaPipe with OpenCV Haar-cascade fallback
- Confidence scores and per-class probabilities
- Production-ready config for Render (Gunicorn, `render.yaml`, `Procfile`)

---

### Project Structure
```
emotion-recognition-project/
├── app.py                 # Flask backend and ML inference
├── run_app.py             # Convenience launcher for local dev
├── start_app.sh           # Bash startup helper (uses venv)
├── model1_best.h5         # Trained Keras model (not tracked if large)
├── requirements.txt       # Python dependencies (CPU-friendly builds)
├── Procfile               # Gunicorn start command (Render/Procfile-based hosts)
├── render.yaml            # Render blueprint (Infra as code)
├── runtime.txt            # Pin Python version for Render
├── templates/
│   └── index.html         # Frontend UI
├── static/
│   └── styles.css         # Styling
└── uploads/               # Uploaded images (created at runtime)
```

---

### Prerequisites
- Python 3.11 (project pins `3.11.9` for Render)
- pip
- macOS/Linux/WSL. Windows works but commands may differ slightly

Optional but recommended for local:
- Virtual environment (venv)

---

### Setup
1) Clone the repository and switch to the project directory.
2) (Recommended) Create and activate a virtual environment:

```bash
python3 -m venv env
source env/bin/activate   # macOS/Linux
# On Windows PowerShell: .\\env\\Scripts\\Activate.ps1
```

3) Install dependencies:

```bash
pip install -r requirements.txt
```

4) Place the trained model file in the project root:

```text
model1_best.h5
```

---

### Running Locally
Choose one of the options below.

Option A: Convenience launcher
```bash
python run_app.py
```
This checks deps and the model file, then starts the server (defaults to port 5000 inside `run_app.py`).

Option B: Direct Flask entrypoint
```bash
python app.py
```
`app.py` reads `PORT` from the environment (defaults to 8080) and will print a local URL. Example:

```bash
PORT=8080 python app.py
```

Option C: Bash helper (if you created `env/` venv via Setup)
```bash
bash start_app.sh
```

Once running, open your browser to the printed URL. By default:
- `run_app.py`: http://localhost:5000
- `app.py` default: http://localhost:8080

---

### Using the App
- Upload tab: drag-and-drop or choose an image (JPG/PNG/WEBP/GIF, < 5MB) and click "Analyze Emotion".
- Webcam tab: click "Start Webcam", then "Capture & Analyze".
- The result panel shows the predicted emotion, confidence, and all class probabilities. A face box overlays uploaded images when a face is detected.

Supported classes are defined in `app.py` (`classes = ['angry', 'happy', 'neutral']`). To add more, retrain your model and update this list and any UI labels.

---

### API Endpoints
- `GET /` — Serves the `index.html` UI
- `POST /upload` — Multipart form upload; field name: `file`
  - Response: JSON `{ success, results: [{ emotion, confidence, all_predictions, face_box }], filename }`
- `POST /predict` — JSON body `{ image: "data:image/jpeg;base64,..." }`
  - Response: JSON `{ success, results: [...] }`
- `GET /test` — Health check generating a synthetic prediction

Notes:
- Images are preprocessed to 48x48 grayscale with CLAHE and normalization to match the model.
- Face detection prefers MediaPipe and falls back to OpenCV Haar cascade. If no face is found, a centered crop is analyzed to ensure a prediction is always returned.

---

### Environment and Performance
- TensorFlow is configured to run CPU-only with limited threads for small instances
- OpenCV thread count is set to 1 to reduce contention
- `opencv-python-headless` is used for server environments without GUI

You can tune performance via environment variables:
- `PORT`: server port (used by `app.py`; defaults to 8080)
- `FLASK_DEBUG`: set `1` for debug locally; Render sets it to `0`

---

### Deployment (Render)
This repo includes a ready-to-use Render blueprint and process files.

Files:
- `render.yaml` — defines the web service and start command
- `Procfile` — Gunicorn process declaration
- `runtime.txt` — pins Python 3.11.9

Steps:
1) Push the project to GitHub (ensure `model1_best.h5` is in the repo or downloaded during build).
2) On Render, choose New → Blueprint and select your repo.
3) Confirm service details and create resources.
4) Render builds (`pip install -r requirements.txt`) and starts:
   ```
   gunicorn app:app --workers=1 --threads=1 --timeout=120 --bind 0.0.0.0:$PORT
   ```
5) Open the Render-provided URL.

If your model is large, host it and download during build. Example snippet for `render.yaml`:
```yaml
buildCommand: |
  pip install -r requirements.txt
  curl -L -o model1_best.h5 "https://YOUR_HOSTED_URL/model1_best.h5"
```

---

### Customization
- Change classes: edit `classes` in `app.py` and retrain/update the model
- Styling: tweak `static/styles.css`
- Frontend behavior: edit `templates/index.html`

---

### Troubleshooting
- Missing model: ensure `model1_best.h5` is in the project root
- Import errors: run `pip install -r requirements.txt` inside your venv
- Port in use: set a different `PORT` or stop the conflicting service
- macOS gatekeeper for venv activation: run `chmod +x start_app.sh` or use `source env/bin/activate`
- OpenCV on headless servers: use `opencv-python-headless` (already in `requirements.txt`)

Health check: `GET /test` returns a JSON indicating the model is working.

---

### License
This repository is for educational/demo purposes. Add your license of choice here.


