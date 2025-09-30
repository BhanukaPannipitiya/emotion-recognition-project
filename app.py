from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import logging
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image, ImageOps
import os
import math
import mediapipe as mp
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg', 'webp', 'gif' }
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# TensorFlow runtime constraints for small instances
try:
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    logger.info("Configured TensorFlow to use CPU-only with 1 intra/inter op threads")
except Exception as _tf_cfg_err:
    logger.warning("Could not fully configure TensorFlow threading/visibility: %s", _tf_cfg_err)

# Reduce OpenCV threading to avoid CPU contention on small instances
try:
    cv2.setNumThreads(1)
except Exception:
    pass

@app.before_request
def log_request_info():
    try:
        logger.info("%s %s - headers=%s", request.method, request.path, dict(request.headers))
    except Exception:
        logger.info("%s %s", request.method, request.path)

# Lazy model loading to avoid blocking startup and static file serving
model = None
_model_lock = threading.Lock()
_model_warmup_started = False

def _warmup_model():
    try:
        m = get_model()
        dummy = np.zeros((1, 48, 48, 1), dtype=np.float32)
        _ = m.predict(dummy, verbose=0)
        logger.info("Model warm-up prediction completed")
    except Exception as _warm_err:
        logger.warning("Model warm-up failed: %s", _warm_err)

def get_model():
    global model, _model_warmup_started
    if model is None:
        with _model_lock:
            if model is None:
                logger.info("Lazy-loading model1_best.h5...")
                local_model = load_model('model1_best.h5')
                model_ref = local_model
                # Assign only after successful load to avoid half-initialized state
                globals()['model'] = model_ref
                logger.info("Model loaded successfully!")
                if not _model_warmup_started:
                    _model_warmup_started = True
                    threading.Thread(target=_warmup_model, daemon=True).start()
    return model

# Define classes and image size
classes = ['angry', 'happy', 'neutral']
img_size = (48, 48)

# MediaPipe face detector (CPU-friendly and robust)
mp_face = mp.solutions.face_detection
_mp_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# OpenCV Haar Cascade fallback (robust on some edge cases and small faces)
try:
    _haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    _haar = None

def _detect_faces_mediapipe(image_bgr):
    """Detect faces and return list of [x, y, w, h] in pixel coords.
    Uses MediaPipe FaceDetection on RGB input internally.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = _mp_detector.process(image_rgb)
    boxes = []
    if results.detections:
        h, w = image_bgr.shape[:2]
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            bw = max(1, int(bbox.width * w))
            bh = max(1, int(bbox.height * h))
            boxes.append([x, y, bw, bh])
    return boxes

def _detect_faces_haar(image_bgr):
    """Detect faces using OpenCV Haar cascade. Returns list of [x, y, w, h]."""
    if _haar is None:
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Equalize to improve detection in varying lighting
    gray = cv2.equalizeHist(gray)
    faces = _haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    boxes = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in faces]
    return boxes

def _resize_long_side(image_bgr, max_side=1280):
    """Downscale extremely large images for faster/more robust detection, keep aspect ratio."""
    h, w = image_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    return image_bgr, 1.0

def detect_faces(image_bgr):
    """Try multiple detectors and return face boxes in original image coordinates."""
    # Work on a smaller copy for speed if very large
    work_img, scale = _resize_long_side(image_bgr, max_side=1280)
    boxes = _detect_faces_mediapipe(work_img)
    if not boxes:
        boxes = _detect_faces_haar(work_img)
    # Map boxes back to original coordinates if resized
    if scale != 1.0 and boxes:
        inv = 1.0 / scale
        boxes = [[int(x * inv), int(y * inv), int(w * inv), int(h * inv)] for x, y, w, h in boxes]
    return boxes

def _expand_box(x, y, w, h, img_w, img_h, margin=0.2):
    """Expand box by margin while clamping to image bounds."""
    cx = x + w / 2.0
    cy = y + h / 2.0
    size = max(w, h)
    size = size * (1.0 + margin)
    nx = int(max(0, math.floor(cx - size / 2.0)))
    ny = int(max(0, math.floor(cy - size / 2.0)))
    nsize = int(min(size, min(img_w - nx, img_h - ny)))
    return nx, ny, nsize, nsize

def preprocess_image(image_gray):
    """Preprocess face ROI for model prediction: CLAHE + resize + normalize."""
    if len(image_gray.shape) == 3:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
    # CLAHE for lighting robustness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_eq = clahe.apply(image_gray)
    # Resize to model input size
    image_resized = cv2.resize(image_eq, img_size)
    image_norm = image_resized.astype(np.float32) / 255.0
    image_norm = np.expand_dims(image_norm, axis=-1)
    image_norm = np.expand_dims(image_norm, axis=0)
    return image_norm

def predict_emotion(image):
    """Predict emotion from preprocessed image"""
    m = get_model()
    predictions = m.predict(image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    emotion = classes[predicted_class]
    
    return emotion, confidence, predictions[0].tolist()

def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    logger.info("GET / - rendering index.html")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            logger.warning("/upload called without 'file' in form-data")
            return jsonify({'success': False, 'error': "Missing 'file' in form-data"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not _allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Unsupported file type'}), 400

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        logger.info("/upload saved file to %s", save_path)

        # Open and process image (handle EXIF rotation)
        image = Image.open(save_path)
        image = ImageOps.exif_transpose(image)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        faces = detect_faces(image_cv)
        results = []
        if len(faces) > 0:
            h_img, w_img = image_cv.shape[:2]
            faces_sorted = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
            x, y, w, h = faces_sorted[0]
            x, y, w, h = _expand_box(x, y, w, h, w_img, h_img, margin=0.2)
            face_roi_color = image_cv[y:y+h, x:x+w]
            processed_face = preprocess_image(face_roi_color)
            emotion, confidence, all_predictions = predict_emotion(processed_face)
            results.append({
                'emotion': emotion,
                'confidence': float(confidence),
                'all_predictions': dict(zip(classes, all_predictions)),
                'face_box': [int(x), int(y), int(w), int(h)]
            })
        else:
            # Final fallback: centered square crop to always produce a result
            h_img, w_img = image_cv.shape[:2]
            side = int(min(h_img, w_img) * 0.6)
            cx, cy = w_img // 2, h_img // 2
            x = max(0, cx - side // 2)
            y = max(0, cy - side // 2)
            x = min(x, w_img - side)
            y = min(y, h_img - side)
            face_roi_color = image_cv[y:y+side, x:x+side]
            processed_face = preprocess_image(face_roi_color)
            emotion, confidence, all_predictions = predict_emotion(processed_face)
            results.append({
                'emotion': emotion,
                'confidence': float(confidence),
                'all_predictions': dict(zip(classes, all_predictions)),
                'face_box': [int(x), int(y), int(side), int(side)]
            })

        return jsonify({'success': True, 'results': results, 'filename': filename})
    except Exception as e:
        logger.exception("/upload failed: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.get_json(silent=True) or {}
        if 'image' not in data:
            logger.warning("/predict called without 'image' in JSON body")
            return jsonify({'success': False, 'error': "Missing 'image' in request body"}), 400
        image_data = data['image']
        logger.info("/predict received image payload size=%s", len(image_data) if isinstance(image_data, str) else 'N/A')
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        logger.info("/predict step: decoding base64")
        image_bytes = base64.b64decode(image_data)
        logger.info("/predict step: opening image with PIL (%d bytes)", len(image_bytes))
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        logger.info("/predict step: PIL image size=%sx%s mode=%s", image.width, image.height, image.mode)
        
        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        logger.info("/predict step: converted to OpenCV BGR shape=%s", tuple(image_cv.shape))
        
        # Detect faces using cascaded detectors
        faces = detect_faces(image_cv)
        logger.info("/predict step: detected %d face(s)", len(faces))
        
        results = []
        
        if len(faces) > 0:
            # Pick largest face and add margin
            h_img, w_img = image_cv.shape[:2]
            faces_sorted = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
            x, y, w, h = faces_sorted[0]
            x, y, w, h = _expand_box(x, y, w, h, w_img, h_img, margin=0.2)
            # Extract face region
            face_roi_color = image_cv[y:y+h, x:x+w]
            processed_face = preprocess_image(face_roi_color)
            logger.info("/predict step: running model.predict for face crop")
            emotion, confidence, all_predictions = predict_emotion(processed_face)
            results.append({
                'emotion': emotion,
                'confidence': float(confidence),
                'all_predictions': dict(zip(classes, all_predictions)),
                'face_box': [int(x), int(y), int(w), int(h)]
            })
        else:
            logger.info("/predict step: no faces detected - using center crop fallback")
            h_img, w_img = image_cv.shape[:2]
            side = int(min(h_img, w_img) * 0.6)
            cx, cy = w_img // 2, h_img // 2
            x = max(0, cx - side // 2)
            y = max(0, cy - side // 2)
            x = min(x, w_img - side)
            y = min(y, h_img - side)
            face_roi_color = image_cv[y:y+side, x:x+side]
            processed_face = preprocess_image(face_roi_color)
            emotion, confidence, all_predictions = predict_emotion(processed_face)
            results.append({
                'emotion': emotion,
                'confidence': float(confidence),
                'all_predictions': dict(zip(classes, all_predictions)),
                'face_box': [int(x), int(y), int(side), int(side)]
            })
        
        logger.info("/predict returning %d result(s)", len(results))
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.exception("/predict failed: %s", e)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify the model is working"""
    try:
        logger.info("GET /test - generating synthetic prediction")
        # Create a test image (random noise)
        test_image = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        processed_image = preprocess_image(test_image)
        emotion, confidence, all_predictions = predict_emotion(processed_image)
        
        resp = {
            'success': True,
            'message': 'Model is working correctly',
            'test_prediction': {
                'emotion': emotion,
                'confidence': float(confidence),
                'all_predictions': dict(zip(classes, all_predictions))
            }
        }
        logger.info("/test ok - emotion=%s conf=%.3f", emotion, float(confidence))
        return jsonify(resp)
    except Exception as e:
        logger.exception("/test failed: %s", e)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    port = int(os.environ.get('PORT', 8080))
    print("Starting Flask server...")
    print(f"Open your browser and go to: http://localhost:{port}")
    # In production on Render, PORT is provided; disable debug
    app.run(debug=bool(os.environ.get('FLASK_DEBUG', '1') == '1'), host='0.0.0.0', port=port)
