from flask import Flask, render_template, request, jsonify
import logging
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image
import os

app = Flask(__name__)
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

# Load the model
print("Loading model1_best.h5...")
model = load_model('model1_best.h5')
print("Model loaded successfully!")

# Warm-up model to avoid first-request stall/compile overhead
try:
    dummy = np.zeros((1, 48, 48, 1), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)
    logger.info("Model warm-up prediction completed")
except Exception as _warm_err:
    logger.warning("Model warm-up failed: %s", _warm_err)

# Define classes and image size
classes = ['angry', 'happy', 'neutral']
img_size = (48, 48)

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to model input size
    image = cv2.resize(image, img_size)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Add channel dimension and batch dimension
    image = np.expand_dims(image, axis=-1)  # (48, 48, 1)
    image = np.expand_dims(image, axis=0)   # (1, 48, 48, 1)
    
    return image

def predict_emotion(image):
    """Predict emotion from preprocessed image"""
    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    emotion = classes[predicted_class]
    
    return emotion, confidence, predictions[0].tolist()

@app.route('/')
def index():
    logger.info("GET / - rendering index.html")
    return render_template('index.html')

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
        logger.info("/predict step: PIL image size=%sx%s mode=%s", image.width, image.height, image.mode)
        
        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        logger.info("/predict step: converted to OpenCV BGR shape=%s", tuple(image_cv.shape))
        
        # Detect faces
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        logger.info("/predict step: grayscale shape=%s", tuple(gray.shape))
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        logger.info("/predict step: detected %d face(s)", 0 if faces is None else len(faces))
        
        results = []
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Preprocess face for emotion prediction
                processed_face = preprocess_image(face_roi)
                
                # Predict emotion
                logger.info("/predict step: running model.predict for detected face")
                emotion, confidence, all_predictions = predict_emotion(processed_face)
                
                results.append({
                    'emotion': emotion,
                    'confidence': float(confidence),
                    'all_predictions': dict(zip(classes, all_predictions)),
                    'face_box': [int(x), int(y), int(w), int(h)]
                })
        else:
            # If no face detected, process the entire image
            logger.info("/predict step: no faces, predicting on full frame")
            processed_image = preprocess_image(gray)
            logger.info("/predict step: running model.predict for full frame")
            emotion, confidence, all_predictions = predict_emotion(processed_image)
            
            results.append({
                'emotion': emotion,
                'confidence': float(confidence),
                'all_predictions': dict(zip(classes, all_predictions)),
                'face_box': None
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
