from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image
import os

app = Flask(__name__)

# Load the model
print("Loading model1_best.h5...")
model = load_model('model1_best.h5')
print("Model loaded successfully!")

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
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image']
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        results = []
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Preprocess face for emotion prediction
                processed_face = preprocess_image(face_roi)
                
                # Predict emotion
                emotion, confidence, all_predictions = predict_emotion(processed_face)
                
                results.append({
                    'emotion': emotion,
                    'confidence': float(confidence),
                    'all_predictions': dict(zip(classes, all_predictions)),
                    'face_box': [int(x), int(y), int(w), int(h)]
                })
        else:
            # If no face detected, process the entire image
            processed_image = preprocess_image(gray)
            emotion, confidence, all_predictions = predict_emotion(processed_image)
            
            results.append({
                'emotion': emotion,
                'confidence': float(confidence),
                'all_predictions': dict(zip(classes, all_predictions)),
                'face_box': None
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify the model is working"""
    try:
        # Create a test image (random noise)
        test_image = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        processed_image = preprocess_image(test_image)
        emotion, confidence, all_predictions = predict_emotion(processed_image)
        
        return jsonify({
            'success': True,
            'message': 'Model is working correctly',
            'test_prediction': {
                'emotion': emotion,
                'confidence': float(confidence),
                'all_predictions': dict(zip(classes, all_predictions))
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    port = int(os.environ.get('PORT', 8080))
    print("Starting Flask server...")
    print(f"Open your browser and go to: http://localhost:{port}")
    # In production on Render, PORT is provided; disable debug
    app.run(debug=bool(os.environ.get('FLASK_DEBUG', '1') == '1'), host='0.0.0.0', port=port)
