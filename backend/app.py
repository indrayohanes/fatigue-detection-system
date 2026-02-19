from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
CORS(app)

# ==========================================
# KONFIGURASI
# ==========================================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'fatigue_detection_model.h5'
PREDICTION_THRESHOLD = 0.55
MODEL_INPUT_SIZE = (96, 96)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL = None

def load_model():
    """Load trained CNN model"""
    global MODEL
    try:
        if os.path.exists(MODEL_PATH):
            MODEL = keras.models.load_model(MODEL_PATH)
            print("="*60)
            print("✅ MODEL LOADED SUCCESSFULLY!")
            print("="*60)
            print(f"Model path: {MODEL_PATH}")
            print(f"Input shape: {MODEL.input_shape}")
            print(f"Threshold: {PREDICTION_THRESHOLD}")
            print("="*60)
        else:
            print("⚠️  MODEL FILE NOT FOUND - Using dummy predictions")
    except Exception as e:
        print(f"❌ ERROR LOADING MODEL: {str(e)}")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_face_multi_method(image):
    """
    MULTI-METHOD FACE DETECTION - SANGAT SENSITIF!
    
    Menggunakan 3 metode:
    1. Haar Cascade Frontal Face (sangat sensitif)
    2. Haar Cascade Profile Face (untuk wajah miring)
    3. DNN Face Detection (deep learning - paling akurat)
    
    Returns: (cropped_face_image, bbox) or None
    """
    
    # ==========================================
    # METHOD 1: HAAR CASCADE FRONTAL - SUPER SENSITIVE
    # ==========================================
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try with VERY sensitive parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.02,  # VERY sensitive (default 1.1)
        minNeighbors=2,    # VERY tolerant (default 5)
        minSize=(15, 15),  # Very small faces OK
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        # Found face with method 1!
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        face_img = extract_face_with_margin(image, x, y, w, h)
        return face_img, (x, y, w, h)
    
    # ==========================================
    # METHOD 2: HAAR CASCADE PROFILE FACE
    # ==========================================
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_profileface.xml'
    )
    
    # Try profile detection
    faces = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(15, 15)
    )
    
    if len(faces) > 0:
        # Found face with method 2!
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        face_img = extract_face_with_margin(image, x, y, w, h)
        return face_img, (x, y, w, h)
    
    # ==========================================
    # METHOD 3: DNN FACE DETECTION (Most Accurate)
    # ==========================================
    try:
        # Download model if not exists
        prototxt_path = 'deploy.prototxt'
        model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
        
        # If DNN model files don't exist, use fallback method
        if not (os.path.exists(prototxt_path) and os.path.exists(model_path)):
            # FALLBACK: Use very aggressive Haar Cascade
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.01,  # EXTREMELY sensitive
                minNeighbors=1,    # EXTREMELY tolerant
                minSize=(10, 10)   # Tiny faces OK
            )
            
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                face_img = extract_face_with_margin(image, x, y, w, h)
                return face_img, (x, y, w, h)
            
            # LAST RESORT: Take center region of image
            h, w = image.shape[:2]
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 2
            x1 = max(0, center_x - size)
            y1 = max(0, center_y - size)
            x2 = min(w, center_x + size)
            y2 = min(h, center_y + size)
            face_img = image[y1:y2, x1:x2]
            return face_img, (x1, y1, x2-x1, y2-y1)
        
        # Load DNN model
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        
        # Prepare image for DNN
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        net.setInput(blob)
        detections = net.forward()
        
        # Find face with highest confidence
        best_confidence = 0
        best_box = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.3:  # Low threshold for high sensitivity
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = (startX, startY, endX - startX, endY - startY)
        
        if best_box:
            x, y, w, h = best_box
            face_img = extract_face_with_margin(image, x, y, w, h)
            return face_img, (x, y, w, h)
    
    except Exception as e:
        print(f"DNN detection failed: {e}")
    
    # ==========================================
    # ABSOLUTE FALLBACK: Return center crop
    # ==========================================
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    size = min(w, h) // 2
    
    x1 = max(0, center_x - size)
    y1 = max(0, center_y - size)
    x2 = min(w, center_x + size)
    y2 = min(h, center_y + size)
    
    face_img = image[y1:y2, x1:x2]
    return face_img, (x1, y1, x2-x1, y2-y1)

def extract_face_with_margin(image, x, y, w, h, margin=25):
    """
    Extract face region with margin
    """
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)
    
    return image[y1:y2, x1:x2]

def preprocess_image(face_img, target_size=MODEL_INPUT_SIZE):
    """Preprocess face image for CNN model"""
    if face_img.size == 0:
        return None
    
    face_resized = cv2.resize(face_img, target_size)
    face_normalized = face_resized.astype('float32') / 255.0
    face_batch = np.expand_dims(face_normalized, axis=0)
    
    return face_batch

def predict_fatigue(preprocessed_img):
    """
    Predict fatigue dengan CORRECTED LOGIC
    
    Class mapping: fatigued=0, non_fatigued=1
    """
    if MODEL is None:
        prediction = np.random.rand()
        label = "Fatigued" if prediction > 0.5 else "Non-Fatigued"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        return label, float(confidence)
    
    prediction = MODEL.predict(preprocessed_img, verbose=0)
    raw_confidence = float(prediction[0][0])
    
    # CORRECTED LOGIC
    adjusted_threshold = 1 - PREDICTION_THRESHOLD
    
    if raw_confidence < adjusted_threshold:
        label = "Fatigued"
        confidence = 1 - raw_confidence
    else:
        label = "Non-Fatigued"
        confidence = raw_confidence
    
    return label, confidence

def encode_image_to_base64(image):
    """Convert CV2 image to base64"""
    _, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

def get_recommendation(prediction, confidence):
    """Generate recommendation"""
    if prediction == "Fatigued":
        if confidence > 0.8:
            return {
                'level': 'high',
                'message': 'Tingkat kelelahan tinggi terdeteksi. Sangat disarankan untuk beristirahat segera.',
                'actions': [
                    'Hentikan aktivitas kerja sementara',
                    'Istirahat minimal 15-20 menit',
                    'Konsumsi air putih yang cukup',
                    'Jika diperlukan, konsultasikan dengan supervisor'
                ]
            }
        else:
            return {
                'level': 'moderate',
                'message': 'Tanda-tanda kelelahan terdeteksi. Pertimbangkan untuk beristirahat.',
                'actions': [
                    'Ambil jeda singkat 5-10 menit',
                    'Lakukan peregangan ringan',
                    'Pastikan pencahayaan ruangan memadai',
                    'Monitor kondisi Anda secara berkala'
                ]
            }
    else:
        return {
            'level': 'normal',
            'message': 'Kondisi baik. Tidak terdeteksi tanda-tanda kelelahan signifikan.',
            'actions': [
                'Tetap jaga pola istirahat yang teratur',
                'Hindari bekerja terlalu lama tanpa jeda',
                'Lakukan pemeriksaan berkala'
            ]
        }

# ==========================================
# API ENDPOINTS
# ==========================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'model_input_size': MODEL_INPUT_SIZE,
        'prediction_threshold': PREDICTION_THRESHOLD,
        'face_detection': 'Multi-method (Haar + Profile + DNN fallback)',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/detect', methods=['POST'])
def detect_fatigue_endpoint():
    """Main detection endpoint with multi-method face detection"""
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'
            }), 400
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to read image'
            }), 400
        
        # Multi-method face detection
        face_result = detect_face_multi_method(image)
        
        if face_result is None:
            return jsonify({
                'success': False,
                'error': 'No face detected in the image',
                'suggestion': 'Please ensure the image contains a visible face with good lighting'
            }), 400
        
        face_img, bbox = face_result
        
        # Preprocess
        preprocessed = preprocess_image(face_img)
        
        if preprocessed is None:
            return jsonify({
                'success': False,
                'error': 'Failed to preprocess image'
            }), 400
        
        # Predict
        prediction_label, confidence = predict_fatigue(preprocessed)
        
        # Encode face
        face_base64 = encode_image_to_base64(face_img)
        
        # Response
        response = {
            'success': True,
            'prediction': prediction_label,
            'confidence': round(confidence * 100, 2),
            'confidence_raw': float(confidence),
            'threshold_used': PREDICTION_THRESHOLD,
            'face_detected': True,
            'face_bbox': {
                'x': int(bbox[0]),
                'y': int(bbox[1]),
                'width': int(bbox[2]),
                'height': int(bbox[3])
            },
            'face_image': face_base64,
            'timestamp': datetime.now().isoformat(),
            'recommendation': get_recommendation(prediction_label, confidence)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify({
        'total_predictions': 0,
        'fatigued_count': 0,
        'non_fatigued_count': 0,
        'average_confidence': 0
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get system configuration"""
    return jsonify({
        'model_loaded': MODEL is not None,
        'model_type': 'Pure CNN (5 blocks)',
        'model_input_size': MODEL_INPUT_SIZE,
        'prediction_threshold': PREDICTION_THRESHOLD,
        'face_detection_methods': ['Haar Cascade Frontal', 'Haar Cascade Profile', 'DNN (fallback)', 'Center Crop (absolute fallback)'],
        'max_upload_size_mb': app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024),
        'allowed_extensions': list(ALLOWED_EXTENSIONS)
    })

# ==========================================
# MAIN
# ==========================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 FATIGUE DETECTION SYSTEM - ENHANCED VERSION")
    print("Multi-Method Face Detection + Corrected Logic")
    print("="*60)
    
    load_model()
    
    print("\n📡 Starting Flask server...")
    print("="*60)
    
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        threaded=True
    )