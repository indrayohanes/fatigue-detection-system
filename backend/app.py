from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
import sqlite3
import hashlib
import uuid
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, send_from_directory
import base64
from io import BytesIO
from PIL import Image

# TensorFlow is optional - app works without it using dummy predictions
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    print("✅ TensorFlow loaded successfully")
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None
    print("⚠️  TensorFlow not available - will use dummy predictions")

app = Flask(__name__)
CORS(app)

# ==========================================
# KONFIGURASI
# ==========================================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'fatigue_detection_model.h5'
PREDICTION_THRESHOLD = 0.5
MODEL_INPUT_SIZE = (96, 96)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL = None
DB_PATH = 'fatigue_detection.db'

# Session storage (in-memory)
SESSIONS = {}

# ==========================================
# DATABASE
# ==========================================

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    return conn

def init_db():
    """Initialize database tables"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            face_image TEXT,
            gradcam_image TEXT,
            face_regions TEXT,
            explanation TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def hash_password(password):
    """Hash password with SHA-256 + salt"""
    salt = uuid.uuid4().hex
    return salt + ':' + hashlib.sha256((salt + password).encode()).hexdigest()

def verify_password(stored_hash, password):
    """Verify password against stored hash"""
    salt = stored_hash.split(':')[0]
    return stored_hash == salt + ':' + hashlib.sha256((salt + password).encode()).hexdigest()

def get_current_user():
    """Get current user from Authorization header"""
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]
        if token in SESSIONS:
            return SESSIONS[token]
    return None

def load_model():
    """Load trained CNN model"""
    global MODEL
    if not TF_AVAILABLE:
        print("⚠️  TensorFlow not available - Using dummy predictions")
        return
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

def is_valid_face(x, y, w, h, img_h, img_w, min_ratio=0.10):
    """
    Validate if a detected region is likely a real face.
    - Aspect ratio check (faces are roughly square, 0.4 to 2.5)
    - Minimum size relative to image (at least min_ratio of image dimension)
    - Accept faces ANYWHERE in frame (no position restriction)
    """
    aspect_ratio = w / h if h > 0 else 0
    if aspect_ratio < 0.4 or aspect_ratio > 2.5:
        return False
    
    # Minimum size: face should be at least min_ratio of image width or height
    min_size = min(img_w, img_h) * min_ratio
    if w < min_size or h < min_size:
        return False
    
    # Accept faces anywhere in the frame — no position restriction
    # This allows edge/corner faces from webcam to be detected
    return True

def select_best_face(faces, img_h, img_w, min_ratio=0.10):
    """
    From a list of detected faces, select the best one.
    Priority: valid face closest to center-top of image + largest.
    """
    valid_faces = []
    for (x, y, w, h) in faces:
        if is_valid_face(x, y, w, h, img_h, img_w, min_ratio):
            # Score: prefer larger faces that are centered and towards top
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Distance from horizontal center (normalized)
            dist_x = abs(center_x - img_w / 2) / img_w
            # Prefer upper part of image (lower y = better)
            dist_y = center_y / img_h
            
            # Size score (bigger = better)
            size_score = (w * h) / (img_w * img_h)
            
            # Combined score: size is most important, position has minimal effect
            # Low position weight so off-center faces are NOT penalized
            score = size_score * 10 - dist_x * 0.5 - dist_y * 0.3
            valid_faces.append((x, y, w, h, score))
    
    if not valid_faces:
        return None
    
    # Return face with highest score
    best = max(valid_faces, key=lambda f: f[4])
    return (best[0], best[1], best[2], best[3])

def detect_face_multi_method(image):
    """
    MULTI-METHOD FACE DETECTION - LAYERED STRATEGY
    
    Menggunakan strategi berlapis dari parameter ketat ke longgar:
    1. Pass 1: Haar Cascade Frontal (parameter NORMAL - paling akurat)
    2. Pass 2: Haar Cascade Frontal (parameter SENSITIF)
    3. Pass 3: Haar Cascade Profile Face
    4. Pass 4: Haar Cascade Frontal (parameter SANGAT SENSITIF + validasi ketat)
    5. Fallback: Center-top crop
    
    Semua pass menggunakan validasi ukuran dan aspek rasio.
    
    Returns: (cropped_face_image, bbox) or None
    """
    
    img_h, img_w = image.shape[:2]
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    
    # ==========================================
    # PASS 1: HAAR CASCADE FRONTAL - PARAMETER NORMAL (paling akurat)
    # ==========================================
    min_face_size = max(30, int(min(img_w, img_h) * 0.15))
    
    faces = face_cascade.detectMultiScale(
        gray_enhanced,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        best = select_best_face(faces, img_h, img_w, min_ratio=0.12)
        if best:
            x, y, w, h = best
            print(f"✅ Face detected (Pass 1 - Normal): bbox=({x},{y},{w},{h})")
            face_img = extract_face_with_margin(image, x, y, w, h)
            return face_img, (x, y, w, h)
    
    # ==========================================
    # PASS 2: HAAR CASCADE FRONTAL - PARAMETER SENSITIF
    # ==========================================
    min_face_size_2 = max(25, int(min(img_w, img_h) * 0.12))
    
    faces = face_cascade.detectMultiScale(
        gray_enhanced,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(min_face_size_2, min_face_size_2),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        best = select_best_face(faces, img_h, img_w, min_ratio=0.10)
        if best:
            x, y, w, h = best
            print(f"✅ Face detected (Pass 2 - Sensitive): bbox=({x},{y},{w},{h})")
            face_img = extract_face_with_margin(image, x, y, w, h)
            return face_img, (x, y, w, h)
    
    # ==========================================
    # PASS 3: HAAR CASCADE PROFILE FACE
    # ==========================================
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_profileface.xml'
    )
    
    for g in [gray_enhanced, gray]:
        faces = profile_cascade.detectMultiScale(
            g,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(min_face_size_2, min_face_size_2)
        )
        
        if len(faces) > 0:
            best = select_best_face(faces, img_h, img_w, min_ratio=0.10)
            if best:
                x, y, w, h = best
                print(f"✅ Face detected (Pass 3 - Profile): bbox=({x},{y},{w},{h})")
                face_img = extract_face_with_margin(image, x, y, w, h)
                return face_img, (x, y, w, h)
    
    # Also try flipped image for profile detection
    gray_flipped = cv2.flip(gray_enhanced, 1)
    faces = profile_cascade.detectMultiScale(
        gray_flipped,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(min_face_size_2, min_face_size_2)
    )
    
    if len(faces) > 0:
        best = select_best_face(faces, img_h, img_w, min_ratio=0.10)
        if best:
            x, y, w, h = best
            # Flip x coordinate back
            x = img_w - x - w
            print(f"✅ Face detected (Pass 3 - Profile Flipped): bbox=({x},{y},{w},{h})")
            face_img = extract_face_with_margin(image, x, y, w, h)
            return face_img, (x, y, w, h)
    
    # ==========================================
    # PASS 4: HAAR CASCADE FRONTAL - SANGAT SENSITIF + VALIDASI KETAT
    # ==========================================
    min_face_size_3 = max(20, int(min(img_w, img_h) * 0.10))
    
    faces = face_cascade.detectMultiScale(
        gray_enhanced,
        scaleFactor=1.02,
        minNeighbors=2,
        minSize=(min_face_size_3, min_face_size_3),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        best = select_best_face(faces, img_h, img_w, min_ratio=0.10)
        if best:
            x, y, w, h = best
            print(f"✅ Face detected (Pass 4 - Very Sensitive): bbox=({x},{y},{w},{h})")
            face_img = extract_face_with_margin(image, x, y, w, h)
            return face_img, (x, y, w, h)
    
    # ==========================================
    # PASS 5: DNN FACE DETECTION (if available)
    # ==========================================
    try:
        prototxt_path = 'deploy.prototxt'
        model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
        
        if os.path.exists(prototxt_path) and os.path.exists(model_path):
            net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 
                1.0, 
                (300, 300), 
                (104.0, 177.0, 123.0)
            )
            
            net.setInput(blob)
            detections = net.forward()
            
            best_confidence = 0
            best_box = None
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > 0.3:
                    box = detections[0, 0, i, 3:7] * np.array([img_w, img_h, img_w, img_h])
                    (startX, startY, endX, endY) = box.astype("int")
                    bw, bh = endX - startX, endY - startY
                    
                    if is_valid_face(startX, startY, bw, bh, img_h, img_w, 0.08):
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_box = (startX, startY, bw, bh)
            
            if best_box:
                x, y, w, h = best_box
                print(f"✅ Face detected (Pass 5 - DNN, conf={best_confidence:.2f}): bbox=({x},{y},{w},{h})")
                face_img = extract_face_with_margin(image, x, y, w, h)
                return face_img, (x, y, w, h)
    
    except Exception as e:
        print(f"⚠️ DNN detection failed: {e}")
    
    # ==========================================
    # FALLBACK: Center-top crop (wajah biasanya di atas-tengah)
    # ==========================================
    print("⚠️ No face detected with any method, using center-top crop fallback")
    
    # Crop center-top region (upper 60% of image, centered horizontally)
    crop_h = int(img_h * 0.6)
    crop_w = min(img_w, crop_h)  # Keep roughly square
    
    x1 = max(0, (img_w - crop_w) // 2)
    y1 = 0  # Start from top
    x2 = min(img_w, x1 + crop_w)
    y2 = min(img_h, y1 + crop_h)
    
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
    
    # Convert BGR (OpenCV) to RGB (Keras/TensorFlow training format)
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    face_resized = cv2.resize(face_rgb, target_size)
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

# ==========================================
# ANALISIS EKSPRESI WAJAH
# ==========================================

def analyze_facial_expressions(face_img, prediction_label, confidence):
    """
    Analyze facial expressions using OpenCV Haar cascades.
    Detects eye state, mouth state, and overall expression based on
    the CNN prediction and visual feature detection.
    
    Returns a list of expression indicators with icon, label, value, and status.
    """
    try:
        h, w = face_img.shape[:2]
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        is_fatigued = prediction_label == 'Fatigued'
        conf_pct = confidence * 100
        
        # --- Eye Detection ---
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        eye_tree_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )
        
        # Search in upper 60% of face
        eye_region = gray[0:int(h * 0.6), :]
        eyes = eye_cascade.detectMultiScale(
            eye_region, scaleFactor=1.1, minNeighbors=3,
            minSize=(max(15, int(w * 0.08)), max(15, int(h * 0.05)))
        )
        
        # Fallback with eyeglasses cascade
        if len(eyes) == 0:
            eyes = eye_tree_cascade.detectMultiScale(
                eye_region, scaleFactor=1.1, minNeighbors=3,
                minSize=(max(15, int(w * 0.08)), max(15, int(h * 0.05)))
            )
        
        eyes_detected = len(eyes)
        
        # Determine eye state based on detection + prediction
        if is_fatigued:
            if eyes_detected == 0:
                eye_state = 'Tertutup'
                eye_status = 'danger'
            elif conf_pct > 75:
                eye_state = 'Setengah Tertutup'
                eye_status = 'warning'
            else:
                eye_state = 'Sayu / Kurang Fokus'
                eye_status = 'warning'
        else:
            if eyes_detected >= 2:
                eye_state = 'Terbuka Normal'
                eye_status = 'good'
            elif eyes_detected == 1:
                eye_state = 'Terbuka'
                eye_status = 'good'
            else:
                eye_state = 'Tidak Terdeteksi'
                eye_status = 'neutral'
        
        # --- Mouth Detection ---
        mouth_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        
        # Search in lower 50% of face
        mouth_region = gray[int(h * 0.5):, :]
        mouths = mouth_cascade.detectMultiScale(
            mouth_region, scaleFactor=1.3, minNeighbors=15,
            minSize=(max(20, int(w * 0.15)), max(10, int(h * 0.05)))
        )
        
        mouths_detected = len(mouths)
        
        # Check for wide open mouth (yawning) - larger mouth detection
        wide_mouths = mouth_cascade.detectMultiScale(
            mouth_region, scaleFactor=1.1, minNeighbors=5,
            minSize=(max(30, int(w * 0.25)), max(15, int(h * 0.08)))
        )
        
        if is_fatigued:
            if len(wide_mouths) > 0 and conf_pct > 70:
                mouth_state = 'Menguap'
                mouth_status = 'danger'
            elif conf_pct > 75:
                mouth_state = 'Terbuka / Menguap'
                mouth_status = 'warning'
            else:
                mouth_state = 'Kendur / Lesu'
                mouth_status = 'warning'
        else:
            if mouths_detected > 0:
                mouth_state = 'Normal'
                mouth_status = 'good'
            else:
                mouth_state = 'Normal / Tertutup'
                mouth_status = 'good'
        
        # --- Overall Expression ---
        if is_fatigued:
            if conf_pct > 80:
                expression_state = 'Sangat Lesu'
                expression_status = 'danger'
            elif conf_pct > 65:
                expression_state = 'Lesu / Mengantuk'
                expression_status = 'warning'
            else:
                expression_state = 'Sedikit Lelah'
                expression_status = 'warning'
        else:
            if conf_pct > 80:
                expression_state = 'Segar & Aktif'
                expression_status = 'good'
            elif conf_pct > 65:
                expression_state = 'Normal'
                expression_status = 'good'
            else:
                expression_state = 'Cukup Baik'
                expression_status = 'good'
        
        # --- Overall Conclusion ---
        if is_fatigued:
            conclusion_state = 'Terdeteksi Kelelahan'
            conclusion_status = 'danger'
        else:
            conclusion_state = 'Kondisi Baik'
            conclusion_status = 'good'
        
        expressions = [
            {
                'icon': 'eye',
                'label': 'Mata',
                'value': eye_state,
                'status': eye_status,
                'detail': f'{eyes_detected} mata terdeteksi'
            },
            {
                'icon': 'mouth',
                'label': 'Mulut',
                'value': mouth_state,
                'status': mouth_status,
                'detail': None
            },
            {
                'icon': 'face',
                'label': 'Ekspresi',
                'value': expression_state,
                'status': expression_status,
                'detail': None
            },
            {
                'icon': 'result',
                'label': 'Keseluruhan',
                'value': conclusion_state,
                'status': conclusion_status,
                'detail': None
            }
        ]
        
        print(f"✅ Expression analysis: eyes={eye_state}, mouth={mouth_state}, expr={expression_state}")
        return expressions
        
    except Exception as e:
        print(f"❌ Expression analysis failed: {e}")
        # Return default expression based on prediction
        is_fatigued = prediction_label == 'Fatigued'
        return [
            {'icon': 'eye', 'label': 'Mata', 'value': 'Sayu' if is_fatigued else 'Normal', 'status': 'warning' if is_fatigued else 'good', 'detail': None},
            {'icon': 'mouth', 'label': 'Mulut', 'value': 'Lesu' if is_fatigued else 'Normal', 'status': 'warning' if is_fatigued else 'good', 'detail': None},
            {'icon': 'face', 'label': 'Ekspresi', 'value': 'Lelah' if is_fatigued else 'Segar', 'status': 'warning' if is_fatigued else 'good', 'detail': None},
            {'icon': 'result', 'label': 'Keseluruhan', 'value': 'Kelelahan' if is_fatigued else 'Baik', 'status': 'danger' if is_fatigued else 'good', 'detail': None},
        ]


def get_fatigue_level(prediction_label, confidence):
    """
    Determine fatigue level based on prediction and confidence.
    
    Returns: dict with level key, label, and color class
    - 'normal'  → Tidak Lelah
    - 'ringan'  → Kelelahan Ringan (fatigued, confidence < 65%)
    - 'sedang'  → Kelelahan Sedang (fatigued, 65-80%)
    - 'berat'   → Kelelahan Berat (fatigued, > 80%)
    """
    conf_pct = confidence * 100
    
    if prediction_label != 'Fatigued':
        return {
            'level': 'normal',
            'label': 'Tidak Lelah',
            'severity': 0
        }
    
    if conf_pct >= 80:
        return {
            'level': 'berat',
            'label': 'Kelelahan Berat',
            'severity': 3
        }
    elif conf_pct >= 65:
        return {
            'level': 'sedang',
            'label': 'Kelelahan Sedang',
            'severity': 2
        }
    else:
        return {
            'level': 'ringan',
            'label': 'Kelelahan Ringan',
            'severity': 1
        }


def get_recommendation(prediction, confidence):
    """Generate recommendation based on fatigue level"""
    fatigue = get_fatigue_level(prediction, confidence)
    level = fatigue['level']
    
    if level == 'berat':
        return {
            'level': 'berat',
            'message': 'Anda mengalami kelelahan berat. Segera hentikan aktivitas dan istirahat.',
            'actions': [
                'Hentikan semua aktivitas kerja segera',
                'Istirahat minimal 20-30 menit',
                'Cuci muka dengan air dingin',
                'Konsumsi air putih dan makanan ringan',
                'Jika memungkinkan, tidur sejenak (power nap)',
                'Laporkan kondisi Anda ke supervisor'
            ]
        }
    elif level == 'sedang':
        return {
            'level': 'sedang',
            'message': 'Tanda-tanda kelelahan sedang terdeteksi. Pertimbangkan untuk segera beristirahat.',
            'actions': [
                'Ambil jeda istirahat 10-15 menit',
                'Lakukan peregangan ringan',
                'Cuci muka atau kompres mata',
                'Minum air putih yang cukup',
                'Hindari tugas yang memerlukan konsentrasi tinggi'
            ]
        }
    elif level == 'ringan':
        return {
            'level': 'ringan',
            'message': 'Tanda awal kelelahan mulai terdeteksi. Jaga kondisi Anda.',
            'actions': [
                'Ambil jeda singkat 5-10 menit',
                'Pastikan pencahayaan ruangan memadai',
                'Lakukan peregangan mata (lihat jauh 20 detik)',
                'Monitor kondisi Anda secara berkala'
            ]
        }
    else:
        return {
            'level': 'normal',
            'message': 'Kondisi baik. Tidak terdeteksi tanda-tanda kelelahan.',
            'actions': [
                'Tetap jaga pola istirahat yang teratur',
                'Hindari bekerja terlalu lama tanpa jeda',
                'Lakukan pemeriksaan berkala'
            ]
        }

# ==========================================
# AUTH ENDPOINTS
# ==========================================
@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../frontend', path)


@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        if not data or not data.get('username') or not data.get('password'):
            return jsonify({'success': False, 'error': 'Username dan password harus diisi'}), 400
        
        username = data['username'].strip().lower()
        password = data['password']
        
        if len(username) < 3:
            return jsonify({'success': False, 'error': 'Username minimal 3 karakter'}), 400
        if len(password) < 4:
            return jsonify({'success': False, 'error': 'Password minimal 4 karakter'}), 400
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Check if username exists
        existing = cursor.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
        if existing:
            conn.close()
            return jsonify({'success': False, 'error': 'Username sudah digunakan'}), 409
        
        # Create user
        pw_hash = hash_password(password)
        cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, pw_hash))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        # Auto login
        token = uuid.uuid4().hex
        SESSIONS[token] = {'user_id': user_id, 'username': username}
        
        print(f"✅ User registered: {username}")
        return jsonify({
            'success': True,
            'token': token,
            'user': {'id': user_id, 'username': username}
        }), 201
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        if not data or not data.get('username') or not data.get('password'):
            return jsonify({'success': False, 'error': 'Username dan password harus diisi'}), 400
        
        username = data['username'].strip().lower()
        password = data['password']
        
        conn = get_db()
        cursor = conn.cursor()
        user = cursor.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if not user or not verify_password(user['password_hash'], password):
            return jsonify({'success': False, 'error': 'Username atau password salah'}), 401
        
        # Create session
        token = uuid.uuid4().hex
        SESSIONS[token] = {'user_id': user['id'], 'username': user['username']}
        
        print(f"✅ User logged in: {username}")
        return jsonify({
            'success': True,
            'token': token,
            'user': {'id': user['id'], 'username': user['username']}
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout user"""
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]
        if token in SESSIONS:
            username = SESSIONS[token].get('username', '?')
            del SESSIONS[token]
            print(f"✅ User logged out: {username}")
    return jsonify({'success': True}), 200

@app.route('/api/me', methods=['GET'])
def get_me():
    """Get current user info"""
    user = get_current_user()
    if not user:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    return jsonify({
        'success': True,
        'user': {'id': user['user_id'], 'username': user['username']}
    }), 200

# ==========================================
# HISTORY ENDPOINTS
# ==========================================

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get detection history for current user"""
    user = get_current_user()
    if not user:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        offset = (page - 1) * per_page
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Get total count
        total = cursor.execute(
            'SELECT COUNT(*) FROM detection_history WHERE user_id = ?',
            (user['user_id'],)
        ).fetchone()[0]
        
        # Get records
        records = cursor.execute(
            '''SELECT id, prediction, confidence, face_image, gradcam_image,
                      face_regions, explanation, timestamp
               FROM detection_history
               WHERE user_id = ?
               ORDER BY timestamp DESC
               LIMIT ? OFFSET ?''',
            (user['user_id'], per_page, offset)
        ).fetchall()
        
        conn.close()
        
        history = []
        for r in records:
            history.append({
                'id': r['id'],
                'prediction': r['prediction'],
                'confidence': r['confidence'],
                'face_image': r['face_image'],
                'gradcam_image': r['gradcam_image'],
                'face_regions': json.loads(r['face_regions']) if r['face_regions'] else None,
                'explanation': r['explanation'],
                'timestamp': r['timestamp']
            })
        
        return jsonify({
            'success': True,
            'history': history,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/history/<int:record_id>', methods=['DELETE'])
def delete_history(record_id):
    """Delete a detection history record"""
    user = get_current_user()
    if not user:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'DELETE FROM detection_history WHERE id = ? AND user_id = ?',
            (record_id, user['user_id'])
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted == 0:
            return jsonify({'success': False, 'error': 'Record not found'}), 404
        
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

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
        
        # Facial Expression Analysis
        expressions = analyze_facial_expressions(face_img, prediction_label, confidence)
        
        # Fatigue Level
        fatigue_level = get_fatigue_level(prediction_label, confidence)
        
        # Response
        detection_timestamp = datetime.now().isoformat()
        conf_value = round(confidence * 100, 2)
        
        response = {
            'success': True,
            'prediction': prediction_label,
            'confidence': conf_value,
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
            'expressions': expressions,
            'fatigue_level': fatigue_level,
            'timestamp': detection_timestamp,
            'recommendation': get_recommendation(prediction_label, confidence)
        }
        
        # Save to history if user is logged in
        current_user = get_current_user()
        if current_user:
            try:
                conn = get_db()
                cursor = conn.cursor()
                cursor.execute(
                    '''INSERT INTO detection_history
                       (user_id, prediction, confidence, face_image, gradcam_image,
                        face_regions, explanation, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (
                        current_user['user_id'],
                        prediction_label,
                        conf_value,
                        face_base64,
                        None,
                        json.dumps(expressions) if expressions else None,
                        fatigue_level.get('label', ''),
                        detection_timestamp
                    )
                )
                conn.commit()
                response['history_saved'] = True
                response['history_id'] = cursor.lastrowid
                conn.close()
                print(f"💾 Detection saved to history for user: {current_user['username']}")
            except Exception as db_err:
                print(f"⚠️ Failed to save to history: {db_err}")
                response['history_saved'] = False
        
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
    print("Login/Logout + Detection History + Grad-CAM")
    print("="*60)
    
    init_db()
    load_model()
    
    print("\n📡 Starting Flask server...")
    print("="*60)
    
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        threaded=True
    )