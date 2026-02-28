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
    - Aspect ratio check (faces are roughly square, 0.5 to 2.0)
    - Minimum size relative to image (at least min_ratio of image dimension)
    - Position check: face should be in upper 75% of image
    """
    aspect_ratio = w / h if h > 0 else 0
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False
    
    # Minimum size: face should be at least min_ratio of image width or height
    min_size = min(img_w, img_h) * min_ratio
    if w < min_size or h < min_size:
        return False
    
    # Face should generally be in the upper 75% of the image
    face_center_y = y + h / 2
    if face_center_y > img_h * 0.85:
        return False
    
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
            
            # Combined score: size is most important, position is secondary
            score = size_score * 10 - dist_x * 2 - dist_y * 1
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
# GRAD-CAM (Explainable AI)
# ==========================================

def generate_feature_heatmap(face_img, prediction_label):
    """
    Generate a simulated attention heatmap using OpenCV feature detection.
    This is used as a fallback when TensorFlow is not available.
    
    Detects eyes and mouth regions using Haar cascades, then creates
    a heatmap highlighting fatigue-relevant areas (eyes, mouth).
    """
    try:
        h, w = face_img.shape[:2]
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Create base heatmap
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Load Haar cascades for facial features
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        mouth_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        
        # Detect eyes (upper half of face)
        eye_region = gray[0:int(h*0.6), :]
        eyes = eye_cascade.detectMultiScale(
            eye_region, scaleFactor=1.1, minNeighbors=3,
            minSize=(max(15, int(w*0.08)), max(15, int(h*0.05)))
        )
        
        # Detect mouth (lower half of face)
        mouth_region = gray[int(h*0.5):, :]
        mouths = mouth_cascade.detectMultiScale(
            mouth_region, scaleFactor=1.2, minNeighbors=10,
            minSize=(max(20, int(w*0.15)), max(10, int(h*0.05)))
        )
        
        is_fatigued = prediction_label == 'Fatigued'
        
        # Generate Gaussian blobs at detected feature locations
        for (ex, ey, ew, eh) in eyes:
            cx, cy = ex + ew//2, ey + eh//2
            # Create Gaussian blob around eyes
            intensity = 0.9 if is_fatigued else 0.5
            sigma_x = ew * 0.8
            sigma_y = eh * 0.8
            for y_pos in range(max(0, cy - ew), min(h, cy + ew)):
                for x_pos in range(max(0, cx - ew), min(w, cx + ew)):
                    dist = ((x_pos - cx)**2 / (2*sigma_x**2)) + ((y_pos - cy)**2 / (2*sigma_y**2))
                    heatmap[y_pos, x_pos] = max(heatmap[y_pos, x_pos], intensity * np.exp(-dist))
        
        for (mx, my, mw, mh) in mouths:
            cy = int(h*0.5) + my + mh//2
            cx = mx + mw//2
            intensity = 0.7 if is_fatigued else 0.3
            sigma_x = mw * 0.7
            sigma_y = mh * 0.7
            for y_pos in range(max(0, cy - mh), min(h, cy + mh)):
                for x_pos in range(max(0, cx - mw), min(w, cx + mw)):
                    dist = ((x_pos - cx)**2 / (2*sigma_x**2)) + ((y_pos - cy)**2 / (2*sigma_y**2))
                    heatmap[y_pos, x_pos] = max(heatmap[y_pos, x_pos], intensity * np.exp(-dist))
        
        # If no features detected, create default region-based heatmap
        if len(eyes) == 0 and len(mouths) == 0:
            # Focus on eye area (25-45% height) and mouth area (65-85% height)
            eye_center_y = int(h * 0.35)
            mouth_center_y = int(h * 0.75)
            center_x = w // 2
            
            for y_pos in range(h):
                for x_pos in range(w):
                    # Eye region contribution
                    eye_dist = ((x_pos - center_x)**2 / (2*(w*0.3)**2)) + \
                               ((y_pos - eye_center_y)**2 / (2*(h*0.1)**2))
                    eye_val = 0.8 * np.exp(-eye_dist)
                    
                    # Mouth region contribution
                    mouth_dist = ((x_pos - center_x)**2 / (2*(w*0.25)**2)) + \
                                 ((y_pos - mouth_center_y)**2 / (2*(h*0.08)**2))
                    mouth_val = 0.5 * np.exp(-mouth_dist)
                    
                    heatmap[y_pos, x_pos] = max(eye_val, mouth_val)
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Apply Gaussian blur for smoother visualization
        ksize = max(3, int(min(h, w) * 0.15)) | 1  # Ensure odd number
        heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), 0)
        
        # Re-normalize after blur
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Resize heatmap to face image size (should already be same size)
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Apply colormap (JET: blue=cold, red=hot)
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        
        # Overlay on face image
        overlay = cv2.addWeighted(face_img, 0.6, heatmap_colored, 0.4, 0)
        
        print(f"✅ Feature-based heatmap generated (eyes={len(eyes)}, mouths={len(mouths)})")
        return overlay, heatmap_resized
        
    except Exception as e:
        import traceback
        print(f"❌ Feature heatmap generation failed: {e}")
        traceback.print_exc()
        return None


def generate_gradcam(model, preprocessed_img, face_img):
    """
    Generate Grad-CAM heatmap overlay on face image.
    Uses TensorFlow GradientTape to compute gradients of the output
    w.r.t. the last convolutional layer's feature maps.
    
    Compatible with TF 2.16+ Sequential models (avoids model.input).
    """
    try:
        if not TF_AVAILABLE or tf is None:
            print("❌ TensorFlow not available for Grad-CAM")
            return None
        
        # Find the last convolutional layer name
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        
        if last_conv_layer_name is None:
            print("❌ No Conv2D layer found for Grad-CAM")
            return None
        
        print(f"📊 Grad-CAM: Using last conv layer '{last_conv_layer_name}'")
        
        # Forward pass manually through layers, capturing conv output
        img_tensor = tf.cast(preprocessed_img, tf.float32)
        
        with tf.GradientTape() as tape:
            # Manual forward pass through each layer
            x = img_tensor
            conv_output = None
            
            for layer in model.layers:
                x = layer(x, training=False)
                if layer.name == last_conv_layer_name:
                    conv_output = x
                    tape.watch(conv_output)  # Watch conv output for gradient computation
            
            predictions = x
            loss = predictions[:, 0]  # Output neuron
        
        if conv_output is None:
            print("❌ Conv layer output not captured during forward pass")
            return None
        
        # Gradients of the output w.r.t. the conv layer output
        grads = tape.gradient(loss, conv_output)
        
        if grads is None:
            print("❌ Gradients are None - cannot compute Grad-CAM")
            return None
        
        # Global average pooling of the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs by the pooled gradients
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to face image size
        heatmap_resized = cv2.resize(heatmap, (face_img.shape[1], face_img.shape[0]))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        
        # Overlay on face image
        overlay = cv2.addWeighted(face_img, 0.6, heatmap_colored, 0.4, 0)
        
        print("✅ Grad-CAM heatmap generated successfully")
        return overlay, heatmap_resized
        
    except Exception as e:
        import traceback
        print(f"❌ Grad-CAM generation failed: {e}")
        traceback.print_exc()
        return None

def analyze_face_regions(heatmap, face_shape):
    """
    Analyze which face regions contribute most to the prediction.
    Uses sum of activations above a threshold for accurate contribution.
    """
    try:
        h, w = face_shape[:2]
        
        # Resize heatmap to match face dimensions if needed
        if heatmap.shape[0] != h or heatmap.shape[1] != w:
            heatmap = cv2.resize(heatmap, (w, h))
        
        # Apply threshold to focus on significant activations only
        threshold = 0.2
        heatmap_threshed = np.where(heatmap > threshold, heatmap, 0)
        
        # Define face regions (approximate proportions for typical face crops)
        regions = {
            'Dahi (Forehead)': (0, 0, w, int(h * 0.3)),
            'Mata (Eyes)': (0, int(h * 0.2), w, int(h * 0.5)),
            'Hidung (Nose)': (0, int(h * 0.4), w, int(h * 0.65)),
            'Mulut (Mouth)': (0, int(h * 0.6), w, int(h * 0.85)),
            'Dagu (Chin)': (0, int(h * 0.8), w, h)
        }
        
        region_scores = []
        total_activation = 0
        
        for name, (x1, y1, x2, y2) in regions.items():
            region_heat = heatmap_threshed[y1:y2, x1:x2]
            if region_heat.size > 0:
                # Use SUM of activations (not mean) to avoid bias from region size
                score = float(np.sum(region_heat))
            else:
                score = 0.0
            region_scores.append({'region': name, 'score': score})
            total_activation += score
        
        # Convert to percentages
        if total_activation > 0:
            for item in region_scores:
                item['contribution'] = round((item['score'] / total_activation) * 100, 1)
        else:
            for item in region_scores:
                item['contribution'] = 20.0  # Equal distribution
        
        # Sort by contribution (highest first)
        region_scores.sort(key=lambda x: x['contribution'], reverse=True)
        
        # Remove the raw score from the response
        for item in region_scores:
            del item['score']
        
        return region_scores
        
    except Exception as e:
        print(f"Face region analysis failed: {e}")
        return None

def generate_explanation(prediction, confidence, face_regions):
    """
    Generate a human-readable explanation of why the model made its prediction.
    """
    try:
        if face_regions is None or len(face_regions) == 0:
            return None
        
        top_region = face_regions[0]['region']
        top_contribution = face_regions[0]['contribution']
        
        if prediction == 'Fatigued':
            explanation = (
                f"Model CNN mendeteksi **kelelahan** dengan tingkat keyakinan "
                f"**{confidence}%**. Area wajah yang paling berpengaruh adalah "
                f"**{top_region}** ({top_contribution}% kontribusi). "
            )
            if 'Mata' in top_region:
                explanation += (
                    "Hal ini menunjukkan bahwa pola area mata (seperti kelopak mata "
                    "turun, lingkaran hitam, atau mata setengah tertutup) menjadi "
                    "indikator utama kelelahan yang terdeteksi oleh model."
                )
            elif 'Mulut' in top_region:
                explanation += (
                    "Hal ini menunjukkan bahwa pola area mulut (seperti menguap "
                    "atau ekspresi wajah yang kendur) menjadi indikator utama "
                    "kelelahan yang terdeteksi oleh model."
                )
            else:
                explanation += (
                    "Pola visual di area tersebut menunjukkan tanda-tanda "
                    "kelelahan yang dikenali oleh model CNN."
                )
        else:
            explanation = (
                f"Model CNN mendeteksi wajah dalam kondisi **tidak lelah** dengan "
                f"tingkat keyakinan **{confidence}%**. Area wajah yang paling "
                f"berpengaruh dalam keputusan ini adalah **{top_region}** "
                f"({top_contribution}% kontribusi). "
                "Pola visual menunjukkan ekspresi wajah yang segar dan aktif."
            )
        
        return explanation
        
    except Exception as e:
        print(f"Explanation generation failed: {e}")
        return None

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
# AUTH ENDPOINTS
# ==========================================

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
        
        # Grad-CAM / Feature Heatmap
        gradcam_base64 = None
        face_regions = None
        explanation = None
        
        print(f"📊 Heatmap: MODEL loaded = {MODEL is not None}, TF_AVAILABLE = {TF_AVAILABLE}")
        
        heatmap_result = None
        
        if MODEL is not None and TF_AVAILABLE:
            # Use real Grad-CAM with TensorFlow model
            heatmap_result = generate_gradcam(MODEL, preprocessed, face_img)
            print(f"📊 Grad-CAM result: {heatmap_result is not None}")
        
        if heatmap_result is None:
            # Fallback: use OpenCV feature-based heatmap
            print("📊 Using feature-based heatmap (fallback)")
            heatmap_result = generate_feature_heatmap(face_img, prediction_label)
        
        if heatmap_result is not None:
            gradcam_overlay, heatmap = heatmap_result
            gradcam_base64 = encode_image_to_base64(gradcam_overlay)
            face_regions = analyze_face_regions(heatmap, face_img.shape)
            conf_pct = round(confidence * 100, 2)
            explanation = generate_explanation(prediction_label, conf_pct, face_regions)
            print(f"📊 Heatmap: overlay generated, regions={face_regions is not None}, explanation={explanation is not None}")
        else:
            print("⚠️ Heatmap: no heatmap could be generated")
        
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
            'gradcam_image': gradcam_base64,
            'face_regions': face_regions,
            'explanation': explanation,
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
                        gradcam_base64,
                        json.dumps(face_regions) if face_regions else None,
                        explanation,
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