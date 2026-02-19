"""
THRESHOLD TESTER
Script untuk menemukan threshold optimal yang mengurangi kesalahan prediksi

Cara pakai:
1. Siapkan beberapa gambar test (5 fatigued + 5 non-fatigued)
2. Jalankan script ini
3. Lihat threshold mana yang paling akurat
4. Update PREDICTION_THRESHOLD di app.py
"""

import cv2
import numpy as np
from tensorflow import keras
import os

# Load model
MODEL = keras.models.load_model('backend/fatigue_detection_model.h5')
IMG_SIZE = (96, 96)

def preprocess_image(image_path):
    """Preprocess image"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Simple face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
    else:
        # Use whole image
        face = img
    
    face_resized = cv2.resize(face, IMG_SIZE)
    face_normalized = face_resized.astype('float32') / 255.0
    face_batch = np.expand_dims(face_normalized, axis=0)
    
    return face_batch

def test_threshold(image_path, threshold, true_label):
    """Test dengan threshold tertentu"""
    preprocessed = preprocess_image(image_path)
    if preprocessed is None:
        return None, None
    
    # Predict
    prediction = MODEL.predict(preprocessed, verbose=0)
    raw_confidence = float(prediction[0][0])
    
    # Apply threshold
    adjusted_threshold = 1 - threshold
    
    if raw_confidence < adjusted_threshold:
        predicted_label = "fatigued"
        confidence = 1 - raw_confidence
    else:
        predicted_label = "non_fatigued"
        confidence = raw_confidence
    
    # Check if correct
    is_correct = (predicted_label == true_label)
    
    return predicted_label, is_correct, confidence, raw_confidence

def main():
    """Main tester"""
    
    print("="*60)
    print("🔍 THRESHOLD TESTER")
    print("="*60)
    
    # GANTI PATH INI dengan gambar test Anda!
    test_images = [
        # Format: (path, true_label)
        ('model/dataset/test/fatigued/0001.jpg', 'fatigued'),
        ('model/dataset/test/fatigued/0002.jpg', 'fatigued'),
        ('model/dataset/test/fatigued/0003.jpg', 'fatigued'),
        ('model/dataset/test/fatigued/0004.jpg', 'fatigued'),
        ('model/dataset/test/fatigued/0005.jpg', 'fatigued'),
        ('model/dataset/test/non_fatigued/0001.jpg', 'non_fatigued'),
        ('model/dataset/test/non_fatigued/0002.jpg', 'non_fatigued'),
        ('model/dataset/test/non_fatigued/0003.jpg', 'non_fatigued'),
        ('model/dataset/test/non_fatigued/0004.jpg', 'non_fatigued'),
        ('model/dataset/test/non_fatigued/0005.jpg', 'non_fatigued'),
    ]
    
    # Test berbagai threshold
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    
    print("\nTesting thresholds...")
    print("-" * 60)
    
    best_threshold = 0.5
    best_accuracy = 0
    
    for threshold in thresholds:
        correct = 0
        total = 0
        
        for img_path, true_label in test_images:
            if not os.path.exists(img_path):
                continue
            
            result = test_threshold(img_path, threshold, true_label)
            if result[0] is not None:
                predicted, is_correct, conf, raw = result
                if is_correct:
                    correct += 1
                total += 1
        
        if total > 0:
            accuracy = (correct / total) * 100
            
            marker = ""
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                marker = "  ⭐ BEST"
            
            print(f"Threshold {threshold:.2f}: {correct}/{total} correct ({accuracy:.1f}%){marker}")
    
    print("-" * 60)
    print(f"\n🏆 BEST THRESHOLD: {best_threshold}")
    print(f"   Accuracy: {best_accuracy:.1f}%")
    print("\n💡 ACTION:")
    print(f"   Edit backend/app.py")
    print(f"   PREDICTION_THRESHOLD = {best_threshold}")
    print("="*60)

if __name__ == "__main__":
    main()