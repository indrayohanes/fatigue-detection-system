# Sistem Deteksi Kelelahan Kerja Berbasis CNN

Sistem deteksi kelelahan kerja menggunakan Convolutional Neural Network (CNN) yang menganalisis ekspresi wajah secara non-invasif untuk meningkatkan keselamatan dan produktivitas kerja.

**Penelitian Skripsi**  
Program Studi Teknik Informatika  
Universitas Bunda Mulia - 2026

---

## 📋 Daftar Isi

- [Tentang Sistem](#tentang-sistem)
- [Fitur Utama](#fitur-utama)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Training Model](#training-model)
- [Evaluasi Model](#evaluasi-model)
- [Struktur Proyek](#struktur-proyek)
- [API Documentation](#api-documentation)
- [Screenshots](#screenshots)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Tentang Sistem

Sistem ini dikembangkan sebagai bagian dari penelitian skripsi untuk mendeteksi kelelahan kerja berdasarkan analisis ekspresi wajah menggunakan teknologi Deep Learning (CNN). 

### Latar Belakang

Kelelahan kerja merupakan kondisi yang dapat menurunkan produktivitas dan meningkatkan risiko kecelakaan kerja. Metode deteksi tradisional menggunakan sensor fisiologis bersifat invasif dan kurang praktis. Sistem ini menawarkan solusi **non-invasif** menggunakan analisis visual ekspresi wajah.

### Tujuan Penelitian

1. Merancang sistem deteksi kelelahan berbasis ekspresi wajah menggunakan CNN
2. Membangun aplikasi web dashboard yang user-friendly
3. Menganalisis performa model CNN dengan metrik evaluasi standar

---

## ✨ Fitur Utama

### Backend Features
- ✅ **Model CNN** terlatih untuk klasifikasi kelelahan (Fatigued/Non-Fatigued)
- ✅ **Face Detection** otomatis menggunakan Haar Cascade
- ✅ **Preprocessing** citra untuk normalisasi input
- ✅ **RESTful API** untuk integrasi frontend-backend
- ✅ **Confidence Score** untuk setiap prediksi
- ✅ **Recommendation System** berbasis hasil deteksi

### Frontend Features
- ✅ **Upload Interface** dengan drag-and-drop support
- ✅ **Real-time Preview** gambar yang diupload
- ✅ **Visual Results** dengan status indikator warna
- ✅ **Detailed Recommendations** untuk tindakan lanjutan
- ✅ **Responsive Design** untuk berbagai perangkat
- ✅ **Modern UI/UX** dengan animasi smooth

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────┐
│   Web Browser   │
│   (Frontend)    │
└────────┬────────┘
         │ HTTP Request
         │ (Upload Image)
         ▼
┌─────────────────┐
│   Flask API     │
│   (Backend)     │
├─────────────────┤
│ 1. File Upload  │
│ 2. Face Detect  │
│ 3. Preprocess   │
│ 4. CNN Predict  │
│ 5. Response     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CNN Model      │
│  (TensorFlow)   │
├─────────────────┤
│ - Conv Layers   │
│ - Pooling       │
│ - Dense Layers  │
│ - Output: 0/1   │
└─────────────────┘
```

---

## 🛠️ Teknologi yang Digunakan

### Backend
- **Flask** - Web framework Python
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Computer vision library
- **NumPy** - Numerical computing
- **Pillow** - Image processing

### Frontend
- **HTML5** - Markup
- **CSS3** - Styling dengan modern design
- **JavaScript (Vanilla)** - Interactivity
- **Fetch API** - AJAX requests

### Model
- **CNN Architecture** - Custom convolutional neural network
- **Dataset** - Fatigue Dataset dari Kaggle

---

## 📥 Instalasi

### Prerequisites
- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Node.js (opsional, untuk development)

### 1. Clone Repository

```bash
git clone <repository-url>
cd fatigue-detection-system
```

### 2. Setup Backend

```bash
cd backend

# Buat virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Model

```bash
cd ../model

# Download dataset dari Kaggle
# Letakkan dataset di folder: dataset/train/ dan dataset/test/

# Train model (opsional jika sudah ada model)
python train_model.py

# Copy trained model ke backend
cp saved_models/best_model.h5 ../backend/fatigue_detection_model.h5
```

### 4. Setup Frontend

Frontend menggunakan vanilla JavaScript, tidak perlu instalasi khusus. Cukup buka file HTML di browser atau jalankan dengan live server.

---

## 🚀 Cara Penggunaan

### Menjalankan Backend API

```bash
cd backend
python app.py
```

Backend akan berjalan di `http://localhost:5000`

### Menjalankan Frontend

#### Opsi 1: Langsung di Browser
```bash
cd frontend
# Buka index.html di browser
```

#### Opsi 2: Menggunakan Live Server (Recommended)
```bash
cd frontend
# Jika sudah install http-server (npm install -g http-server)
http-server -p 8080

# Atau gunakan Python
python -m http.server 8080
```

Frontend akan berjalan di `http://localhost:8080`

### Menggunakan Sistem

1. **Buka aplikasi** di browser
2. **Upload gambar wajah** dengan klik atau drag-and-drop
3. **Klik "Analisis Gambar"** untuk memulai deteksi
4. **Lihat hasil** deteksi dengan rekomendasi tindakan

---

## 🎓 Training Model

### Persiapan Dataset

Struktur dataset yang diperlukan:
```
dataset/
├── train/
│   ├── fatigued/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── non_fatigued/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── test/
    ├── fatigued/
    └── non_fatigued/
```

### Training Process

```bash
cd model
python train_model.py
```

Script akan:
1. Load dan preprocess dataset
2. Build arsitektur CNN
3. Train model dengan data augmentation
4. Save best model berdasarkan validation accuracy
5. Generate visualisasi (training history, confusion matrix)

### Hyperparameter Tuning

Edit `train_model.py` untuk menyesuaikan:
- `IMG_SIZE` - Ukuran input image (default: 48x48)
- `BATCH_SIZE` - Batch size untuk training (default: 32)
- `EPOCHS` - Jumlah epoch (default: 50)
- Learning rate dan optimizer settings

---

## 📊 Evaluasi Model

### Metrik yang Digunakan

Model dievaluasi menggunakan metrik standar:

| Metrik | Formula | Target |
|--------|---------|--------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | ≥ 85% |
| **Precision** | TP / (TP + FP) | ≥ 80% |
| **Recall** | TP / (TP + FN) | ≥ 85% |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | ≥ 83% |

### Confusion Matrix

Setelah training, confusion matrix akan disimpan di `model/saved_models/confusion_matrix.png`

### Training History

Grafik loss dan accuracy tersimpan di `model/saved_models/training_history.png`

---

## 📁 Struktur Proyek

```
fatigue-detection-system/
│
├── backend/
│   ├── app.py                      # Flask API
│   ├── requirements.txt            # Python dependencies
│   └── uploads/                    # Temporary upload folder
│
├── frontend/
│   ├── index.html                  # Main HTML
│   ├── style.css                   # Styling
│   └── script.js                   # JavaScript logic
│
├── model/
│   ├── train_model.py              # Training script
│   ├── dataset/                    # Dataset folder
│   │   ├── train/
│   │   └── test/
│   └── saved_models/               # Trained models
│       ├── best_model.h5
│       ├── final_model.h5
│       ├── confusion_matrix.png
│       └── training_history.png
│
└── README.md                       # Documentation
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### 1. Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-02-14T12:00:00.000Z"
}
```

#### 2. Detect Fatigue
```http
POST /api/detect
Content-Type: multipart/form-data
```

**Request Body:**
- `image` (file) - Image file (JPG, PNG, max 16MB)

**Success Response (200):**
```json
{
  "success": true,
  "prediction": "Fatigued",
  "confidence": 87.5,
  "confidence_raw": 0.875,
  "face_detected": true,
  "face_bbox": {
    "x": 100,
    "y": 150,
    "width": 200,
    "height": 200
  },
  "face_image": "base64_encoded_image",
  "timestamp": "2026-02-14T12:00:00.000Z",
  "recommendation": {
    "level": "high",
    "message": "Tingkat kelelahan tinggi terdeteksi...",
    "actions": [
      "Hentikan aktivitas kerja sementara",
      "Istirahat minimal 15-20 menit",
      ...
    ]
  }
}
```

**Error Response (400):**
```json
{
  "success": false,
  "error": "No face detected in the image",
  "suggestion": "Please upload an image with a clear, visible face"
}
```

---

## 🖼️ Screenshots

### Landing Page
![Landing Page](screenshots/landing.png)

### Detection Interface
![Detection Interface](screenshots/detection.png)

### Results Display
![Results](screenshots/results.png)

---

## 🔧 Troubleshooting

### Backend Issues

**Problem: Model not loading**
```
Error: Model file not found
```
**Solution:**
- Pastikan file model ada di path yang benar: `../model/fatigue_detection_model.h5`
- Atau jalankan training terlebih dahulu dengan `python train_model.py`

**Problem: Face detection fails**
```
Error: No face detected
```
**Solution:**
- Pastikan gambar memiliki wajah yang jelas dan terlihat
- Gunakan gambar dengan pencahayaan yang baik
- Pastikan wajah tidak terhalang (tidak pakai masker, kacamata hitam, dll)

### Frontend Issues

**Problem: CORS Error**
```
Access to fetch has been blocked by CORS policy
```
**Solution:**
- Pastikan backend menggunakan `flask-cors`
- Check apakah `CORS(app)` sudah ditambahkan di `app.py`

**Problem: API not responding**
```
Failed to fetch
```
**Solution:**
- Pastikan backend berjalan di `http://localhost:5000`
- Check firewall atau antivirus yang mungkin blocking port

### Training Issues

**Problem: Out of memory during training**
```
ResourceExhaustedError: OOM when allocating tensor
```
**Solution:**
- Kurangi `BATCH_SIZE` di `train_model.py`
- Gunakan GPU dengan memory lebih besar
- Reduce model complexity

---

## 📝 Catatan Penelitian

### Pembatasan Masalah
1. Deteksi dilakukan pada citra statis (tidak real-time video)
2. Dataset terbatas pada Fatigue Dataset dari Kaggle
3. Klasifikasi binary: Fatigued vs Non-Fatigued
4. Tidak menggunakan sensor fisiologis tambahan

### Asumsi Sistem
1. Input berupa foto wajah dengan kualitas memadai
2. Wajah terlihat jelas tanpa obstruksi signifikan
3. Pencahayaan cukup untuk deteksi wajah

### Kontribusi Penelitian
1. Implementasi CNN untuk deteksi kelelahan berbasis visual
2. Sistem web-based yang praktis dan user-friendly
3. Solusi non-invasif untuk monitoring kelelahan kerja

---

## 👨‍💻 Author

**Indra Yohanes**  
NIM: 32220135  
Program Studi Teknik Informatika  
Universitas Bunda Mulia

---

## 📄 License

Penelitian ini dikembangkan untuk keperluan akademik (Skripsi).

---

## 🙏 Acknowledgments

- Universitas Bunda Mulia
- Dosen Pembimbing
- Dataset providers (Kaggle)
- Open source communities (TensorFlow, OpenCV, Flask)

---

**Last Updated:** February 2026
