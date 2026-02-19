# 🚀 QUICK START GUIDE
## Sistem Deteksi Kelelahan Kerja - CNN

---

## ⚡ Cara Cepat Memulai (5 Menit)

### 1️⃣ Persiapan Dataset

**Download Fatigue Dataset dari Kaggle:**
- URL: https://www.kaggle.com/datasets (cari "fatigue detection dataset")
- Extract dataset dengan struktur:
```
model/dataset/
├── train/
│   ├── fatigued/
│   └── non_fatigued/
└── test/
    ├── fatigued/
    └── non_fatigued/
```

### 2️⃣ Training Model (Jika Belum Ada Model)

```bash
cd model
pip install tensorflow keras numpy matplotlib scikit-learn opencv-python

python train_model.py
```

**Output:**
- `saved_models/best_model.h5` ← Model terbaik
- `saved_models/confusion_matrix.png` ← Evaluasi visual
- `saved_models/training_history.png` ← Grafik training

⏱️ **Waktu Training:** ~30-60 menit (tergantung hardware)

### 3️⃣ Setup Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Copy model yang sudah di-train
cp ../model/saved_models/best_model.h5 ./fatigue_detection_model.h5

# Jalankan server
python app.py
```

✅ **Backend berjalan di:** http://localhost:5000

### 4️⃣ Setup Frontend

**Buka terminal baru:**

```bash
cd frontend

# Opsi 1: Langsung buka index.html di browser
# (double-click index.html)

# Opsi 2: Menggunakan Python HTTP Server
python -m http.server 8080
```

✅ **Frontend berjalan di:** http://localhost:8080

### 5️⃣ Gunakan Sistem

1. **Buka browser** → http://localhost:8080
2. **Upload gambar wajah** (klik atau drag-drop)
3. **Klik "Analisis Gambar"**
4. **Lihat hasil deteksi!**

---

## 🎯 Testing dengan Gambar Sample

**Gunakan gambar untuk testing:**
- ✅ Foto wajah frontal
- ✅ Pencahayaan baik
- ✅ Wajah terlihat jelas
- ❌ Jangan pakai foto dengan wajah tertutup masker
- ❌ Jangan pakai foto dengan kacamata hitam

---

## 🐛 Troubleshooting Cepat

### Problem: Backend tidak bisa start
```bash
# Cek apakah port 5000 sudah digunakan
# Windows:
netstat -ano | findstr :5000

# Linux/Mac:
lsof -i :5000

# Solusi: Ganti port di app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Problem: Model tidak ditemukan
```bash
# Pastikan file ada
ls -la backend/fatigue_detection_model.h5

# Atau copy ulang
cp model/saved_models/best_model.h5 backend/fatigue_detection_model.h5
```

### Problem: CORS Error di frontend
```bash
# Pastikan backend sudah install flask-cors
pip install flask-cors

# Restart backend
python app.py
```

---

## 📁 File Penting

| File | Lokasi | Fungsi |
|------|--------|--------|
| Model | `backend/fatigue_detection_model.h5` | Model CNN trained |
| Backend API | `backend/app.py` | Flask API server |
| Frontend | `frontend/index.html` | Web interface |
| Training Script | `model/train_model.py` | Script training model |
| README | `README.md` | Dokumentasi lengkap |

---

## 🎓 Untuk Presentasi Skripsi

### Demo Live

1. **Persiapkan 3-5 gambar sample:**
   - 2-3 gambar "fatigued"
   - 2-3 gambar "non-fatigued"

2. **Flow demo:**
   - Tunjukkan landing page
   - Explain fitur sistem
   - Upload gambar non-fatigued → lihat hasil
   - Upload gambar fatigued → lihat hasil + rekomendasi
   - Tunjukkan confidence score
   - Explain metrik evaluasi

3. **Siapkan backup:**
   - Screenshot hasil deteksi
   - Video recording demo
   - Presentasi slides dengan hasil training

### Menampilkan Hasil Training

```bash
# Open hasil training
cd model/saved_models

# Tunjukkan:
# 1. confusion_matrix.png
# 2. training_history.png
# 3. Nilai accuracy, precision, recall, F1-score
```

---

## 📊 Expected Results (Target)

| Metrik | Target | Keterangan |
|--------|--------|------------|
| Accuracy | ≥85% | Akurasi keseluruhan |
| Precision | ≥80% | Mengurangi false positive |
| Recall | ≥85% | Penting untuk keselamatan |
| F1-Score | ≥83% | Balance precision-recall |

---

## 🔄 Development Workflow

### Untuk Improvement Model

```bash
# 1. Modify arsitektur CNN di train_model.py
# 2. Re-train model
python model/train_model.py

# 3. Copy model baru
cp model/saved_models/best_model.h5 backend/fatigue_detection_model.h5

# 4. Restart backend
# Backend akan load model baru
```

### Untuk Update UI/UX

```bash
# 1. Edit frontend/style.css atau frontend/index.html
# 2. Refresh browser (Ctrl+F5 untuk hard refresh)
# No need restart server!
```

### Untuk Update Backend Logic

```bash
# 1. Edit backend/app.py
# 2. Restart Flask server
# Ctrl+C lalu python app.py
```

---

## 📚 Next Steps

1. **Baca README.md lengkap** untuk detail teknis
2. **Baca DEPLOYMENT.md** untuk deploy ke production
3. **Eksperimen dengan dataset** lebih besar
4. **Fine-tune hyperparameter** untuk hasil lebih baik
5. **Tambah fitur** seperti batch processing atau analytics

---

## 💡 Tips Pro

1. **Gunakan GPU** untuk training lebih cepat
2. **Data augmentation** untuk dataset kecil
3. **Transfer learning** (VGG, ResNet) untuk akurasi lebih tinggi
4. **Logging deteksi** untuk analisis long-term
5. **A/B testing** berbagai arsitektur CNN

---

## 📞 Butuh Bantuan?

- 📖 Baca **README.md** untuk dokumentasi lengkap
- 🐛 Check **Troubleshooting** section
- 💬 Review code comments untuk penjelasan detail
- 🔍 Debug dengan `print()` statements

---

**Good luck dengan penelitian skripsi! 🎓✨**

---

*Last updated: February 2026*
