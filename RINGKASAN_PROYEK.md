# 📊 RINGKASAN PROYEK
## Sistem Deteksi Kelelahan Kerja Berbasis CNN

**Peneliti:** Indra Yohanes (32220135)  
**Program Studi:** Teknik Informatika  
**Universitas:** Bunda Mulia  
**Tahun:** 2026

---

## 🎯 TUJUAN PENELITIAN

1. **Merancang sistem deteksi kelelahan** berbasis ekspresi wajah menggunakan CNN
2. **Membangun aplikasi web dashboard** yang user-friendly dan praktis
3. **Menganalisis performa model** dengan metrik evaluasi standar

---

## 🔬 RUMUSAN MASALAH

1. Bagaimana merancang sistem deteksi kelelahan berbasis ekspresi wajah menggunakan CNN?
2. Bagaimana kinerja model CNN dalam mendeteksi kelelahan pada kondisi mendekati real-world?
3. Bagaimana mengintegrasikan model ke dalam aplikasi web yang mudah digunakan?

---

## 💡 SOLUSI YANG DIKEMBANGKAN

### 🧠 Backend (AI/ML Engine)
✅ **Model CNN** dengan arsitektur custom untuk facial analysis
✅ **Face Detection** otomatis menggunakan OpenCV Haar Cascade
✅ **Preprocessing pipeline** untuk normalisasi input
✅ **RESTful API** menggunakan Flask
✅ **Confidence scoring** untuk setiap prediksi
✅ **Recommendation engine** berbasis hasil deteksi

### 🎨 Frontend (Web Dashboard)
✅ **Modern UI/UX** dengan desain profesional
✅ **Drag-and-drop upload** untuk kemudahan penggunaan
✅ **Real-time preview** dan hasil analisis
✅ **Visual indicators** dengan sistem warna
✅ **Responsive design** untuk semua perangkat
✅ **Detailed recommendations** untuk tindak lanjut

### 🔧 Model Training Pipeline
✅ **Data augmentation** untuk generalisasi model
✅ **Batch normalization** dan dropout untuk regularisasi
✅ **Early stopping** dan model checkpointing
✅ **Comprehensive evaluation** dengan multiple metrics
✅ **Visualization** hasil training dan confusion matrix

---

## 🏗️ ARSITEKTUR TEKNIS

```
┌──────────────────────────────────────────┐
│           WEB BROWSER (Frontend)          │
│  ┌────────────────────────────────────┐  │
│  │  • Upload Interface                │  │
│  │  • Results Display                 │  │
│  │  • Recommendations                 │  │
│  └────────────────────────────────────┘  │
└──────────────┬───────────────────────────┘
               │ HTTP/REST API
               ▼
┌──────────────────────────────────────────┐
│         FLASK API SERVER (Backend)        │
│  ┌────────────────────────────────────┐  │
│  │  1. Image Upload Handler           │  │
│  │  2. Face Detection (OpenCV)        │  │
│  │  3. Image Preprocessing            │  │
│  │  4. CNN Model Inference            │  │
│  │  5. Results + Recommendations      │  │
│  └────────────────────────────────────┘  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│      CNN MODEL (TensorFlow/Keras)        │
│  ┌────────────────────────────────────┐  │
│  │  • 4 Convolutional Blocks          │  │
│  │  • Batch Normalization Layers      │  │
│  │  • Max Pooling + Dropout           │  │
│  │  • Dense Layers (512→256→1)        │  │
│  │  • Binary Classification Output    │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

---

## 📈 TARGET PERFORMA

| Metrik | Target Minimal | Alasan |
|--------|---------------|---------|
| **Accuracy** | ≥ 85% | Akurasi keseluruhan sistem |
| **Precision** | ≥ 80% | Mengurangi false alarm |
| **Recall** | ≥ 85% | Prioritas keselamatan (detect semua kasus lelah) |
| **F1-Score** | ≥ 83% | Balance antara precision dan recall |

**Catatan:** Recall diprioritaskan karena dalam konteks keselamatan kerja, lebih baik sistem memberi peringatan palsu (FP) daripada melewatkan kasus kelelahan sebenarnya (FN).

---

## 🛠️ TEKNOLOGI YANG DIGUNAKAN

### Backend Stack
- **Python 3.9+** - Programming language
- **Flask** - Web framework
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Computer vision library
- **NumPy** - Numerical computing
- **Pillow** - Image processing

### Frontend Stack
- **HTML5** - Semantic markup
- **CSS3** - Modern styling (Grid, Flexbox, Animations)
- **JavaScript (Vanilla)** - Dynamic functionality
- **Fetch API** - Asynchronous HTTP requests

### Development Tools
- **Git** - Version control
- **VSCode** - Code editor
- **Postman** - API testing
- **Docker** - Containerization (optional)

---

## 📊 METODOLOGI PENGEMBANGAN

Menggunakan **Software Development Life Cycle (SDLC)** dengan model Waterfall:

1. **📋 Analisis Kebutuhan** (2 minggu)
   - Studi literatur CNN dan facial expression recognition
   - Analisis dataset Fatigue dari Kaggle
   - Identifikasi requirements sistem

2. **🎨 Perancangan Sistem** (2 minggu)
   - Desain arsitektur CNN
   - Perancangan API endpoints
   - UI/UX design web dashboard

3. **🔨 Implementasi** (8 minggu)
   - Preprocessing data dan augmentation (1 minggu)
   - Training model CNN (3 minggu)
   - Development backend API (2 minggu)
   - Development frontend dashboard (3 minggu)

4. **🧪 Pengujian** (2 minggu)
   - Unit testing backend
   - Integration testing API-Frontend
   - Usability testing
   - Performance evaluation

5. **🚀 Deployment & Dokumentasi** (2 minggu)
   - Deployment ke server
   - Dokumentasi API
   - User manual
   - Laporan skripsi

**Total Durasi:** 16 minggu

---

## 🎓 KONTRIBUSI PENELITIAN

### Kontribusi Akademis
1. Implementasi CNN untuk deteksi kelelahan berbasis visual cues
2. Analisis performa model pada dataset real-world
3. Studi komparatif metode non-invasif vs invasif

### Kontribusi Praktis
1. Sistem deteksi kelelahan yang **mudah digunakan**
2. Solusi **non-invasif** tanpa sensor fisik
3. **Web-based platform** yang accessible dari mana saja
4. **Real-time detection** untuk monitoring aktif

### Kontribusi Teknologis
1. **Open-source codebase** untuk penelitian lanjutan
2. **Modular architecture** untuk pengembangan fitur baru
3. **Scalable system** untuk deployment production
4. **Comprehensive documentation** untuk maintenance

---

## 📁 DELIVERABLES

### 1. Source Code
```
fatigue-detection-system/
├── backend/
│   ├── app.py                    # Flask API
│   └── requirements.txt          # Dependencies
├── frontend/
│   ├── index.html                # Main interface
│   ├── style.css                 # Styling
│   └── script.js                 # Functionality
├── model/
│   └── train_model.py            # Training script
└── Documentation/
    ├── README.md                 # Main documentation
    ├── QUICKSTART.md             # Quick start guide
    └── DEPLOYMENT.md             # Deployment guide
```

### 2. Trained Model
- `fatigue_detection_model.h5` - CNN model siap pakai
- Confusion matrix visualization
- Training history graphs

### 3. Dokumentasi
- README lengkap dengan setup instructions
- API documentation
- User manual
- Deployment guide
- Quick start guide

### 4. Laporan Penelitian
- BAB I: Pendahuluan
- BAB II: Tinjauan Pustaka
- BAB III: Analisis dan Perancangan
- BAB IV: Implementasi dan Pengujian
- BAB V: Kesimpulan dan Saran

---

## 🎯 FITUR UTAMA SISTEM

### Untuk User
1. ✅ **Upload gambar** dengan drag-and-drop
2. ✅ **Hasil instant** dalam hitungan detik
3. ✅ **Visual feedback** yang jelas (warna, ikon)
4. ✅ **Rekomendasi aksi** berdasarkan hasil
5. ✅ **Confidence score** untuk transparansi
6. ✅ **Face detection** visualization

### Untuk Peneliti
1. ✅ **Modular codebase** untuk eksperimen
2. ✅ **Training pipeline** yang customizable
3. ✅ **Evaluation metrics** comprehensive
4. ✅ **Visualization tools** untuk analisis
5. ✅ **API endpoints** untuk integrasi
6. ✅ **Logging system** untuk monitoring

---

## 🔄 FUTURE ENHANCEMENTS

### Short-term (3-6 bulan)
- [ ] Real-time video detection
- [ ] Multi-face detection
- [ ] Mobile app (iOS/Android)
- [ ] Database untuk history tracking
- [ ] Advanced analytics dashboard

### Long-term (6-12 bulan)
- [ ] Multi-class classification (tingkat kelelahan)
- [ ] Transfer learning dengan pre-trained models
- [ ] Edge deployment (Raspberry Pi)
- [ ] Integration dengan IoT sensors
- [ ] Machine learning pipeline automation

---

## 💼 POTENSI APLIKASI

### 1. **Industri Manufaktur**
- Monitoring pekerja shift panjang
- Deteksi kelelahan operator mesin
- Pencegahan kecelakaan kerja

### 2. **Transportasi**
- Deteksi kelelahan pengemudi
- Safety system untuk truk/bus
- Monitoring pilot pesawat

### 3. **Healthcare**
- Monitoring tenaga medis
- Deteksi burnout dokter
- Patient safety monitoring

### 4. **Pendidikan**
- Monitoring siswa/mahasiswa
- E-learning engagement analysis
- Study fatigue detection

---

## 📞 INFORMASI KONTAK

**Peneliti:** Indra Yohanes  
**NIM:** 32220135  
**Email:** [email mahasiswa]  
**Institusi:** Universitas Bunda Mulia  
**Program Studi:** Teknik Informatika  
**Tahun:** 2026

---

## 📄 LISENSI

Penelitian ini dikembangkan untuk keperluan akademik (Skripsi).  
Source code tersedia untuk tujuan penelitian dan pendidikan.

---

## 🙏 ACKNOWLEDGMENTS

Terima kasih kepada:
- Dosen pembimbing yang telah membimbing penelitian ini
- Universitas Bunda Mulia atas fasilitas penelitian
- Komunitas open-source (TensorFlow, OpenCV, Flask)
- Dataset providers di Kaggle
- Keluarga dan teman-teman atas dukungannya

---

**"Technology should work for people, not the other way around."**

---

*Dokumen ini dibuat: February 2026*
