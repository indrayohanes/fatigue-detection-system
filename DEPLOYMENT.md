# Panduan Deployment - Sistem Deteksi Kelelahan Kerja

Dokumen ini menjelaskan berbagai cara untuk deploy sistem deteksi kelelahan kerja ke production.

---

## 📋 Daftar Isi

1. [Deployment Lokal](#deployment-lokal)
2. [Deployment dengan Docker](#deployment-dengan-docker)
3. [Deployment ke Heroku](#deployment-ke-heroku)
4. [Deployment ke Google Cloud Platform](#deployment-ke-google-cloud-platform)
5. [Deployment ke AWS](#deployment-ke-aws)

---

## 1. Deployment Lokal

### Setup untuk Development

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py

# Frontend (terminal baru)
cd frontend
python -m http.server 8080
```

### Setup untuk Production (Gunicorn)

```bash
# Install Gunicorn
pip install gunicorn

# Jalankan dengan Gunicorn
cd backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## 2. Deployment dengan Docker

### 2.1 Dockerfile untuk Backend

Buat `backend/Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Copy model (jika ada)
COPY ../model/saved_models/best_model.h5 ./fatigue_detection_model.h5

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### 2.2 Dockerfile untuk Frontend

Buat `frontend/Dockerfile`:

```dockerfile
FROM nginx:alpine

# Copy static files
COPY . /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### 2.3 Docker Compose

Buat `docker-compose.yml` di root:

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "5000:5000"
    volumes:
      - ./backend/uploads:/app/uploads
      - ./model/saved_models:/app/models
    environment:
      - FLASK_ENV=production
    networks:
      - fatigue-net

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    networks:
      - fatigue-net

networks:
  fatigue-net:
    driver: bridge
```

### 2.4 Menjalankan dengan Docker

```bash
# Build dan jalankan
docker-compose up -d

# Stop
docker-compose down

# Rebuild
docker-compose up -d --build
```

---

## 3. Deployment ke Heroku

### 3.1 Persiapan File

Buat `backend/Procfile`:
```
web: gunicorn app:app
```

Buat `backend/runtime.txt`:
```
python-3.9.16
```

Update `backend/requirements.txt`, tambahkan:
```
gunicorn==20.1.0
```

### 3.2 Deployment Steps

```bash
# Login ke Heroku
heroku login

# Create app
heroku create fatigue-detection-api

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
cd backend
git init
git add .
git commit -m "Initial commit"
git push heroku master

# Scale dyno
heroku ps:scale web=1

# Check logs
heroku logs --tail
```

### 3.3 Environment Variables

```bash
heroku config:set FLASK_ENV=production
heroku config:set MAX_UPLOAD_SIZE=16777216
```

### 3.4 Upload Model ke Heroku

Model file besar tidak bisa di-push ke Git. Solusi:

**Opsi 1: Menggunakan Git LFS**
```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
git commit -m "Add LFS tracking"
```

**Opsi 2: Upload ke Cloud Storage**
- Upload model ke Google Drive / AWS S3
- Download saat aplikasi start
- Update `app.py` untuk download model

---

## 4. Deployment ke Google Cloud Platform

### 4.1 Setup GCP

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### 4.2 App Engine Deployment

Buat `backend/app.yaml`:
```yaml
runtime: python39

instance_class: F2

env_variables:
  FLASK_ENV: "production"

handlers:
- url: /static
  static_dir: static

- url: /.*
  script: auto
```

Deploy:
```bash
cd backend
gcloud app deploy
gcloud app browse
```

### 4.3 Cloud Run Deployment

```bash
# Build container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/fatigue-backend

# Deploy
gcloud run deploy fatigue-backend \
  --image gcr.io/YOUR_PROJECT_ID/fatigue-backend \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated
```

---

## 5. Deployment ke AWS

### 5.1 Elastic Beanstalk

Buat `.ebextensions/python.config`:
```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: app:app
```

Deploy:
```bash
# Install EB CLI
pip install awsebcli

# Initialize
cd backend
eb init -p python-3.9 fatigue-detection

# Create environment
eb create fatigue-detection-env

# Deploy
eb deploy

# Open
eb open
```

### 5.2 EC2 Manual Setup

```bash
# SSH ke EC2
ssh -i your-key.pem ec2-user@your-instance-ip

# Update system
sudo yum update -y

# Install Python 3.9
sudo yum install python39 -y

# Clone repository
git clone <your-repo-url>
cd fatigue-detection-system/backend

# Setup virtual environment
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install and setup Nginx
sudo yum install nginx -y

# Configure Nginx (create /etc/nginx/conf.d/fatigue.conf)
sudo nano /etc/nginx/conf.d/fatigue.conf
```

Nginx config:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Setup systemd service `/etc/systemd/system/fatigue.service`:
```ini
[Unit]
Description=Fatigue Detection API
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/fatigue-detection-system/backend
Environment="PATH=/home/ec2-user/fatigue-detection-system/backend/venv/bin"
ExecStart=/home/ec2-user/fatigue-detection-system/backend/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app

[Install]
WantedBy=multi-user.target
```

Start services:
```bash
sudo systemctl start fatigue
sudo systemctl enable fatigue
sudo systemctl start nginx
sudo systemctl enable nginx
```

---

## 📊 Performance Optimization

### Backend Optimization

1. **Caching**
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@cache.memoize(timeout=300)
def predict_fatigue(preprocessed_img):
    # ... prediction logic
```

2. **Model Loading**
```python
# Load model once at startup, not per request
MODEL = None

def load_model():
    global MODEL
    if MODEL is None:
        MODEL = keras.models.load_model(MODEL_PATH)
```

3. **Request Size Limit**
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

### Frontend Optimization

1. **Image Compression** sebelum upload
2. **Lazy Loading** untuk images
3. **Minify** CSS dan JavaScript
4. **CDN** untuk static assets

---

## 🔒 Security Considerations

### Backend Security

1. **CORS Configuration**
```python
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://your-frontend-domain.com"],
        "methods": ["GET", "POST"]
    }
})
```

2. **Rate Limiting**
```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per hour"]
)

@app.route('/api/detect', methods=['POST'])
@limiter.limit("10 per minute")
def detect_fatigue_endpoint():
    # ...
```

3. **File Upload Validation**
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024

def validate_file(file):
    # Check extension
    # Check size
    # Check mime type
    # Scan for malicious content
```

### HTTPS Setup

1. **Let's Encrypt (Free SSL)**
```bash
sudo certbot --nginx -d your-domain.com
```

2. **Cloudflare** (Free SSL + CDN)
- Add domain to Cloudflare
- Update DNS
- Enable SSL/TLS

---

## 🔍 Monitoring

### Logging

```python
import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

@app.route('/api/detect', methods=['POST'])
def detect_fatigue_endpoint():
    logging.info(f"Detection request from {request.remote_addr}")
    # ...
```

### Error Tracking

Gunakan Sentry untuk tracking errors:
```bash
pip install sentry-sdk[flask]
```

```python
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FlaskIntegration()]
)
```

---

## 📝 Checklist Pre-Deployment

- [ ] Model sudah trained dan tested
- [ ] Environment variables sudah diset
- [ ] Database connection tested (jika ada)
- [ ] CORS configured untuk domain production
- [ ] SSL/HTTPS enabled
- [ ] Error handling comprehensive
- [ ] Logging configured
- [ ] Rate limiting enabled
- [ ] File upload validation
- [ ] Backend tested dengan production settings
- [ ] Frontend API URL updated untuk production
- [ ] Performance testing done
- [ ] Security audit completed
- [ ] Backup strategy defined
- [ ] Monitoring setup

---

**Happy Deploying! 🚀**
