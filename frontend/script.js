// ========================================
// Configuration
// ========================================
const API_BASE_URL = 'http://localhost:5000/api';

// ========================================
// State Management
// ========================================
let selectedFile = null;
let detectionCount = 0;

// ========================================
// DOM Elements
// ========================================
const elements = {
    uploadArea: document.getElementById('uploadArea'),
    uploadContent: document.getElementById('uploadContent'),
    previewContainer: document.getElementById('previewContainer'),
    imageInput: document.getElementById('imageInput'),
    imagePreview: document.getElementById('imagePreview'),
    removeImage: document.getElementById('removeImage'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    analyzeBtnText: document.getElementById('analyzeBtnText'),
    btnLoader: document.getElementById('btnLoader'),
    resultsCard: document.getElementById('resultsCard'),
    resultsEmpty: document.getElementById('resultsEmpty'),
    resultsContent: document.getElementById('resultsContent'),
    statusIcon: document.getElementById('statusIcon'),
    statusValue: document.getElementById('statusValue'),
    confidenceValue: document.getElementById('confidenceValue'),
    faceImage: document.getElementById('faceImage'),
    recommendationMessage: document.getElementById('recommendationMessage'),
    recommendationActions: document.getElementById('recommendationActions'),
    detectionTime: document.getElementById('detectionTime'),
    accuracyStat: document.getElementById('accuracy-stat'),
    detectionStat: document.getElementById('detection-stat')
};

// ========================================
// Initialization
// ========================================
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkAPIHealth();
    animateStats();
});

function initializeEventListeners() {
    // Upload area click
    elements.uploadArea.addEventListener('click', (e) => {
        if (e.target !== elements.removeImage && !elements.removeImage.contains(e.target)) {
            elements.imageInput.click();
        }
    });

    // File input change
    elements.imageInput.addEventListener('change', handleFileSelect);

    // Remove image button
    elements.removeImage.addEventListener('click', (e) => {
        e.stopPropagation();
        removeSelectedImage();
    });

    // Analyze button
    elements.analyzeBtn.addEventListener('click', analyzeImage);

    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);

    // Prevent default drag behaviors
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());
}

// ========================================
// File Handling
// ========================================
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        validateAndPreviewFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    elements.uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file) {
        validateAndPreviewFile(file);
    }
}

function validateAndPreviewFile(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!allowedTypes.includes(file.type)) {
        showError('Format file tidak valid. Hanya JPG, JPEG, dan PNG yang diperbolehkan.');
        return;
    }

    // Validate file size (16MB)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('Ukuran file terlalu besar. Maksimal 16MB.');
        return;
    }

    selectedFile = file;
    previewImage(file);
}

function previewImage(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        elements.imagePreview.src = e.target.result;
        elements.uploadContent.style.display = 'none';
        elements.previewContainer.style.display = 'block';
        elements.analyzeBtn.disabled = false;
        elements.analyzeBtnText.textContent = 'Analisis Gambar';
        
        // Hide results
        hideResults();
    };
    
    reader.readAsDataURL(file);
}

function removeSelectedImage() {
    selectedFile = null;
    elements.imagePreview.src = '';
    elements.uploadContent.style.display = 'block';
    elements.previewContainer.style.display = 'none';
    elements.imageInput.value = '';
    elements.analyzeBtn.disabled = true;
    elements.analyzeBtnText.textContent = 'Pilih Gambar Terlebih Dahulu';
    
    // Hide results
    hideResults();
}

// ========================================
// API Calls
// ========================================
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            console.log('API is healthy:', data);
            if (data.model_loaded) {
                elements.accuracyStat.textContent = '85%+';
            }
        }
    } catch (error) {
        console.error('API health check failed:', error);
        showError('Tidak dapat terhubung ke server. Pastikan backend API berjalan.');
    }
}

async function analyzeImage() {
    if (!selectedFile) {
        showError('Tidak ada gambar yang dipilih');
        return;
    }

    // Show loading state
    setLoadingState(true);

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);

        const response = await fetch(`${API_BASE_URL}/detect`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Gagal melakukan deteksi');
        }

        if (data.success) {
            displayResults(data);
            detectionCount++;
            updateDetectionCount();
        } else {
            showError(data.error || 'Gagal melakukan deteksi');
        }

    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message || 'Terjadi kesalahan saat menganalisis gambar');
    } finally {
        setLoadingState(false);
    }
}

// ========================================
// UI Updates
// ========================================
function setLoadingState(isLoading) {
    elements.analyzeBtn.disabled = isLoading;
    
    if (isLoading) {
        elements.analyzeBtnText.style.display = 'none';
        elements.btnLoader.style.display = 'block';
    } else {
        elements.analyzeBtnText.style.display = 'block';
        elements.btnLoader.style.display = 'none';
    }
}

function displayResults(data) {
    // Hide empty state, show results
    elements.resultsEmpty.style.display = 'none';
    elements.resultsContent.style.display = 'block';

    // Update status
    const isFatigued = data.prediction === 'Fatigued';
    
    // Status icon
    elements.statusIcon.className = 'status-icon ' + (isFatigued ? 'fatigued' : 'non-fatigued');
    elements.statusIcon.textContent = isFatigued ? '⚠️' : '✓';
    
    // Status value
    elements.statusValue.textContent = data.prediction;
    elements.statusValue.className = 'status-value ' + (isFatigued ? 'fatigued' : 'non-fatigued');
    
    // Confidence
    elements.confidenceValue.textContent = data.confidence + '%';
    
    // Face image
    if (data.face_image) {
        elements.faceImage.src = 'data:image/jpeg;base64,' + data.face_image;
    }
    
    // Recommendation
    if (data.recommendation) {
        elements.recommendationMessage.textContent = data.recommendation.message;
        
        // Actions list
        elements.recommendationActions.innerHTML = '';
        data.recommendation.actions.forEach(action => {
            const li = document.createElement('li');
            li.textContent = action;
            elements.recommendationActions.appendChild(li);
        });
    }
    
    // Timestamp
    const timestamp = new Date(data.timestamp);
    elements.detectionTime.textContent = timestamp.toLocaleString('id-ID');
    
    // Scroll to results
    elements.resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideResults() {
    elements.resultsEmpty.style.display = 'flex';
    elements.resultsContent.style.display = 'none';
}

function showError(message) {
    alert('Error: ' + message);
}

function updateDetectionCount() {
    elements.detectionStat.textContent = detectionCount;
}

function animateStats() {
    // Animate accuracy stat
    let accuracy = 0;
    const targetAccuracy = 85;
    const accuracyInterval = setInterval(() => {
        accuracy += 1;
        elements.accuracyStat.textContent = accuracy + '%';
        
        if (accuracy >= targetAccuracy) {
            clearInterval(accuracyInterval);
            elements.accuracyStat.textContent = '85%+';
        }
    }, 20);
}

// ========================================
// Navigation
// ========================================
function scrollToDetection() {
    document.getElementById('detection').scrollIntoView({ behavior: 'smooth' });
}

function scrollToAbout() {
    document.getElementById('about').scrollIntoView({ behavior: 'smooth' });
}

// Navigation active state
const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('section[id]');

window.addEventListener('scroll', () => {
    let current = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        
        if (pageYOffset >= sectionTop - 200) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + current) {
            link.classList.add('active');
        }
    });
});

// ========================================
// Utility Functions
// ========================================
function formatTimestamp(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString('id-ID', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

// Make functions available globally
window.scrollToDetection = scrollToDetection;
window.scrollToAbout = scrollToAbout;
