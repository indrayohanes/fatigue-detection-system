// ========================================
// Configuration
// ========================================
const API_BASE_URL = 'http://localhost:5000/api';

// ========================================
// State
// ========================================
let selectedFile = null;
let detectionCount = 0;
let authToken = localStorage.getItem('authToken') || null;
let currentUser = null;
let currentHistoryPage = 1;

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
    resultStatusBanner: document.getElementById('resultStatusBanner'),
    statusIcon: document.getElementById('statusIcon'),
    statusValue: document.getElementById('statusValue'),
    confidenceValue: document.getElementById('confidenceValue'),
    confidenceFill: document.getElementById('confidenceFill'),
    faceImage: document.getElementById('faceImage'),
    recommendationMessage: document.getElementById('recommendationMessage'),
    recommendationActions: document.getElementById('recommendationActions'),
    detectionTime: document.getElementById('detectionTime'),
    accuracyStat: document.getElementById('accuracy-stat'),
    detectionStat: document.getElementById('detection-stat'),
    toastContainer: document.getElementById('toastContainer'),
    // Grad-CAM elements
    gradcamCard: document.getElementById('gradcamCard'),
    gradcamImage: document.getElementById('gradcamImage'),
    heatmapNote: document.getElementById('heatmapNote'),
    regionAnalysisCard: document.getElementById('regionAnalysisCard'),
    regionBars: document.getElementById('regionBars'),
    aiExplanationCard: document.getElementById('aiExplanationCard'),
    aiExplanationText: document.getElementById('aiExplanationText')
};

// ========================================
// Initialization
// ========================================
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkAPIHealth();
    checkAuth();
    animateStats();
    initScrollAnimations();
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

    // Remove image
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

    // Prevent default drag behaviors globally
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());
}

// ========================================
// Toast Notification System
// ========================================
function showToast(message, type = 'info', duration = 5000) {
    const container = elements.toastContainer;

    const icons = {
        error: '⚠️',
        success: '✅',
        info: 'ℹ️'
    };

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span class="toast-message">${message}</span>
        <button class="toast-close" onclick="this.parentElement.classList.add('toast-exit'); setTimeout(() => this.parentElement.remove(), 300)">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
    `;

    container.appendChild(toast);

    // Auto remove
    setTimeout(() => {
        if (toast.parentElement) {
            toast.classList.add('toast-exit');
            setTimeout(() => toast.remove(), 300);
        }
    }, duration);
}

function showError(message) {
    showToast(message, 'error');
}

function showSuccess(message) {
    showToast(message, 'success');
}

// ========================================
// File Handling
// ========================================
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) validateAndPreviewFile(file);
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
    if (file) validateAndPreviewFile(file);
}

function validateAndPreviewFile(file) {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!allowedTypes.includes(file.type)) {
        showError('Format file tidak valid. Hanya JPG, JPEG, dan PNG yang diperbolehkan.');
        return;
    }

    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('Ukuran file terlalu besar. Maksimal 16MB.');
        return;
    }

    selectedFile = file;
    previewImage(file);
    showToast('Gambar berhasil dimuat. Klik "Analisis Gambar" untuk memulai deteksi.', 'info', 3000);
}

function previewImage(file) {
    const reader = new FileReader();

    reader.onload = (e) => {
        elements.imagePreview.src = e.target.result;
        elements.uploadContent.style.display = 'none';
        elements.previewContainer.style.display = 'flex';
        elements.analyzeBtn.disabled = false;
        elements.analyzeBtnText.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
            Analisis Gambar
        `;
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
    elements.analyzeBtnText.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
        Pilih Gambar Terlebih Dahulu
    `;
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
            console.log('✅ API connected:', data);
            if (data.model_loaded) {
                elements.accuracyStat.textContent = '85%+';
            }
        }
    } catch (error) {
        console.error('API health check failed:', error);
        showError('Tidak dapat terhubung ke server. Pastikan backend API berjalan di port 5000.');
    }
}

async function analyzeImage() {
    if (!selectedFile) {
        showError('Tidak ada gambar yang dipilih');
        return;
    }

    setLoadingState(true);

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);

        const headers = {};
        if (authToken) {
            headers['Authorization'] = 'Bearer ' + authToken;
        }

        const response = await fetch(`${API_BASE_URL}/detect`, {
            method: 'POST',
            headers: headers,
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
            showSuccess('Analisis selesai!');
            // Refresh history immediately
            if (authToken) loadHistory();
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
        elements.btnLoader.style.display = 'flex';
    } else {
        elements.analyzeBtnText.style.display = 'flex';
        elements.btnLoader.style.display = 'none';
    }
}

function displayResults(data) {
    // Show results
    elements.resultsEmpty.style.display = 'none';
    elements.resultsContent.style.display = 'flex';

    const isFatigued = data.prediction === 'Fatigued';

    // Status banner
    if (elements.resultStatusBanner) {
        elements.resultStatusBanner.className = `result-status-banner ${isFatigued ? 'fatigued' : 'non-fatigued'}`;
    }
    if (elements.statusIcon) elements.statusIcon.textContent = isFatigued ? '⚠️' : '✅';
    if (elements.statusValue) elements.statusValue.textContent = isFatigued ? 'Terdeteksi Kelelahan' : 'Tidak Kelelahan';

    // Confidence with animated bar
    const confidence = data.confidence;
    if (elements.confidenceValue) elements.confidenceValue.textContent = confidence + '%';

    // Animate the confidence bar
    if (elements.confidenceFill) {
        elements.confidenceFill.style.width = '0%';
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                elements.confidenceFill.style.width = confidence + '%';
            });
        });
    }

    // Face image
    if (data.face_image) {
        elements.faceImage.src = 'data:image/jpeg;base64,' + data.face_image;
    }

    // Grad-CAM Heatmap
    if (elements.gradcamCard) {
        if (data.gradcam_image) {
            elements.gradcamCard.style.display = 'flex';
            if (elements.gradcamImage) elements.gradcamImage.src = 'data:image/jpeg;base64,' + data.gradcam_image;
            if (elements.heatmapNote) elements.heatmapNote.style.display = 'flex';
        } else {
            elements.gradcamCard.style.display = 'none';
            if (elements.heatmapNote) elements.heatmapNote.style.display = 'none';
        }
    }

    // Face Region Analysis
    if (elements.regionAnalysisCard) {
        if (data.face_regions && data.face_regions.length > 0) {
            elements.regionAnalysisCard.style.display = 'block';
            if (elements.regionBars) {
                elements.regionBars.innerHTML = data.face_regions.map((region, index) => `
                    <div class="region-bar-item">
                        <div class="region-bar-header">
                            <span class="region-bar-name">${region.region}</span>
                            <span class="region-bar-value">${region.contribution}%</span>
                        </div>
                        <div class="region-bar-track">
                            <div class="region-bar-fill fill-${index + 1}" style="width: 0%" data-width="${region.contribution}"></div>
                        </div>
                    </div>
                `).join('');

                // Animate the bars
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        document.querySelectorAll('.region-bar-fill').forEach(bar => {
                            bar.style.width = bar.dataset.width + '%';
                        });
                    });
                });
            }
        } else {
            elements.regionAnalysisCard.style.display = 'none';
        }
    }

    // AI Explanation
    if (elements.aiExplanationCard) {
        if (data.explanation) {
            elements.aiExplanationCard.style.display = 'flex';
            if (elements.aiExplanationText) elements.aiExplanationText.innerHTML = data.explanation.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        } else {
            elements.aiExplanationCard.style.display = 'none';
        }
    }

    // Recommendation (generate locally since API may not provide it)
    if (isFatigued) {
        elements.recommendationMessage.textContent = 'Anda terdeteksi mengalami kelelahan. Segera lakukan istirahat untuk menjaga keselamatan dan produktivitas kerja.';
        elements.recommendationActions.innerHTML = `
            <li>Istirahat selama 15-30 menit</li>
            <li>Cuci muka dengan air dingin</li>
            <li>Lakukan peregangan ringan</li>
            <li>Minum air putih yang cukup</li>
        `;
    } else {
        elements.recommendationMessage.textContent = 'Kondisi Anda terlihat baik. Tetap jaga kesehatan dan produktivitas kerja Anda.';
        elements.recommendationActions.innerHTML = `
            <li>Tetap jaga pola istirahat yang teratur</li>
            <li>Pertahankan hidrasi yang baik</li>
            <li>Lakukan pengecekan berkala</li>
        `;
    }

    // Timestamp
    const timestamp = new Date(data.timestamp);
    elements.detectionTime.textContent = timestamp.toLocaleString('id-ID');

    // Scroll to results
    elements.resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideResults() {
    if (elements.resultsEmpty) elements.resultsEmpty.style.display = 'flex';
    if (elements.resultsContent) elements.resultsContent.style.display = 'none';
    if (elements.confidenceFill) elements.confidenceFill.style.width = '0%';
    // Hide Grad-CAM elements
    if (elements.gradcamCard) elements.gradcamCard.style.display = 'none';
    if (elements.heatmapNote) elements.heatmapNote.style.display = 'none';
    if (elements.regionAnalysisCard) elements.regionAnalysisCard.style.display = 'none';
    if (elements.aiExplanationCard) elements.aiExplanationCard.style.display = 'none';
}

function updateDetectionCount() {
    elements.detectionStat.textContent = detectionCount;
}

function animateStats() {
    let accuracy = 0;
    const targetAccuracy = 85;
    const interval = setInterval(() => {
        accuracy += 2;
        elements.accuracyStat.textContent = accuracy + '%';

        if (accuracy >= targetAccuracy) {
            clearInterval(interval);
            elements.accuracyStat.textContent = '85%+';
        }
    }, 25);
}

// ========================================
// Scroll Animations
// ========================================
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    // Observe sections
    document.querySelectorAll('.panel, .feature-card, .tech-card, .metrics-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// Add visible class styles
document.head.insertAdjacentHTML('beforeend', `
    <style>
        .visible {
            opacity: 1 !important;
            transform: translateY(0) !important;
        }
    </style>
`);

// ========================================
// Navigation
// ========================================
function scrollToDetection() {
    document.getElementById('detection').scrollIntoView({ behavior: 'smooth' });
}

function scrollToAbout() {
    document.getElementById('about').scrollIntoView({ behavior: 'smooth' });
}

function toggleMobileNav() {
    const navLinks = document.querySelector('.nav-links');
    navLinks.classList.toggle('show');
}

// Active nav link on scroll
const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('section[id]');

window.addEventListener('scroll', () => {
    let current = '';

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
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

// Make functions global
window.scrollToDetection = scrollToDetection;
window.scrollToAbout = scrollToAbout;
window.toggleMobileNav = toggleMobileNav;

// ========================================
// Auth Functions
// ========================================

function getAuthHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (authToken) headers['Authorization'] = 'Bearer ' + authToken;
    return headers;
}

function showAuthModal() {
    document.getElementById('authModal').style.display = 'flex';
    switchAuthTab('login');
}

function hideAuthModal() {
    document.getElementById('authModal').style.display = 'none';
    document.getElementById('loginError').style.display = 'none';
    document.getElementById('registerError').style.display = 'none';
    document.getElementById('loginForm').reset();
    document.getElementById('registerForm').reset();
}

function switchAuthTab(tab) {
    const loginTab = document.getElementById('loginTab');
    const registerTab = document.getElementById('registerTab');
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    
    if (tab === 'login') {
        loginTab.classList.add('active');
        registerTab.classList.remove('active');
        loginForm.style.display = 'block';
        registerForm.style.display = 'none';
    } else {
        loginTab.classList.remove('active');
        registerTab.classList.add('active');
        loginForm.style.display = 'none';
        registerForm.style.display = 'block';
    }
    document.getElementById('loginError').style.display = 'none';
    document.getElementById('registerError').style.display = 'none';
}

async function handleLogin(e) {
    e.preventDefault();
    const username = document.getElementById('loginUsername').value.trim();
    const password = document.getElementById('loginPassword').value;
    const errorEl = document.getElementById('loginError');
    const btn = document.getElementById('loginSubmitBtn');
    
    btn.disabled = true;
    btn.textContent = 'Memproses...';
    errorEl.style.display = 'none';
    
    try {
        const response = await fetch(`${API_BASE_URL}/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        const data = await response.json();
        
        if (data.success) {
            authToken = data.token;
            currentUser = data.user;
            localStorage.setItem('authToken', authToken);
            updateAuthUI(true);
            hideAuthModal();
            showSuccess(`Selamat datang, ${data.user.username}!`);
            loadHistory();
        } else {
            errorEl.textContent = data.error || 'Login gagal';
            errorEl.style.display = 'block';
        }
    } catch (err) {
        errorEl.textContent = 'Tidak dapat terhubung ke server';
        errorEl.style.display = 'block';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Masuk';
    }
}

async function handleRegister(e) {
    e.preventDefault();
    const username = document.getElementById('registerUsername').value.trim();
    const password = document.getElementById('registerPassword').value;
    const password2 = document.getElementById('registerPassword2').value;
    const errorEl = document.getElementById('registerError');
    const btn = document.getElementById('registerSubmitBtn');
    
    if (password !== password2) {
        errorEl.textContent = 'Password tidak cocok';
        errorEl.style.display = 'block';
        return;
    }
    
    btn.disabled = true;
    btn.textContent = 'Memproses...';
    errorEl.style.display = 'none';
    
    try {
        const response = await fetch(`${API_BASE_URL}/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        const data = await response.json();
        
        if (data.success) {
            authToken = data.token;
            currentUser = data.user;
            localStorage.setItem('authToken', authToken);
            updateAuthUI(true);
            hideAuthModal();
            showSuccess(`Akun berhasil dibuat! Selamat datang, ${data.user.username}!`);
            loadHistory();
        } else {
            errorEl.textContent = data.error || 'Registrasi gagal';
            errorEl.style.display = 'block';
        }
    } catch (err) {
        errorEl.textContent = 'Tidak dapat terhubung ke server';
        errorEl.style.display = 'block';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Daftar';
    }
}

async function handleLogout() {
    try {
        await fetch(`${API_BASE_URL}/logout`, {
            method: 'POST',
            headers: { 'Authorization': 'Bearer ' + authToken }
        });
    } catch (e) { /* ignore */ }
    
    authToken = null;
    currentUser = null;
    localStorage.removeItem('authToken');
    updateAuthUI(false);
    showSuccess('Berhasil logout');
}

async function checkAuth() {
    if (!authToken) {
        updateAuthUI(false);
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/me`, {
            headers: { 'Authorization': 'Bearer ' + authToken }
        });
        const data = await response.json();
        
        if (data.success) {
            currentUser = data.user;
            updateAuthUI(true);
            loadHistory();
        } else {
            authToken = null;
            localStorage.removeItem('authToken');
            updateAuthUI(false);
        }
    } catch (err) {
        updateAuthUI(false);
    }
}

function updateAuthUI(isLoggedIn) {
    const authBtn = document.getElementById('authBtn');
    const userMenu = document.getElementById('userMenu');
    const userName = document.getElementById('userName');
    const navHistory = document.getElementById('navHistory');
    const historySection = document.getElementById('history');
    
    if (isLoggedIn && currentUser) {
        authBtn.style.display = 'none';
        userMenu.style.display = 'flex';
        userName.textContent = currentUser.username;
        navHistory.style.display = 'inline-block';
        historySection.style.display = 'block';
    } else {
        authBtn.style.display = 'inline-flex';
        userMenu.style.display = 'none';
        userName.textContent = '';
        navHistory.style.display = 'none';
        historySection.style.display = 'none';
    }
}

// ========================================
// History Functions
// ========================================

async function loadHistory(page = 1) {
    if (!authToken) return;
    currentHistoryPage = page;
    
    try {
        const response = await fetch(`${API_BASE_URL}/history?page=${page}&per_page=10`, {
            headers: { 'Authorization': 'Bearer ' + authToken }
        });
        const data = await response.json();
        
        if (!data.success) return;
        
        const listEl = document.getElementById('historyList');
        const emptyEl = document.getElementById('historyEmpty');
        const countEl = document.getElementById('historyCount');
        const paginationEl = document.getElementById('historyPagination');
        
        countEl.textContent = `${data.total} hasil deteksi`;
        
        if (data.history.length === 0) {
            listEl.innerHTML = '';
            listEl.appendChild(emptyEl);
            emptyEl.style.display = 'flex';
            paginationEl.style.display = 'none';
            return;
        }
        
        emptyEl.style.display = 'none';
        listEl.innerHTML = data.history.map(item => renderHistoryItem(item)).join('');
        
        // Pagination
        if (data.total_pages > 1) {
            paginationEl.style.display = 'flex';
            document.getElementById('prevPageBtn').disabled = page <= 1;
            document.getElementById('nextPageBtn').disabled = page >= data.total_pages;
            document.getElementById('pageInfo').textContent = `Halaman ${page} dari ${data.total_pages}`;
        } else {
            paginationEl.style.display = 'none';
        }
    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

function renderHistoryItem(item) {
    const date = new Date(item.timestamp);
    const dateStr = date.toLocaleDateString('id-ID', { day: 'numeric', month: 'short', year: 'numeric' });
    const timeStr = date.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
    const isFatigued = item.prediction === 'Fatigued';
    const statusClass = isFatigued ? 'fatigued' : 'non-fatigued';
    const statusText = isFatigued ? 'Kelelahan' : 'Tidak Kelelahan';
    const statusIcon = isFatigued ? '⚠️' : '✅';
    
    return `
        <div class="history-item ${statusClass}">
            <div class="history-item-face">
                ${item.face_image ? `<img src="data:image/jpeg;base64,${item.face_image}" alt="Face">` : '<div class="no-face">No Image</div>'}
            </div>
            <div class="history-item-info">
                <div class="history-item-status">
                    <span class="history-status-badge ${statusClass}">
                        ${statusIcon} ${statusText}
                    </span>
                    <span class="history-confidence">${item.confidence}%</span>
                </div>
                <div class="history-item-date">${dateStr} • ${timeStr}</div>
                ${item.explanation ? `<div class="history-item-explanation">${item.explanation.substring(0, 120)}...</div>` : ''}
            </div>
            <button class="history-delete-btn" onclick="deleteHistoryItem(${item.id})" title="Hapus">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 4L12 12M4 12L12 4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
            </button>
        </div>
    `;
}

async function deleteHistoryItem(id) {
    if (!confirm('Hapus riwayat deteksi ini?')) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/history/${id}`, {
            method: 'DELETE',
            headers: { 'Authorization': 'Bearer ' + authToken }
        });
        const data = await response.json();
        
        if (data.success) {
            showSuccess('Riwayat berhasil dihapus');
            loadHistory(currentHistoryPage);
        }
    } catch (err) {
        showError('Gagal menghapus riwayat');
    }
}

// Make functions global
window.showAuthModal = showAuthModal;
window.hideAuthModal = hideAuthModal;
window.switchAuthTab = switchAuthTab;
window.handleLogin = handleLogin;
window.handleRegister = handleRegister;
window.handleLogout = handleLogout;
window.deleteHistoryItem = deleteHistoryItem;
window.loadHistory = loadHistory;
