// ========================================
// Configuration
// ========================================
const API_BASE_URL = '/api';

// ========================================
// State
// ========================================
let detectionCount = 0;
let authToken = localStorage.getItem('authToken') || null;
let currentUser = null;
let currentHistoryPage = 1;
let cameraStream = null;
let useFrontCamera = true;

// ========================================
// DOM Elements
// ========================================
const elements = {
    // Camera elements
    cameraVideo: document.getElementById('cameraVideo'),
    captureCanvas: document.getElementById('captureCanvas'),
    cameraContainer: document.getElementById('cameraContainer'),
    cameraOverlay: document.getElementById('cameraOverlay'),
    cameraLoading: document.getElementById('cameraLoading'),
    cameraError: document.getElementById('cameraError'),
    cameraErrorText: document.getElementById('cameraErrorText'),
    captureBtn: document.getElementById('captureBtn'),
    switchCameraBtn: document.getElementById('switchCameraBtn'),
    // Result modal elements
    resultModal: document.getElementById('resultModal'),
    resultModalClose: document.getElementById('resultModalClose'),
    resultLoading: document.getElementById('resultLoading'),
    resultModalContent: document.getElementById('resultModalContent'),
    resultStatusBanner: document.getElementById('resultStatusBanner'),
    statusIcon: document.getElementById('statusIcon'),
    statusValue: document.getElementById('statusValue'),
    confidenceValue: document.getElementById('confidenceValue'),
    confidenceFill: document.getElementById('confidenceFill'),
    faceImage: document.getElementById('faceImage'),
    recommendationMessage: document.getElementById('recommendationMessage'),
    recommendationActions: document.getElementById('recommendationActions'),
    detectionTime: document.getElementById('detectionTime'),
    retakeBtn: document.getElementById('retakeBtn'),
    retryBtn: document.getElementById('retryBtn'),
    resultError: document.getElementById('resultError'),
    resultErrorMessage: document.getElementById('resultErrorMessage'),
    // Expression & Fatigue Level elements
    fatigueLevelBadge: document.getElementById('fatigueLevelBadge'),
    fatigueLevelIcon: document.getElementById('fatigueLevelIcon'),
    fatigueLevelText: document.getElementById('fatigueLevelText'),
    expressionList: document.getElementById('expressionList'),
    // Stats
    accuracyStat: document.getElementById('accuracy-stat'),
    detectionStat: document.getElementById('detection-stat'),
    toastContainer: document.getElementById('toastContainer'),
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
    startCamera();
});

function initializeEventListeners() {
    // Capture button
    elements.captureBtn.addEventListener('click', captureAndAnalyze);

    // Switch camera button
    elements.switchCameraBtn.addEventListener('click', switchCamera);

    // Result modal close
    elements.resultModalClose.addEventListener('click', closeResultModal);

    // Retake button
    elements.retakeBtn.addEventListener('click', closeResultModal);

    // Retry button (on error state)
    if (elements.retryBtn) {
        elements.retryBtn.addEventListener('click', closeResultModal);
    }

    // Close modal on overlay click
    elements.resultModal.addEventListener('click', (e) => {
        if (e.target === elements.resultModal) {
            closeResultModal();
        }
    });

    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && elements.resultModal.style.display === 'flex') {
            closeResultModal();
        }
    });
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
// Camera Functions
// ========================================
async function startCamera() {
    elements.cameraLoading.style.display = 'flex';
    elements.cameraError.style.display = 'none';
    elements.cameraOverlay.style.display = 'none';
    elements.captureBtn.disabled = true;

    // Stop any existing stream
    stopCamera();

    try {
        const constraints = {
            video: {
                facingMode: useFrontCamera ? 'user' : 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };

        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        elements.cameraVideo.srcObject = cameraStream;

        // Wait for video to be ready
        elements.cameraVideo.onloadedmetadata = () => {
            elements.cameraVideo.play();
            elements.cameraLoading.style.display = 'none';
            elements.cameraOverlay.style.display = 'flex';
            elements.captureBtn.disabled = false;

            // Check if multiple cameras available
            checkMultipleCameras();
        };
    } catch (error) {
        console.error('Camera access failed:', error);
        elements.cameraLoading.style.display = 'none';
        elements.cameraError.style.display = 'flex';

        if (error.name === 'NotAllowedError') {
            elements.cameraErrorText.textContent = 'Akses kamera ditolak. Izinkan akses kamera di pengaturan browser Anda.';
        } else if (error.name === 'NotFoundError') {
            elements.cameraErrorText.textContent = 'Kamera tidak ditemukan. Pastikan perangkat Anda memiliki kamera.';
        } else {
            elements.cameraErrorText.textContent = 'Tidak dapat mengakses kamera: ' + error.message;
        }
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
}

async function checkMultipleCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');
        if (videoDevices.length > 1) {
            elements.switchCameraBtn.style.display = 'flex';
        }
    } catch (e) {
        // Ignore
    }
}

async function switchCamera() {
    useFrontCamera = !useFrontCamera;
    await startCamera();
}

// ========================================
// Capture & Analyze
// ========================================
async function captureAndAnalyze() {
    if (!cameraStream) {
        showError('Kamera belum aktif');
        return;
    }

    const video = elements.cameraVideo;
    const canvas = elements.captureCanvas;

    // Set canvas size to video dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    const ctx = canvas.getContext('2d');

    // If front camera, mirror the image
    if (useFrontCamera) {
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Reset transform
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // Flash effect
    elements.cameraContainer.classList.add('flash');
    setTimeout(() => elements.cameraContainer.classList.remove('flash'), 300);

    // Show result modal with loading
    showResultModal();

    // Convert canvas to blob
    canvas.toBlob(async (blob) => {
        if (!blob) {
            showError('Gagal mengambil foto');
            closeResultModal();
            return;
        }

        try {
            const formData = new FormData();
            formData.append('image', blob, 'capture.jpg');

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
                // Refresh history
                if (authToken) loadHistory();
            } else {
                // Show error IN the modal instead of closing it
                showResultError(data.error || 'Gagal melakukan deteksi');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            showResultError(error.message || 'Terjadi kesalahan saat menganalisis gambar');
        }
    }, 'image/jpeg', 0.92);
}

// ========================================
// Result Modal
// ========================================
function showResultModal() {
    elements.resultModal.style.display = 'flex';
    elements.resultLoading.style.display = 'flex';
    elements.resultModalContent.style.display = 'none';
    elements.resultError.style.display = 'none';
    document.body.style.overflow = 'hidden';

    // Animate in
    requestAnimationFrame(() => {
        elements.resultModal.classList.add('show');
    });
}

function closeResultModal() {
    elements.resultModal.classList.remove('show');
    setTimeout(() => {
        elements.resultModal.style.display = 'none';
        document.body.style.overflow = '';
        // Reset for next use
        elements.resultLoading.style.display = 'flex';
        elements.resultModalContent.style.display = 'none';
        elements.resultError.style.display = 'none';
    }, 300);
}

function showResultError(message) {
    elements.resultLoading.style.display = 'none';
    elements.resultModalContent.style.display = 'none';
    elements.resultError.style.display = 'flex';
    if (elements.resultErrorMessage) {
        elements.resultErrorMessage.textContent = message || 'Wajah tidak terdeteksi. Pastikan wajah Anda terlihat jelas di kamera.';
    }
}

// ========================================
// Display Results (in popup modal)
// ========================================
function displayResults(data) {
    // Switch from loading to content
    elements.resultLoading.style.display = 'none';
    elements.resultModalContent.style.display = 'flex';

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

    // Fatigue Level Badge
    if (data.fatigue_level && elements.fatigueLevelBadge) {
        const level = data.fatigue_level;
        const levelIcons = { normal: '✅', ringan: '🟡', sedang: '🟠', berat: '🔴' };
        elements.fatigueLevelBadge.className = `fatigue-level-badge level-${level.level}`;
        elements.fatigueLevelIcon.textContent = levelIcons[level.level] || '❓';
        elements.fatigueLevelText.textContent = level.label;
    }

    // Expression Analysis
    if (data.expressions && elements.expressionList) {
        const iconMap = {
            eye: '👁️',
            mouth: '👄',
            face: '🙂',
            result: '📊'
        };
        const statusIcons = {
            good: '✅',
            warning: '⚠️',
            danger: '🔴',
            neutral: '⬜'
        };
        elements.expressionList.innerHTML = data.expressions.map(expr => `
            <div class="expression-item status-${expr.status}">
                <span class="expression-icon">${iconMap[expr.icon] || '🔍'}</span>
                <span class="expression-label">${expr.label}</span>
                <span class="expression-value">${expr.value}</span>
                <span class="expression-status">${statusIcons[expr.status] || ''}</span>
            </div>
        `).join('');
    }

    // Recommendation (from API data)
    if (data.recommendation) {
        elements.recommendationMessage.textContent = data.recommendation.message;
        elements.recommendationActions.innerHTML = data.recommendation.actions
            .map(action => `<li>${action}</li>`).join('');
    }

    // Timestamp
    const timestamp = new Date(data.timestamp);
    elements.detectionTime.textContent = timestamp.toLocaleString('id-ID');
}

function updateDetectionCount() {
    elements.detectionStat.textContent = detectionCount;
}

// ========================================
// API Health Check
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
window.startCamera = startCamera;

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
