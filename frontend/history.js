// ========================================
// History Page Script
// ========================================
const API_BASE_URL = '/api';

let authToken = localStorage.getItem('authToken');
let currentUser = null;
let currentHistoryPage = 1;
let pendingDeleteId = null;
let historyDataCache = {};

// ========================================
// Initialization
// ========================================
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();

    // Close detail modal on overlay click
    document.getElementById('historyDetailModal').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) closeDetailModal();
    });

    // Close delete confirm modal on overlay click
    document.getElementById('deleteConfirmModal').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) cancelDelete();
    });

    // ESC key handler
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeDetailModal();
            cancelDelete();
        }
    });
});

// ========================================
// Toast Notification System
// ========================================
function showToast(message, type = 'info', duration = 5000) {
    const container = document.getElementById('toastContainer');

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
// Auth Functions
// ========================================

function getAuthHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (authToken) headers['Authorization'] = 'Bearer ' + authToken;
    return headers;
}

async function checkAuth() {
    if (!authToken) {
        window.location.href = '/';
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
            window.location.href = '/';
        }
    } catch (err) {
        window.location.href = '/';
    }
}

function updateAuthUI(isLoggedIn) {
    const userMenu = document.getElementById('userMenu');
    const userName = document.getElementById('userName');

    if (isLoggedIn && currentUser) {
        userMenu.style.display = 'flex';
        userName.textContent = currentUser.username;
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
    window.location.href = '/';
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
        const countEl = document.getElementById('historyCount');
        const paginationEl = document.getElementById('historyPagination');

        countEl.textContent = `${data.total} hasil deteksi`;

        // Cache history data for detail view
        historyDataCache = {};
        data.history.forEach(item => {
            historyDataCache[item.id] = item;
        });

        if (data.history.length === 0) {
            listEl.innerHTML = `
                <div class="history-empty">
                    <svg width="48" height="48" viewBox="0 0 64 64" fill="none">
                        <circle cx="32" cy="32" r="28" stroke="currentColor" stroke-width="2" opacity="0.1" />
                        <path d="M22 32H42M32 22V42" stroke="currentColor" stroke-width="2" stroke-linecap="round" opacity="0.2" />
                    </svg>
                    <h3>Belum ada riwayat</h3>
                    <p>Lakukan deteksi untuk melihat riwayat di sini</p>
                </div>
            `;
            paginationEl.style.display = 'none';
            return;
        }

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
        <div class="history-item ${statusClass}" id="history-item-${item.id}" onclick="showHistoryDetail(${item.id})">
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
            <button class="history-delete-btn" onclick="event.stopPropagation(); showDeleteConfirm(${item.id})" title="Hapus">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 4L12 12M4 12L12 4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
            </button>
        </div>
    `;
}

// ========================================
// Detail Analysis Modal
// ========================================

function showHistoryDetail(id) {
    const item = historyDataCache[id];
    if (!item) return;

    const modal = document.getElementById('historyDetailModal');
    const isFatigued = item.prediction === 'Fatigued';

    // Status Banner
    const banner = document.getElementById('detailStatusBanner');
    banner.className = `result-status-banner ${isFatigued ? 'fatigued' : 'non-fatigued'}`;
    document.getElementById('detailStatusIcon').textContent = isFatigued ? '⚠️' : '✅';
    document.getElementById('detailStatusValue').textContent = isFatigued ? 'Terdeteksi Kelelahan' : 'Tidak Kelelahan';

    // Confidence
    const confidence = item.confidence;
    document.getElementById('detailConfidenceValue').textContent = confidence + '%';
    const fill = document.getElementById('detailConfidenceFill');
    fill.style.width = '0%';
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            fill.style.width = confidence + '%';
        });
    });

    // Fatigue Level Badge
    const fatigueBadge = document.getElementById('detailFatigueBadge');
    const fatigueIcon = document.getElementById('detailFatigueIcon');
    const fatigueText = document.getElementById('detailFatigueText');

    if (isFatigued) {
        if (confidence >= 80) {
            fatigueBadge.className = 'fatigue-level-badge level-berat';
            fatigueIcon.textContent = '🔴';
            fatigueText.textContent = 'Kelelahan Berat';
        } else if (confidence >= 65) {
            fatigueBadge.className = 'fatigue-level-badge level-sedang';
            fatigueIcon.textContent = '🟠';
            fatigueText.textContent = 'Kelelahan Sedang';
        } else {
            fatigueBadge.className = 'fatigue-level-badge level-ringan';
            fatigueIcon.textContent = '🟡';
            fatigueText.textContent = 'Kelelahan Ringan';
        }
    } else {
        fatigueBadge.className = 'fatigue-level-badge level-normal';
        fatigueIcon.textContent = '✅';
        fatigueText.textContent = 'Tidak Lelah';
    }

    // Face Image
    if (item.face_image) {
        document.getElementById('detailFaceImage').src = 'data:image/jpeg;base64,' + item.face_image;
    }

    // GradCAM Heatmap
    const gradcamSection = document.getElementById('detailGradcamSection');
    if (item.gradcam_image) {
        gradcamSection.style.display = 'block';
        document.getElementById('detailGradcamImage').src = 'data:image/jpeg;base64,' + item.gradcam_image;
    } else {
        gradcamSection.style.display = 'none';
    }

    // Face Region Analysis (actually stores expressions data)
    const regionCard = document.getElementById('detailRegionCard');
    const regionList = document.getElementById('detailRegionList');
    if (item.face_regions && item.face_regions.length > 0) {
        regionCard.style.display = 'block';

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

        regionList.innerHTML = item.face_regions.map(expr => `
            <div class="expression-item status-${expr.status}">
                <span class="expression-icon">${iconMap[expr.icon] || '🔍'}</span>
                <span class="expression-label">${expr.label}</span>
                <span class="expression-value">${expr.value}</span>
                <span class="expression-status">${statusIcons[expr.status] || ''}</span>
            </div>
        `).join('');
    } else {
        regionCard.style.display = 'none';
    }

    // AI Explanation
    const explanationSection = document.getElementById('detailExplanationSection');
    if (item.explanation) {
        explanationSection.style.display = 'block';
        document.getElementById('detailExplanation').textContent = item.explanation;
    } else {
        explanationSection.style.display = 'none';
    }

    // Timestamp
    const timestamp = new Date(item.timestamp);
    document.getElementById('detailTime').textContent = timestamp.toLocaleString('id-ID');

    // Show modal with animation
    modal.style.display = 'flex';
    requestAnimationFrame(() => {
        modal.classList.add('show');
    });
    document.body.style.overflow = 'hidden';
}

function closeDetailModal() {
    const modal = document.getElementById('historyDetailModal');
    modal.classList.remove('show');
    setTimeout(() => {
        modal.style.display = 'none';
        document.body.style.overflow = '';
    }, 300);
}

// ========================================
// Delete Confirmation Dialog
// ========================================

function showDeleteConfirm(id) {
    pendingDeleteId = id;
    const modal = document.getElementById('deleteConfirmModal');
    modal.style.display = 'flex';
    requestAnimationFrame(() => {
        modal.classList.add('show');
    });
}

function cancelDelete() {
    pendingDeleteId = null;
    const modal = document.getElementById('deleteConfirmModal');
    modal.classList.remove('show');
    setTimeout(() => {
        modal.style.display = 'none';
    }, 300);
}

async function confirmDelete() {
    if (!pendingDeleteId) return;

    const id = pendingDeleteId;

    // Close confirm modal
    cancelDelete();

    // Animate item removal
    const itemEl = document.getElementById(`history-item-${id}`);
    if (itemEl) {
        itemEl.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        itemEl.style.opacity = '0';
        itemEl.style.transform = 'translateX(20px)';
    }

    try {
        const response = await fetch(`${API_BASE_URL}/history/${id}`, {
            method: 'DELETE',
            headers: { 'Authorization': 'Bearer ' + authToken }
        });
        const data = await response.json();

        if (data.success) {
            showSuccess('Riwayat berhasil dihapus');
            // Remove from cache
            delete historyDataCache[id];
            // Wait for animation then reload
            setTimeout(() => {
                loadHistory(currentHistoryPage);
            }, 300);
        } else {
            // Revert animation
            if (itemEl) {
                itemEl.style.opacity = '1';
                itemEl.style.transform = 'translateX(0)';
            }
            showError('Gagal menghapus riwayat');
        }
    } catch (err) {
        if (itemEl) {
            itemEl.style.opacity = '1';
            itemEl.style.transform = 'translateX(0)';
        }
        showError('Gagal menghapus riwayat');
    }
}

// Make functions global
window.handleLogout = handleLogout;
window.deleteHistoryItem = showDeleteConfirm;
window.showDeleteConfirm = showDeleteConfirm;
window.cancelDelete = cancelDelete;
window.confirmDelete = confirmDelete;
window.showHistoryDetail = showHistoryDetail;
window.closeDetailModal = closeDetailModal;
window.loadHistory = loadHistory;
