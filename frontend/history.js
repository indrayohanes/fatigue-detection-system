// ========================================
// History Page Script
// ========================================
const API_BASE_URL = '/api';

let authToken = localStorage.getItem('authToken');
let currentUser = null;
let currentHistoryPage = 1;

// ========================================
// Initialization
// ========================================
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
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
        // Not logged in, redirect to main page
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
            // Token invalid, redirect
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
        <div class="history-item ${statusClass}" id="history-item-${item.id}">
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

    // Instantly remove item from DOM with animation
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
            // Wait for animation then reload list
            setTimeout(() => {
                loadHistory(currentHistoryPage);
            }, 300);
        } else {
            // Revert animation if failed
            if (itemEl) {
                itemEl.style.opacity = '1';
                itemEl.style.transform = 'translateX(0)';
            }
            showError('Gagal menghapus riwayat');
        }
    } catch (err) {
        // Revert animation if failed
        if (itemEl) {
            itemEl.style.opacity = '1';
            itemEl.style.transform = 'translateX(0)';
        }
        showError('Gagal menghapus riwayat');
    }
}

// Make functions global
window.handleLogout = handleLogout;
window.deleteHistoryItem = deleteHistoryItem;
window.loadHistory = loadHistory;
