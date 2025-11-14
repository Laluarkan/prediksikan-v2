// static/js/home.js

document.addEventListener('DOMContentLoaded', () => {
    
    // Fungsi helper untuk mendapatkan token CSRF dari cookie
    function getCsrfToken() {
        // Coba ambil dari tag <input> (jika masih ada)
        const csrfEl = document.querySelector('input[name="csrfmiddlewaretoken"]');
        if (csrfEl) {
            return csrfEl.value;
        }
        // Jika tidak ada, ambil dari cookie (cara yang lebih kuat)
        if (document.cookie) {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.startsWith('csrftoken=')) {
                    return cookie.substring('csrftoken='.length, cookie.length);
                }
            }
        }
        return null;
    }

    const csrfToken = getCsrfToken();
    const historyList = document.getElementById('history-list');
    const clearBtn = document.getElementById('clear-history');
    

    async function renderHistory() {
        if (!historyList) return;
        
        historyList.innerHTML = '<p>(Memuat riwayat...)</p>'; // Loading
        try { 
            const res = await fetch('/api/history'); 
            const data = await res.json();
            
            if (!res.ok || data.status !== 'ok') { 
                historyList.innerHTML = '<p style="color:red;">Gagal memuat riwayat.</p>';
                return;
            }
            
            // Ambil maksimal 3 item pertama dari data API
            const h = data.history.slice(0, 3); // <<--- BATASI 3 ITEM
            
            if (!h.length) {
                historyList.textContent = '(Belum ada riwayat)';
                return;
            }
            historyList.innerHTML = ''; 
            
            h.forEach(item => { 
                const el = document.createElement('a'); 
                el.className = 'hist-item'; 
                el.href = `/history/${item.id}/`;
                
                // Format Waktu ke Bahasa Indonesia
                const timestamp = item.timestamp 
                    ? new Date(item.timestamp).toLocaleString('id-ID', { 
                        day: '2-digit', month: '2-digit', year:'numeric', 
                        hour: '2-digit', minute: '2-digit' 
                    }) 
                    : '(Waktu tidak tersimpan)';

                el.innerHTML = `
                    <strong>${item.league}</strong>
                    <span class="hist-match">${item.home_team || 'Tim Home'} vs ${item.away_team || 'Tim Away'}</span>
                    <small>${timestamp}</small>
                    <span class="hist-preds">
                        HDA: ${item.prediction.HDA?.label || '-'} | 
                        OU: ${item.prediction.OU25?.label || '-'} | 
                        BTTS: ${item.prediction.BTTS?.label || '-'}
                    </span>
                `;
                historyList.appendChild(el);
            });
        } catch (error) {
             console.error("Fetch history error:", error);
             historyList.innerHTML = '<p style="color:red;">Error koneksi saat memuat riwayat.</p>';
        }
    }

    // Event listener clearBtn tidak berubah
    if (clearBtn) {
        clearBtn.addEventListener('click', async () => {
            if (!csrfToken) {
                 alert('Gagal menghapus riwayat: Token keamanan tidak ditemukan.');
                 return;
            }
            if (confirm("Anda yakin ingin menghapus semua riwayat prediksi?")) { // Tambah konfirmasi
                try {
                     await fetch('/api/clear_history', { 
                         method: 'POST',
                         headers: { 'X-CSRFToken': csrfToken }
                     });
                     renderHistory(); // Muat ulang riwayat (akan kosong)
                 } catch (error) {
                     alert('Gagal menghapus riwayat karena error koneksi.');
                 }
            }
        });
    }

    if (historyList) {
        renderHistory();
    }
});