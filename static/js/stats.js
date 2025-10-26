document.addEventListener('DOMContentLoaded', () => {
    // Pastikan API_URLS tersedia dari script di template
    if (typeof API_URLS === 'undefined') {
        console.error("API_URLS belum didefinisikan. Pastikan script di template termuat.");
        return;
    }

    // === Ambil elemen ===
    const leagueSel = document.getElementById('league-select');
    const teamSel = document.getElementById('team-select');
    const loadBtn = document.getElementById('load-stats');
    const container = document.getElementById('team-stats');

    // === 1. Ambil daftar liga ===
    async function loadLeagues() {
        try {
            const res = await fetch(API_URLS.leagues);
            const data = await res.json();
            
            leagueSel.innerHTML = '<option value="">-- Pilih Liga --</option>';

            if (data.status === 'ok') {
                data.leagues.forEach(l => {
                    leagueSel.innerHTML += `<option value="${l}">${l}</option>`;
                });
            } else {
                console.error('Gagal memuat liga:', data.message || 'Error tidak diketahui');
                container.innerHTML = `<p style="color:red;">Gagal memuat daftar liga.</p>`;
            }
        } catch (error) {
            console.error('Error saat fetch leagues:', error);
            container.innerHTML = `<p style="color:red;">Error koneksi saat memuat liga.</p>`;
        }
    }

    // === 2. Ambil tim saat liga dipilih ===
    leagueSel.addEventListener('change', async () => {
        const league = leagueSel.value;
        teamSel.innerHTML = '<option value="">-- Pilih Tim --</option>'; // Reset tim
        container.innerHTML = 'Pilih liga dan tim untuk melihat statistik.'; // Reset kontainer

        if (!league) return;

        try {
            const res = await fetch(`${API_URLS.teams}?league=${encodeURIComponent(league)}`);
            const data = await res.json();
            
            if (data.status === 'ok') {
                data.teams.forEach(t => {
                    teamSel.innerHTML += `<option value="${t}">${t}</option>`;
                });
            } else {
                console.error('Gagal memuat tim:', data.message || 'Error tidak diketahui');
                teamSel.innerHTML = `<option value="">-- Gagal Memuat --</option>`;
            }
        } catch (error) {
            console.error('Error saat fetch teams:', error);
            teamSel.innerHTML = `<option value="">-- Error Koneksi --</option>`;
        }
    });

    // === 3. Tampilkan statistik tim ===
    loadBtn.addEventListener('click', async () => {
        const league = leagueSel.value;
        const team = teamSel.value;
        
        if (!league || !team) {
            alert('Pilih liga dan tim terlebih dahulu!');
            return;
        }

        container.innerHTML = '<p>Sedang memuat data...</p>';

        try {
            const res = await fetch(API_URLS.teamStats, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    // Penting: Tambahkan CSRF token untuk POST di Django
                    'X-CSRFToken': getCookie('csrftoken') 
                },
                body: JSON.stringify({ league, team })
            });
            const data = await res.json();

            if (data.status === 'ok') {
                const s = data.stats;
                const recent = s.recent || {};

                container.innerHTML = `
                    <div class="stats-card">
                        <div class="stat-item"><strong>Last Elo</strong><div>${s.last_elo || '-'}</div></div>
                        <div class="stat-item"><strong>Avg Goals Scored</strong><div>${recent.AvgGoalsScored || 0}</div></div>
                        <div class="stat-item"><strong>Avg Goals Conceded</strong><div>${recent.AvgGoalsConceded || 0}</div></div>
                        <div class="stat-item"><strong>Wins</strong><div>${recent.Wins || 0}</div></div>
                        <div class="stat-item"><strong>Draws</strong><div>${recent.Draws || 0}</div></div>
                        <div class="stat-item"><strong>Losses</strong><div>${recent.Losses || 0}</div></div>
                    </div>
                `;
            } else {
                container.innerHTML = `<p style="color:red;">Gagal memuat statistik: ${data.message || 'Error tidak diketahui'}</p>`;
            }
        } catch (error) {
            console.error('Error saat fetch team_stats:', error);
            container.innerHTML = `<p style="color:red;">Error koneksi saat memuat statistik.</p>`;
        }
    });

    // Helper function untuk mengambil CSRF token dari cookie (standar Django)
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                let cookie = cookies[i].trim();
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    loadLeagues();
});