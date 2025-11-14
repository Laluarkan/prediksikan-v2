document.addEventListener('DOMContentLoaded', async () => {
    // --- Inisialisasi DOM Elements ---
    const leagueSelect = document.getElementById('leagueSelect');
    const csvFileInput = document.getElementById('csvFile');
    const dataTableBody = document.querySelector('#dataTable tbody');
    const saveBtn = document.getElementById('saveBtn');
    const toggleBtn = document.getElementById('toggleThemeBtn'); 
    const body = document.body;
    const csrfTokenInput = document.querySelector('input[name="csrfmiddlewaretoken"]');
    const csrfToken = csrfTokenInput ? csrfTokenInput.value : null;

    let selectedLeague = '';
    let newMatches = []; 

    // --- Fungsi Helper ---
    function autoScrollTable() {
        const tableContainer = document.querySelector('.table-container');
        if (tableContainer) {
            tableContainer.scrollTop = tableContainer.scrollHeight;
        }
    }
    
    const isDarkMode = localStorage.getItem('theme') === 'dark' || (!localStorage.getItem('theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
    if (isDarkMode) {
        body.classList.add('dark-mode');
        if (toggleBtn) toggleBtn.textContent = '☀️ Mode Terang';
    } else {
        body.classList.remove('dark-mode'); // Hapus jika tidak dark (default dari base mungkin sudah terang)
        if (toggleBtn) toggleBtn.textContent = '🌙 Mode Gelap';
    }

    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            body.classList.toggle('dark-mode');
            if (body.classList.contains('dark-mode')) {
                localStorage.setItem('theme', 'dark');
                toggleBtn.textContent = '☀️ Mode Terang';
            } else {
                localStorage.setItem('theme', 'light');
                toggleBtn.textContent = '🌙 Mode Gelap';
            }
        });
    }

    leagueSelect.addEventListener('change', () => {
        selectedLeague = leagueSelect.value;
        dataTableBody.innerHTML = '<tr><td colspan="29" class="text-center">Pilih liga dan upload file CSV untuk melihat pratinjau data baru.</td></tr>'; // Reset table
        newMatches = []; // Reset data
        csvFileInput.value = ''; // Reset file input
    });

    // 2. Proses CSV (Mengirim file langsung ke backend)
    csvFileInput.addEventListener('change', async (e) => {
        if (!selectedLeague) {
            alert('Pilih liga terlebih dahulu!');
            csvFileInput.value = ''; 
            return;
        }
        
        if (!csrfToken) {
            alert('Kesalahan: Token keamanan tidak ditemukan. Coba refresh halaman.');
            csvFileInput.value = '';
            return;
        }

        const file = e.target.files[0];
        if (!file) return;

        // Tampilkan indikator loading
        saveBtn.textContent = 'Memproses Data...';
        saveBtn.disabled = true;
        dataTableBody.innerHTML = '<tr><td colspan="29" class="text-center">Memproses dan menghitung fitur otomatis di server...</td></tr>';
        
        // Buat FormData untuk mengirim file
        const formData = new FormData();
        formData.append('league', selectedLeague);
        formData.append('file', file);
        
        try {
            // Panggil API upload_csv
            const res = await fetch('/api/upload_csv', {
                method: 'POST',
                headers: {
                  'X-CSRFToken': csrfToken // Tambahkan token CSRF ke header
                },
                body: formData // Kirim file
            });

            const data = await res.json();
            
            // Bersihkan tampilan loading
            dataTableBody.innerHTML = '';
            newMatches = [];

            if (res.ok && data.status === 'ok') { // Periksa res.ok juga
                newMatches = data.matches || []; 
                
                if (newMatches.length === 0) {
                    alert(data.message || 'Tidak ada pertandingan baru yang ditemukan.'); 
                    dataTableBody.innerHTML = '<tr><td colspan="29" class="text-center">Tidak ada pertandingan baru yang ditemukan.</td></tr>';
                } else {
                    // Tampilkan di tabel
                    const columns = [
                        'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                        'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5',
                        'HomeTeamElo','AwayTeamElo','EloDifference',
                        'Home_AvgGoalsScored','Home_AvgGoalsConceded','Home_Wins','Home_Draws','Home_Losses',
                        'Away_AvgGoalsScored','Away_AvgGoalsConceded','Away_Wins','Away_Draws','Away_Losses',
                        'HTH_HomeWins','HTH_AwayWins','HTH_Draws','HTH_AvgHomeGoals','HTH_AvgAwayGoals'
                    ];
                    
                    const oddColumns = ['AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5'];
                    const scoreColumns = ['FTHG', 'FTAG', 'Home_Wins', 'Home_Draws', 'Home_Losses', 'Away_Wins', 'Away_Draws', 'Away_Losses', 'HTH_HomeWins', 'HTH_AwayWins', 'HTH_Draws'];

                    newMatches.forEach(match => {
                        const tr = document.createElement('tr');
                        columns.forEach(col => {
                            const td = document.createElement('td');
                            const val = match[col] !== undefined && match[col] !== null ? match[col] : ''; // Handle null
                            
                            let displayValue = val;
                            
                            // Format Tanggal (backend mengirim YYYY-MM-DD HH:MM:SS)
                            if (col === 'Date' && typeof val === 'string' && val) {
                                try {
                                    const dateObj = new Date(val.replace(' ', 'T') + 'Z'); // Asumsikan UTC dari server
                                    if (!isNaN(dateObj.getTime())) {
                                        // Tampilkan sebagai DD-MM-YYYY
                                        displayValue = dateObj.toLocaleDateString('en-GB').replace(/\//g, '-');
                                    } 
                                } catch (e) { /* Biarkan nilai asli jika error */ }
                            } 
                            // Format Angka (dari logika lama Anda)
                            else if (val !== '' && !isNaN(Number(val))) { // Cek apakah bisa jadi angka
                                const num = Number(val);
                                if (scoreColumns.includes(col)) {
                                    displayValue = Math.round(num).toString();
                                } else if (oddColumns.includes(col)) {
                                    displayValue = num.toFixed(2);
                                } else { // Elo, AvgGoals, dll.
                                    displayValue = num % 1 === 0 ? num.toString() : num.toFixed(3); // 3 desimal jika bukan integer
                                }
                            }
                            
                            td.textContent = displayValue;
                            tr.appendChild(td);
                        });
                        dataTableBody.appendChild(tr);
                    });
                    
                    alert(`Ditemukan ${newMatches.length} pertandingan baru siap disimpan.`);
                    autoScrollTable();
                }

            } else { // Jika res.ok false atau data.status bukan 'ok'
                alert(`Gagal memproses file: ${data.message || `Error ${res.status}`}`);
                dataTableBody.innerHTML = `<tr><td colspan="29" class="text-center" style="color:red;">Gagal memproses file: ${data.message || `Error ${res.status}`}</td></tr>`;
            }

        } catch (error) {
            console.error('Fetch error:', error);
            alert('Terjadi kesalahan koneksi atau server saat upload.');
            dataTableBody.innerHTML = '<tr><td colspan="29" class="text-center" style="color:red;">Terjadi kesalahan koneksi.</td></tr>';
        } finally {
            saveBtn.textContent = '💾 Simpan Data';
            saveBtn.disabled = (newMatches.length === 0); // Disable jika tidak ada data
        }
    });

    // 3. Simpan data baru ke dataset
    saveBtn.addEventListener('click', async () => {
        if (!selectedLeague || newMatches.length === 0) {
            alert('Tidak ada data baru untuk disimpan!');
            return;
        }
        
        if (!csrfToken) {
            alert('Kesalahan: Token keamanan tidak ditemukan. Coba refresh halaman.');
            return;
        }

        // Tampilkan loading
        saveBtn.textContent = 'Menyimpan...';
        saveBtn.disabled = true;

        try {
            const res = await fetch('/api/save_new_matches', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken // Sertakan token CSRF
                 },
                // Kirim hanya liga dan data pertandingan
                body: JSON.stringify({ league: selectedLeague, matches: newMatches }) 
            });

            const data = await res.json();
            if (res.ok && data.status === 'ok') { // Periksa res.ok juga
                alert('Data baru berhasil disimpan!');
                dataTableBody.innerHTML = '<tr><td colspan="29" class="text-center">Data berhasil disimpan. Pilih liga dan upload file lagi.</td></tr>'; // Reset table
                newMatches = []; // Reset data
                csvFileInput.value = ''; // Reset input file
            } else {
                alert(`Gagal menyimpan: ${data.message || `Error ${res.status}`}`);
            }
        } catch (error) {
            console.error('Save error:', error);
            alert('Terjadi kesalahan koneksi saat menyimpan data.');
        } finally {
            saveBtn.textContent = '💾 Simpan Data';
            // Biarkan tombol disable jika tidak ada data baru lagi
            saveBtn.disabled = (newMatches.length === 0); 
        }
    });

    // Inisialisasi: nonaktifkan tombol simpan di awal
    saveBtn.disabled = true;

}); 