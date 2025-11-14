window.handleChoiceClick = async function(historyId, choiceType, choiceValue) {
    // Ambil token CSRF di sini karena fungsi ini global
    const csrfToken = document.querySelector('input[name="csrfmiddlewaretoken"]').value;

    if (!historyId) {
        alert('Fitur ini memerlukan login. Silakan login untuk menyimpan pilihan Anda.');
        return;
    }
    
    try {
        const res = await fetch('/api/save_choice', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify({ 
                id: historyId, 
                type: choiceType, 
                value: choiceValue 
            })
        });

        const data = await res.json();
        if (data.status === 'ok') {
            alert(`Pilihan ${choiceType}: ${choiceValue} berhasil disimpan sebagai pilihan terbaik!`);
        } else {
            alert(`Gagal menyimpan pilihan: ${data.message}`);
        }
    } catch (err) {
        alert(`Terjadi error koneksi: ${err.message}`);
    }
}

document.addEventListener('DOMContentLoaded', () => {   
    const csrfToken = document.querySelector('input[name="csrfmiddlewaretoken"]').value;
    const leagueEl = document.getElementById('League');
    const homeEl = document.getElementById('HomeTeam');
    const awayEl = document.getElementById('AwayTeam');
    const featureSection = document.getElementById('feature-section');
    const predictBtn = document.getElementById('PredictBtn');
    const resultEl = document.getElementById('result');
    const loadingSpinner = document.getElementById('loading-spinner');
    const predictionContentEl = document.getElementById('prediction-content'); 
    const aiExplanationContainer = document.getElementById('ai-explanation-container'); 
    const aiExplanationEl = document.getElementById('ai-explanation'); 
    let currentFeatures = null;
    let lastHistoryId = null;

    function renderProbBlockWithButtons(title, probs, type) {
        if (!probs) return '';
        let html = `<div class="prob-group"><h4>${title}</h4>`;
        for (const [label, val] of Object.entries(probs)) {
             const percent = (val * 100).toFixed(1);
             const gradient = getGradientColor(percent);
             
             let buttonHTML = '';
             buttonHTML = `
                <div style="flex-shrink:0; width: 100px; margin-left: 15px;"> 
                    <button class="btn small" 
                       onclick="window.handleChoiceClick(${lastHistoryId}, '${type}', '${label}')">Pilih Ini</button>
                </div>`;
             
             html += `
             <div class="prob-item">
                 <span>${label}</span>
                 <div class="prob-bar"><div class="bar" style="width:${percent}%; background:${gradient}"></div></div>
                 <span>${percent}%</span>
                 ${buttonHTML}
             </div>`;
        }
        html += `</div>`;
        return html;
    }

    // Inisialisasi Choices.js
    const leagueChoice = new Choices(leagueEl, {
        searchEnabled: false,
        itemSelectText: 'Tekan untuk memilih',
        shouldSort: false, 
        allowHTML: false,
        appendLocation: 'origin',
    });

    const homeChoice = new Choices(homeEl, {
        searchEnabled: true,
        itemSelectText: 'Tekan untuk memilih',
        shouldSort: true, 
        allowHTML: false,
        appendLocation: 'origin',
        searchPlaceholderValue: 'Ketik untuk mencari tim...'
    });

    const awayChoice = new Choices(awayEl, {
        searchEnabled: true,
        itemSelectText: 'Tekan untuk memilih',
        shouldSort: true, 
        allowHTML: false,
        appendLocation: 'origin',
        searchPlaceholderValue: 'Ketik untuk mencari tim...'
    });
    
    // ===================== LOAD LEAGUES =====================
    async function loadLeagues() {
        try {
            const res = await fetch('/api/leagues');
            const j = await res.json();
            const choices = []; 

            if (j.status === 'ok' && Array.isArray(j.leagues)) {
                j.leagues.forEach(l => {
                    choices.push({ value: l, label: l });
                });
                leagueChoice.setChoices(choices, 'value', 'label', true);
                leagueChoice.setValue(['']); 
            } else {
                 leagueChoice.setChoices([
                     { value: '', label: '(Gagal memuat liga)' }
                 ], 'value', 'label', true);
                 leagueChoice.setValue(['']);
            }
        } catch {
            leagueChoice.setChoices([
                { value: '', label: '(Error koneksi)', disabled: true }
            ], 'value', 'label', true);
            leagueChoice.setValue(['']); 
        }
    }

    // ===================== LOAD TEAMS =====================
    async function loadTeamsForLeague(league) {
        try {
            const res = await fetch(`/api/teams?league=${encodeURIComponent(league)}`);
            const j = await res.json();
            const teamChoices = [];

            if (j.status === 'ok' && Array.isArray(j.teams)) {
                j.teams.forEach(t => {
                    teamChoices.push({ value: t, label: t });
                });
                homeChoice.setChoices(teamChoices, 'value', 'label', true);
                awayChoice.setChoices(teamChoices, 'value', 'label', true);
                homeChoice.setValue(['']);
                awayChoice.setValue(['']);
                homeChoice.enable();
                awayChoice.enable();
            } else {
                 homeChoice.disable();
                 awayChoice.disable();
                 homeChoice.clearStore();
                 awayChoice.clearStore();
                 homeChoice.setValue(['']); 
                 awayChoice.setValue(['']);
            }
        } catch {
            alert('Gagal memuat tim (catch block)');
            homeChoice.disable();
            awayChoice.disable();
            homeChoice.clearStore();
            awayChoice.clearStore();
            homeChoice.setValue(['']);
            awayChoice.setValue(['']);
        }
    }

    // ===================== FETCH FEATURES =====================
    async function fetchAndStoreFeatures(league, home, away) {
        const res = await fetch('/api/features', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify({ league, home, away })
        });

        const j = await res.json();
        if (j.status === 'ok') {
            currentFeatures = j.features;
            fillFeatureInputsWithTotals(j.features);
        } else {
            throw new Error(j.message || 'Fitur gagal dimuat');
        }
    }

    // ===================== TAMPILKAN FITUR DI INPUT =====================
    function fillFeatureInputsWithTotals(features) {
        const displayData = { ...features };
        const home_matches = features.Home_Wins + features.Home_Draws + features.Home_Losses;
        const away_matches = features.Away_Wins + features.Away_Draws + features.Away_Losses;
        const hth_matches = features.HTH_HomeWins + features.HTH_AwayWins + features.HTH_Draws;
        const home_count = home_matches > 0 ? home_matches : 5;
        const away_count = away_matches > 0 ? away_matches : 5;
        const hth_count = hth_matches > 0 ? hth_matches : 5;

        displayData.Home_AvgGoalsScored = Math.round(features.Home_AvgGoalsScored * home_count);
        displayData.Home_AvgGoalsConceded = Math.round(features.Home_AvgGoalsConceded * home_count);
        displayData.Away_AvgGoalsScored = Math.round(features.Away_AvgGoalsScored * away_count);
        displayData.Away_AvgGoalsConceded = Math.round(features.Away_AvgGoalsConceded * away_count);
        displayData.HTH_AvgHomeGoals = Math.round(features.HTH_AvgHomeGoals * hth_count);
        displayData.HTH_AvgAwayGoals = Math.round(features.HTH_AvgAwayGoals * hth_count);

        const featureIdMap = { 'Avg>2.5': 'AvgOver25', 'Avg<2.5': 'AvgUnder25' };
        for (const [key, val] of Object.entries(displayData)) {
            const elementId = featureIdMap[key] || key;
            const el = document.getElementById(elementId);
            if (el) {
                if (key === 'EloDifference' && typeof val === 'number') {
                    el.value = val.toFixed(2);
                } else {
                    el.value = val;
                }
            }
        }

        ['AvgH', 'AvgD', 'AvgA', 'AvgOver25', 'AvgUnder25'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.value = '';
        });

        featureSection.classList.remove('hidden');
    }

    // ===================== AMBIL INPUT UNTUK PREDIKSI =====================
    function getFeaturesForPrediction() {
        if (!currentFeatures) {
            throw new Error('Fitur pertandingan belum dimuat. Silakan pilih tim terlebih dahulu.');
        }
        const featuresForPrediction = { ...currentFeatures };
        const parseOdd = (id) => {
            const val = document.getElementById(id).value;
            const parsedVal = parseFloat(val);
            
            // Jika val string kosong ("") atau tidak valid (NaN)
            if (isNaN(parsedVal) || val.trim() === '') {
                return 1; // Default ke 1
            }
            // Jika valid (termasuk jika user mengetik 0), gunakan nilai tersebut
            return parsedVal; 
        };

        featuresForPrediction.AvgH = parseOdd('AvgH');
        featuresForPrediction.AvgD = parseOdd('AvgD');
        featuresForPrediction.AvgA = parseOdd('AvgA');
        featuresForPrediction['Avg>2.5'] = parseOdd('AvgOver25');
        featuresForPrediction['Avg<2.5'] = parseOdd('AvgUnder25');
        
        return featuresForPrediction;
    }

    // ===================== TAMPILKAN HASIL =====================
    function showPredictionResult(json) {
        if (predictionContentEl) predictionContentEl.innerHTML = ''; 
        if (aiExplanationContainer) aiExplanationContainer.classList.add('hidden'); 
        if (aiExplanationEl) aiExplanationEl.textContent = 'Memuat ...'; 

        if (resultEl) resultEl.classList.remove('hidden'); 

        if (!json || json.status !== 'ok' || !json.prediction) {
             if (predictionContentEl) predictionContentEl.innerHTML = `<div style="color: red; text-align: center;">❌ Gagal menampilkan hasil prediksi: ${json.message || 'Error tidak diketahui'}</div>`;
            return; 
        }

        lastHistoryId = json.history_id || null;

        const p = json.prediction;
        const mainResultHTML = `
            <div class="result-title">📊 Hasil Prediksi</div>
            <div id="main-result" style="text-align:center; margin-bottom: 20px;">
              🏠 (H/D/A): <b>${p.HDA?.label || '-'}</b><br>
              ⚽ (O/U 2.5): <b>${p.OU25?.label || '-'}</b><br>
              🤝 (BTTS): <b>${p.BTTS?.label || '-'}</b>
            </div>
            <div id="prob-bars" class="prob-section">
              ${renderProbBlockWithButtons('🏠 (Home / Draw / Away)', p.HDA?.probs, 'HDA')}
              ${renderProbBlockWithButtons('⚽ Over / Under 2.5', p.OU25?.probs, 'OU')}
              ${renderProbBlockWithButtons('🤝 Both Team To Score', p.BTTS?.probs, 'BTTS')}
            </div>
        `;
        
        if (predictionContentEl) predictionContentEl.innerHTML = mainResultHTML; 
        animateBars(); 

        // Tampilkan Penjelasan AI 
        if (json.explanation && aiExplanationEl && aiExplanationContainer) {
            aiExplanationEl.textContent = json.explanation; 
            aiExplanationContainer.classList.remove('hidden'); 
        } else if(aiExplanationEl && aiExplanationContainer) {
             aiExplanationEl.textContent = 'Penjelasan AI tidak tersedia.'; 
             aiExplanationContainer.classList.remove('hidden'); 
        }
    }
    
    function getGradientColor(percent) {
        let r, g, b = 0;
        if (percent < 50) {
            r = 255;
            g = Math.round(5.1 * percent);
        } else {
            g = 255;
            r = Math.round(510 - 5.1 * percent);
        }
        return `linear-gradient(90deg, rgb(${r},${g},0) 0%, rgb(${r},${g},0) ${percent}%)`;
    }
    
    function renderProbBlock(title, probs) {
        if (!probs) return '';
        let html = `<div class="prob-group"><h4>${title}</h4>`;
        for (const [label, val] of Object.entries(probs)) {
            const percent = (val * 100).toFixed(1);
            const gradient = getGradientColor(percent);
            html += `
            <div class="prob-item">
                <span>${label}</span>
                <div class="prob-bar"><div class="bar" style="width:${percent}%; background:${gradient}"></div></div>
                <span>${percent}%</span>
            </div>`;
        }
        html += `</div>`;
        return html;
    }

    function animateBars() {
        document.querySelectorAll('.bar').forEach(bar => {
            const percent = bar.style.width.replace('%', '');
            bar.style.width = '0%';
            setTimeout(() => {
                bar.style.transition = 'width 1s ease';
                bar.style.width = `${percent}%`;
            }, 50);
        });
    }

    // ===================== EVENT LISTENERS =====================
    leagueEl.addEventListener('change', e => {
        const league = e.target.value;
        if (!league) return;
        homeChoice.disable();
        awayChoice.disable();
        homeChoice.clearStore();
        awayChoice.clearStore();
        homeChoice.setValue(['']);
        awayChoice.setValue(['']);
        loadTeamsForLeague(league);
        featureSection.classList.add('hidden');
        resultEl.classList.add('hidden');
        currentFeatures = null;
    });

    [homeEl, awayEl].forEach(sel => {
        const choiceInstance = (sel.id === 'HomeTeam') ? homeChoice : awayChoice;
        choiceInstance.passedElement.element.addEventListener('choice', async (event) => {
            featureSection.classList.add('hidden');
            resultEl.classList.add('hidden');
            currentFeatures = null; 
            const leagueValue = leagueChoice.getValue(true);
            const homeValue = homeChoice.getValue(true);
            const awayValue = awayChoice.getValue(true);
            
            if (leagueValue && leagueValue !== '' &&
                homeValue && homeValue !== '' &&
                awayValue && awayValue !== '' &&
                homeValue !== awayValue) {
                try {
                    if (loadingSpinner) loadingSpinner.classList.remove('hidden');
                    await fetchAndStoreFeatures(leagueValue, homeValue, awayValue);
                } catch (err) {
                    console.error('Error fetching features:', err);
                    alert('Gagal memuat fitur otomatis: ' + err.message);
                    featureSection.classList.add('hidden');
                } finally {
                     if (loadingSpinner) loadingSpinner.classList.add('hidden');
                }
            }
        });

        sel.addEventListener('change', () => {
            const homeVal = homeChoice.getValue(true);
            const awayVal = awayChoice.getValue(true);
            if (!homeVal || homeVal === '' || !awayVal || awayVal === '') {
                 featureSection.classList.add('hidden');
                 resultEl.classList.add('hidden');
                 currentFeatures = null;
            }
        });
    });

    // ===================== PREDICT BUTTON =====================
    predictBtn.addEventListener('click', async () => {
        console.log('Tombol Prediksi diklik');

        // Simpan teks asli tombol
        const originalText = predictBtn.innerHTML;
        predictBtn.disabled = true;
        predictBtn.innerHTML = 'Memproses AI...';

        resultEl.classList.add('hidden');
        if (predictionContentEl) predictionContentEl.innerHTML = '';

        try {
            const features = getFeaturesForPrediction();
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({
                    league: leagueChoice.getValue(true),
                    features,
                    home_team: homeChoice.getValue(true),
                    away_team: awayChoice.getValue(true)
                })
            });

            const json = await res.json();
            showPredictionResult(json);
        } catch (err) {
            console.error('Error prediksi:', err);
            alert('Terjadi kesalahan saat prediksi: ' + err.message);
        } finally {
            predictBtn.disabled = false;
            predictBtn.innerHTML = originalText;
            console.log('Prediksi selesai, tombol dikembalikan.');
        }
    });

    loadLeagues();
});