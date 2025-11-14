// Fungsi getGradientColor, renderProbBlock, animateBars (tidak berubah)
      function getGradientColor(percent) { let r, g, b = 0; if (percent < 50) { r = 255; g = Math.round(5.1 * percent); } else { g = 255; r = Math.round(510 - 5.1 * percent); } return `linear-gradient(90deg, rgb(${r},${g},0) 0%, rgb(${r},${g},0) ${percent}%)`;}
      function renderProbBlock(title, probs) { if (!probs) return ''; let html = `<div class="prob-group"><h4>${title}</h4>`; for (const [label, val] of Object.entries(probs)) { const percent = (val * 100).toFixed(1); const gradient = getGradientColor(percent); html += `<div class="prob-item"><span>${label}</span><div class="prob-bar"><div class="bar" style="width:0%; background:${gradient}" data-percent="${percent}"></div></div><span>${percent}%</span></div>`; } html += `</div>`; return html; }
      function animateBars() { document.querySelectorAll('#result-details .bar').forEach(bar => { const percent = bar.dataset.percent || '0'; setTimeout(() => { bar.style.transition = 'width 1s ease'; bar.style.width = `${percent}%`; }, 50); }); }
      const resultContainer = document.getElementById('result-details');
      const predictionDataElement = document.getElementById('prediction-data-json');
      let predictionDataJson = null;

      if (predictionDataElement) {
          try {
              // Parse JSON dari textContent elemen script
              predictionDataJson = JSON.parse(predictionDataElement.textContent);
          } catch (e) {
              console.error("Gagal parse data prediksi JSON dari script tag:", e);
          }
      }

      if (resultContainer && predictionDataJson) {
         const p = predictionDataJson;
         const hdaLabel = p.HDA?.label || '-';
         const ou25Label = p.OU25?.label || '-';
         const bttsLabel = p.BTTS?.label || '-';

         resultContainer.innerHTML = `
            <div class="result-title">📊 Hasil Prediksi</div>
            <div id="main-result">
              🏠 (H/D/A): <b>${hdaLabel}</b><br>
              ⚽ (O/U 2.5): <b>${ou25Label}</b><br>
              🤝 (BTTS): <b>${bttsLabel}</b>
            </div>
            <div id="prob-bars" class="prob-section">
              ${renderProbBlock('🏠 (Home / Draw / Away)', p.HDA?.probs)}
              ${renderProbBlock('⚽ Over / Under 2.5', p.OU25?.probs)}
              ${renderProbBlock('🤝 Both Team To Score', p.BTTS?.probs)}
            </div>
         `;
         animateBars();
      } else if (resultContainer) {
          resultContainer.innerHTML = '<p style="text-align: center; color: var(--muted);">Data hasil prediksi tidak ditemukan atau tidak valid.</p>';
      }