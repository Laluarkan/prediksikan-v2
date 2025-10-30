# predictions/views.py

import json
import pandas as pd
import numpy as np
import os
import joblib
from django.shortcuts import render, redirect, get_object_or_404 # Tambah get_object_or_404
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings
import google.generativeai as genai
import traceback # <<< TAMBAHKAN IMPORT INI
import math # <<< TAMBAHKAN IMPORT INI
from django.db.models import Q
from .models import PredictionHistory
from .decorators import admin_required
from .logic import * 

# ▼▼▼ Konfigurasi Google AI ▼▼▼
try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        print("\033[93mPERINGATAN: GOOGLE_API_KEY environment variable belum di-set. Penjelasan AI tidak akan berfungsi.\033[0m")
        gemini_model = None
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Ganti 'gemini-pro' jika Anda menggunakan model lain
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        print("\033[92mGoogle AI (Gemini) berhasil dikonfigurasi.\033[0m")
except Exception as e:
    print(f"\033[91mERROR: Gagal mengkonfigurasi Google AI: {e}\033[0m")
    gemini_model = None
# ▲▲▲ ----------------------- ▲▲▲

# ==========================================================
# ROUTES HALAMAN (YANG MERENDER HTML)
# ==========================================================

def home_page(request):
    context = {}
    
    # Jika user login, hitung win rate mereka
    if request.user.is_authenticated:
        # Hanya hitung tebakan yang sudah selesai (is_match_completed=True)
        # dan di mana user membuat pilihan (is_preferred_choice=True)
        user_preds = PredictionHistory.objects.filter(
            user=request.user, 
            is_match_completed=True,
            is_preferred_choice=True
        )
        
        total_bets = 0
        total_wins = 0
        
        # Hitung HDA
        hda_bets = user_preds.exclude(hda_chosen='N').count()
        hda_wins = user_preds.filter(hda_result='W').count()
        
        # Hitung O/U
        ou_bets = user_preds.exclude(over_under_chosen='N').count()
        ou_wins = user_preds.filter(ou_result='W').count()
        
        # Hitung BTTS
        btts_bets = user_preds.exclude(btts_chosen='N').count()
        btts_wins = user_preds.filter(btts_result='W').count()
        
        total_bets = hda_bets + ou_bets + btts_bets
        total_wins = hda_wins + ou_wins + btts_wins
        
        win_rate = 0
        if total_bets > 0:
            win_rate = (total_wins / total_bets) * 100
            
        context['win_rate_stats'] = {
            'total_bets': total_bets,
            'total_wins': total_wins,
            'win_rate': win_rate
        }

    return render(request, 'home.html', context)

def index(request):
    leagues = list_leagues()
    context = {'leagues': leagues}
    return render(request, 'predictions/index.html', context)

def stats_page(request):
    return render(request, 'predictions/stats.html')

@login_required
@admin_required
def add_data_page(request):
    leagues = list_leagues()
    context = {'leagues': leagues}
    return render(request, 'predictions/add_data.html', context)

# ▼▼▼ TAMBAHKAN VIEW BARU INI ▼▼▼
@login_required
def history_page(request, history_id=None):
    # Ambil 10 riwayat terbaru untuk grid atas
    latest_histories = PredictionHistory.objects.filter(user=request.user).order_by('-timestamp')[:20]
    
    selected_history = None
    if history_id:
        # Ambil detail riwayat spesifik jika ID diberikan (pastikan hanya milik user)
        selected_history = get_object_or_404(PredictionHistory, id=history_id, user=request.user)
        
    context = {
        'latest_histories': latest_histories,
        'selected_history': selected_history,
        # Definisikan kolom fitur agar mudah diloop di template
        'feature_columns_meta': {
            'elo': ['HomeTeamElo', 'AwayTeamElo', 'EloDifference'],
            'home_stats': ['Home_AvgGoalsScored', 'Home_AvgGoalsConceded', 'Home_Wins', 'Home_Draws', 'Home_Losses'],
            'away_stats': ['Away_AvgGoalsScored', 'Away_AvgGoalsConceded', 'Away_Wins', 'Away_Draws', 'Away_Losses'],
            'hth': ['HTH_HomeWins', 'HTH_AwayWins', 'HTH_Draws', 'HTH_AvgHomeGoals', 'HTH_AvgAwayGoals'],
            'odds': ['AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5']
        },
         'feature_labels': { # Label yang lebih ramah (sesuaikan jika perlu)
            'HomeTeamElo': 'Elo Home', 'AwayTeamElo': 'Elo Away', 'EloDifference': 'Selisih Elo',
            'Home_AvgGoalsScored': 'Gol Dicetak (Home)', 'Home_AvgGoalsConceded': 'Gol Kebobolan (Home)', 'Home_Wins': 'Menang (Home)', 'Home_Draws': 'Seri (Home)', 'Home_Losses': 'Kalah (Home)',
            'Away_AvgGoalsScored': 'Gol Dicetak (Away)', 'Away_AvgGoalsConceded': 'Gol Kebobolan (Away)', 'Away_Wins': 'Menang (Away)', 'Away_Draws': 'Seri (Away)', 'Away_Losses': 'Kalah (Away)',
            'HTH_HomeWins': 'H2H Home Menang', 'HTH_AwayWins': 'H2H Away Menang', 'HTH_Draws': 'H2H Seri', 'HTH_AvgHomeGoals': 'H2H Gol Home', 'HTH_AvgAwayGoals': 'H2H Gol Away',
            'AvgH': 'Odds Home', 'AvgD': 'Odds Draw', 'AvgA': 'Odds Away', 'Avg>2.5': 'Odds Over 2.5', 'Avg<2.5': 'Odds Under 2.5'
        }
    }
    return render(request, 'predictions/history.html', context)
# ▲▲▲ AKHIR VIEW BARU ▲▲▲


# ==========================================================
# ROUTES API (YANG MENGEMBALIKAN JSON)
# ==========================================================

def api_leagues(request):
    # ... (kode Anda tidak berubah) ...
    leagues = list_leagues()
    return JsonResponse({'status': 'ok', 'leagues': leagues})

@require_POST
def api_team_stats(request):
    # ... (kode Anda tidak berubah) ...
    try:
        body = json.loads(request.body)
        league = body.get('league'); team = body.get('team')
    except json.JSONDecodeError: return JsonResponse({'status':'error','message':'Invalid JSON'}, status=400)
    if not all([league, team]): return JsonResponse({'status':'error','message':'league and team required'}, status=400)
    try:
        if league == ALL_LEAGUES:
            df, _ = find_team_league_and_df(team)
            if df is None: raise FileNotFoundError(f"Tim '{team}' tidak ditemukan.")
        else: df = load_league_dataset_by_name(league)
        stats = recent_stats_for_team(df, team); last_elo = None
        if 'HomeTeamElo' in df.columns and 'AwayTeamElo' in df.columns:
            tmp = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
            if not tmp.empty:
                row = tmp.sort_values('Date', ascending=False).iloc[0]
                last_elo = row['HomeTeamElo'] if row['HomeTeam'] == team else row['AwayTeamElo']
        return JsonResponse({'status': 'ok', 'stats': {'recent': stats, 'last_elo': last_elo}})
    except FileNotFoundError as e: return JsonResponse({'status': 'error', 'message': str(e)}, status=404)
    except Exception as e: return JsonResponse({'status':'error','message': f"Error: {e}"}, status=500)


def api_teams(request):
    # ... (kode Anda tidak berubah) ...
    league = request.GET.get('league')
    if not league: return JsonResponse({'status': 'error', 'message': 'parameter league diperlukan'}, status=400)
    try:
        if league == ALL_LEAGUES:
            all_teams = set()
            files = glob.glob(os.path.join(DATASET_DIR, '*.csv'))
            for f in files:
                try:
                    df = pd.read_csv(f); all_teams.update(df['HomeTeam']); all_teams.update(df['AwayTeam'])
                except Exception: continue
            teams = sorted(list(all_teams))
            return JsonResponse({'status': 'ok', 'teams': teams})
        else:
            df = load_league_dataset_by_name(league)
            teams = sorted(set(df['HomeTeam']).union(set(df['AwayTeam'])))
            return JsonResponse({'status': 'ok', 'teams': teams})
    except FileNotFoundError as e: return JsonResponse({'status': 'error', 'message': str(e)}, status=404)
    except Exception as e: return JsonResponse({'status':'error','message': f"Error: {e}"}, status=500)

@require_POST
def api_features(request):
    # ... (kode Anda tidak berubah) ...
    try:
        body = json.loads(request.body)
        league = body.get('league'); home = body.get('home'); away = body.get('away')
    except json.JSONDecodeError: return JsonResponse({'status':'error','message':'Invalid JSON'}, status=400)
    if not all([league, home, away]): return JsonResponse({'status': 'error', 'message': 'league, home, away dibutuhkan'}, status=400)
    try:
        if league == ALL_LEAGUES: feats = compute_features_all_leagues(home, away)
        else: df=load_league_dataset_by_name(league); feats=compute_features_from_dataset(df, home, away)
        return JsonResponse({'status': 'ok', 'features': feats})
    except FileNotFoundError as e: return JsonResponse({'status': 'error', 'message': str(e)}, status=404)
    except Exception as e: return JsonResponse({'status': 'error', 'message': f"Gagal menghitung fitur: {e}"}, status=500)

# ▼▼▼ PERBARUI VIEW INI ▼▼▼
@require_POST
def api_predict(request):
    try:
        body = json.loads(request.body)
        league = body.get('league')
        features = body.get('features') # Fitur input
        home_team = body.get('home_team')
        away_team = body.get('away_team')
    except json.JSONDecodeError:
        return JsonResponse({'status':'error','message':'Invalid JSON'}, status=400)

    if not all([league, features, home_team, away_team]):
        return JsonResponse({'status': 'error', 'message': 'Data liga, fitur, dan tim diperlukan'}, status=400)

    # --- 1. Lakukan Prediksi Model ML ---
    try:
        league_to_use = league
        if league == ALL_LEAGUES:
            _, found_league_name = find_team_league_and_df(home_team)
            if found_league_name is None: return JsonResponse({'status': 'error', 'message': f'Liga {home_team} tidak ditemukan.'}, status=404)
            league_to_use = found_league_name

        league_dir = os.path.join(MODEL_DIR, league_to_use.lower().replace(' ', '_'))
        if not os.path.isdir(league_dir): return JsonResponse({'status':'error','message':f'Model {league_to_use} tidak ditemukan.'}, status=404)

        # Load models, scaler, encoders
        model_hda=joblib.load(os.path.join(league_dir,'model_hda.pkl'))
        model_ou25=joblib.load(os.path.join(league_dir,'model_ou25.pkl'))
        model_btts=joblib.load(os.path.join(league_dir,'model_btts.pkl'))
        scaler=joblib.load(os.path.join(league_dir,'scaler.pkl'))
        le_ftr=joblib.load(os.path.join(league_dir,'le_ftr.pkl'))
        le_ou=joblib.load(os.path.join(league_dir,'le_ou.pkl'))
        le_btts=joblib.load(os.path.join(league_dir,'le_btts.pkl'))

        if not isinstance(features, dict): return JsonResponse({'status': 'error', 'message': 'Format fitur tidak valid.'}, status=400)
        feature_values = {col: features.get(col) for col in FEATURE_COLUMNS}
        df_features = pd.DataFrame([feature_values]).fillna(0) # Handle missing values

        X_scaled = scaler.transform(df_features)

        # Hitung probabilitas
        probs_hda = model_hda.predict_proba(X_scaled)[0]
        probs_ou = model_ou25.predict_proba(X_scaled)[0]
        probs_btts = model_btts.predict_proba(X_scaled)[0]
        probs_hda_dict = {le_ftr.classes_[i]: float(probs_hda[i]) for i in range(len(probs_hda))}
        probs_ou_dict = {le_ou.classes_[i]: float(probs_ou[i]) for i in range(len(probs_ou))}
        probs_btts_dict = {le_btts.classes_[i]: float(probs_btts[i]) for i in range(len(probs_btts))}
        pred_hda = le_ftr.classes_[np.argmax(probs_hda)]
        pred_ou = le_ou.classes_[np.argmax(probs_ou)]
        pred_btts = le_btts.classes_[np.argmax(probs_btts)]

        # Hasil prediksi ML
        ml_result = {
            'HDA': {'label': pred_hda, 'probs': probs_hda_dict},
            'OU25': {'label': pred_ou, 'probs': probs_ou_dict},
            'BTTS': {'label': pred_btts, 'probs': probs_btts_dict}
        }

    except FileNotFoundError as e:
         return JsonResponse({'status':'error','message':f'File model tidak ditemukan: {e}'}, status=500)
    except Exception as e:
        print(f"\033[91mError during ML prediction: {e}\033[0m")
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': f'Kesalahan internal saat prediksi ML: {e}'}, status=500)

    # --- 2. Hasilkan Penjelasan AI (Gemini) ---
    ai_explanation = "Penjelasan AI tidak tersedia." # Default
    if gemini_model:
        try:
            # Hitung total gol untuk data kontekstual
            home_wins = features.get('Home_Wins', 0)
            home_draws = features.get('Home_Draws', 0)
            home_losses = features.get('Home_Losses', 0)
            home_count = home_wins + home_draws + home_losses or 5
            home_scored_total = math.ceil(features.get('Home_AvgGoalsScored', 0) * home_count)
            home_conceded_total = math.ceil(features.get('Home_AvgGoalsConceded', 0) * home_count)

            away_wins = features.get('Away_Wins', 0)
            away_draws = features.get('Away_Draws', 0)
            away_losses = features.get('Away_Losses', 0)
            away_count = away_wins + away_draws + away_losses or 5
            away_scored_total = math.ceil(features.get('Away_AvgGoalsScored', 0) * away_count)
            away_conceded_total = math.ceil(features.get('Away_AvgGoalsConceded', 0) * away_count)

            # ------------------------------
            # PROMPT BARU UNTUK GEMINI
            # ------------------------------
            prompt = f"""
            Anda adalah analis sepak bola profesional. Berikan penjelasan singkat dan objektif (1–3 kalimat)
            tentang hasil prediksi pertandingan {home_team} vs {away_team} di {league}, 
            berdasarkan data dan tren performa berikut.

            📊 **Prediksi Model:**
            - Hasil: {ml_result['HDA']['label']} ({ml_result['HDA']['probs'].get(ml_result['HDA']['label'], 0)*100:.1f}%)
            - Over/Under 2.5 Gol: {ml_result['OU25']['label']} ({ml_result['OU25']['probs'].get(ml_result['OU25']['label'], 0)*100:.1f}%)
            - Kedua Tim Cetak Gol (BTTS): {ml_result['BTTS']['label']} ({ml_result['BTTS']['probs'].get(ml_result['BTTS']['label'], 0)*100:.1f}%)

            ⚙️ **Data Statistik Kunci:**
            - Elo Rating: {home_team} {features.get('HomeTeamElo', 0):.0f}, {away_team} {features.get('AwayTeamElo', 0):.0f}
            (Selisih: {features.get('EloDifference', 0):.1f})
            - Form {home_team} (5 laga terakhir): M{home_wins} S{home_draws} K{home_losses} 
            (Gol: {home_scored_total}-{home_conceded_total})
            - Form {away_team} (5 laga terakhir): M{away_wins} S{away_draws} K{away_losses} 
            (Gol: {away_scored_total}-{away_conceded_total})
            - Head-to-Head (5 laga): {home_team} menang {features.get('HTH_HomeWins', 0)}, 
            {away_team} menang {features.get('HTH_AwayWins', 0)}, Seri {features.get('HTH_Draws', 0)}

            🧠 **Instruksi:**
            Jelaskan *mengapa* hasil prediksi model (terutama hasil utama HDA) bisa terjadi,
            berdasarkan perbedaan kekuatan tim, tren gol, dan performa terkini.
            Gunakan gaya analisis alami seperti komentator sepak bola profesional.
            Hindari kata-kata pasti seperti "pasti menang" atau "sudah pasti".
            """

            print(f"\n--- Prompt Gemini untuk {home_team} vs {away_team} ---")
            print(prompt)
            print("---------------------------------------\n")

            # Kirim ke Gemini
            response = gemini_model.generate_content(prompt)
            ai_explanation = response.text

            print(f"\033[94mGemini Explanation: {ai_explanation}\033[0m")

        except Exception as gemini_error:
            print(f"\033[91mERROR: Gagal memanggil Google AI API: {gemini_error}\033[0m")
            traceback.print_exc()
            ai_explanation = "Tidak dapat memuat penjelasan AI saat ini."

    # --- 3. Simpan Riwayat ---

    new_history_obj = add_prediction_to_history(
        user=request.user, 
        prediction_dict={
            'league': league, 
            'home_team': home_team, 
            'away_team': away_team, 
            'prediction': ml_result
        },
        input_features=features
    )
    
    # Ambil ID dari objek yang baru dibuat
    new_history_id = new_history_obj.id if new_history_obj else None

    # --- 4. Kirim Respons ---
    return JsonResponse({
        'status': 'ok',
        'prediction': ml_result,
        'explanation': ai_explanation, # Penjelasan dari Gemini (atau pesan error)
        'history_id': new_history_id
    })
# ▲▲▲ AKHIR PERUBAHAN API PREDICT ▲▲▲

# ▼▼▼ PERBARUI VIEW INI ▼▼▼
@login_required
def api_history(request):
    # Ambil 10 riwayat terbaru untuk halaman riwayat
    histories_query = PredictionHistory.objects.filter(user=request.user).order_by('-timestamp')[:20] 
    
    history_list = []
    for item in histories_query:
        history_list.append({
            'id': item.id, # <-- TAMBAHKAN ID
            'league': item.league,
            'home_team': item.home_team,
            'away_team': item.away_team,
            'prediction': item.prediction_data,
            'timestamp': item.timestamp.isoformat()
        })
    # Tidak perlu reverse(), order_by('-timestamp') sudah benar
    return JsonResponse({'status': 'ok', 'history': history_list})
# ▲▲▲ AKHIR PERUBAHAN api_history ▲▲▲

@login_required
@require_POST
def api_clear_history(request):
    # ... (kode Anda tidak berubah) ...
    try:
        PredictionHistory.objects.filter(user=request.user).delete()
        return JsonResponse({'status': 'ok', 'message': 'Riwayat dibersihkan'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Gagal membersihkan riwayat: {str(e)}'}, status=500)

# ==========================================================
# ROUTES ADMIN (ADD DATA) - Tidak Berubah
# ==========================================================
@login_required
@admin_required
@require_POST
def api_upload_csv(request):
    """
    Menerima upload CSV, memfilter hanya kolom input dasar,
    menghitung fitur (Elo, Stats, H2H) di backend,
    dan mengembalikan fitur lengkap sebagai JSON.
    """
    league = request.POST.get('league')
    file = request.FILES.get('file')

    if not all([league, file]):
        return JsonResponse({'status': 'error', 'message': 'Liga dan file CSV diperlukan'}, status=400)
    
    # Tentukan 11 kolom dasar yang kita ambil dari CSV
    INPUT_COLUMNS = [
        'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
        'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5'
    ]

    try:
        # 1. Baca CSV
        df_new_raw = pd.read_csv(file)
        
        # 2. Cek apakah kolom dasar ada
        if not all(col in df_new_raw.columns for col in INPUT_COLUMNS):
            missing_cols = [col for col in INPUT_COLUMNS if col not in df_new_raw.columns]
            return JsonResponse({'status': 'error', 'message': f'CSV Anda kekurangan kolom: {", ".join(missing_cols)}'}, status=400)

        # 3. Filter HANYA 11 kolom input (memenuhi permintaan Anda)
        df_new = df_new_raw[INPUT_COLUMNS].copy()

        # 4. Bersihkan tipe data (SANGAT PENTING)
        
        # ▼▼▼ PERBAIKAN FORMAT TANGGAL ▼▼▼
        # Coba tebak format tanggal (misal DD/MM/YYYY atau YYYY-MM-DD)
        # dayfirst=True membantu pandas memprioritaskan DD/MM/YYYY
        df_new['Date'] = pd.to_datetime(df_new['Date'], dayfirst=True, errors='coerce') 
        
        # Buang baris yang tanggalnya tidak valid (NaT)
        if df_new.empty:
             return JsonResponse({'status': 'error', 'message': 'Tidak ada baris dengan format tanggal yang valid di CSV.'}, status=400)
        # ▲▲▲ AKHIR PERBAIKAN TANGGAL ▲▲▲

        # Konversi Skor (sudah ada di kode Anda)
        score_cols = ['FTHG', 'FTAG']
        for col in score_cols:
            if col in df_new.columns:
                df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
                df_new[col] = df_new[col].fillna(0).astype(int)

        # 5. Muat data lama (sudah di-convert ke datetime oleh load_league_dataset_by_name)
        df_existing = load_league_dataset_by_name(league)
        
        # 6. Filter pertandingan baru (sekarang membandingkan datetime vs datetime)
        # Buat 'match_id' unik untuk perbandingan
        df_existing['match_id_str'] = df_existing['Date'].dt.strftime('%Y-%m-%d') + df_existing['HomeTeam'] + df_existing['AwayTeam']
        existing_matches_str = set(df_existing['match_id_str'])
        
        df_new['match_id_str'] = df_new['Date'].dt.strftime('%Y-%m-%d') + df_new['HomeTeam'] + df_new['AwayTeam']

        mask_new = ~df_new['match_id_str'].isin(existing_matches_str)
        
        df_new_only = df_new[mask_new].copy()
        
        # Hapus kolom helper
        df_new_only.drop(columns=['match_id_str'], inplace=True)
        if 'match_id_str' in df_existing.columns:
            df_existing.drop(columns=['match_id_str'], inplace=True)


        if df_new_only.empty:
            return JsonResponse({'status': 'ok', 'message': 'Tidak ada pertandingan baru yang ditemukan.'}, status=200)

        # 7. Hitung fitur (Elo, Stats, H2H) menggunakan logic.py
        df_new_full = update_elo_and_features(df_existing, df_new_only)

        check_prediction_results(df_new_full)
        
        # 8. Format output untuk ditampilkan di tabel
        df_output = df_new_full.copy()
        
        if 'Date' in df_output.columns:
            df_output['Date'] = pd.to_datetime(df_output['Date'], errors='coerce')
            df_output['Date'] = df_output['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        cols_to_format = list(df_output.columns)
        cols_skip = ['HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'Div', 'Date']
        for col in cols_to_format:
            if col not in cols_skip and pd.api.types.is_numeric_dtype(df_output[col]):
                df_output[col] = df_output[col].apply(format_float_clean)

        return JsonResponse({'status': 'ok', 'matches': df_output.to_dict(orient='records')})
    
    except FileNotFoundError as e:
         return JsonResponse({'status': 'error', 'message': f'Gagal memuat dataset liga: {e}'}, status=404)
    except Exception as e:
        traceback.print_exc() # Cetak error lengkap di terminal Django
        return JsonResponse({'status': 'error', 'message': f'Terjadi kesalahan internal: {e}'}, status=500)

@login_required
@admin_required
@require_POST
def api_save_new_matches(request):
    # ... (kode Anda tidak berubah) ...
    try:
        body = json.loads(request.body)
        league = body.get('league'); matches = body.get('matches')
    except json.JSONDecodeError: return JsonResponse({'status':'error','message':'Invalid JSON'}, status=400)
    if not all([league, matches]): return JsonResponse({'status':'error','message':'Liga dan matches diperlukan'}, status=400)
    try:
        df_existing = load_league_dataset_by_name(league)
        df_new = pd.DataFrame(matches)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        league_file = file_name_from_pretty(league)
        path = os.path.join(DATASET_DIR, f"{league_file}.csv")
        df_combined.to_csv(path, index=False)
        return JsonResponse({'status': 'ok', 'message': 'Pertandingan baru berhasil disimpan'})
    except Exception as e: return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@require_POST
@login_required
def api_save_choice(request):
    """ Menerima pilihan user (HDA/OU/BTTS) dan menyimpannya ke database. """
    try:
        body = json.loads(request.body)
        history_id = body.get('id')
        choice_type = body.get('type') # HDA, OU, atau BTTS
        choice_value = body.get('value') # H, D, A, Over, Under, Yes, No
        
        history = PredictionHistory.objects.get(id=history_id, user=request.user)
        
        # Update kolom sesuai pilihan
        if choice_type == 'HDA':
            history.hda_chosen = choice_value
        elif choice_type == 'OU':
            history.over_under_chosen = choice_value
        elif choice_type == 'BTTS':
            history.btts_chosen = choice_value
            
        history.is_preferred_choice = True # Asumsikan pilihan ini yang terbaik
        history.save()
        
        return JsonResponse({'status': 'ok', 'message': f'Pilihan {choice_type} ({choice_value}) berhasil disimpan sebagai pilihan terbaik.'})
        
    except PredictionHistory.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Riwayat tidak ditemukan atau bukan milik Anda.'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Gagal menyimpan pilihan: {str(e)}'}, status=500)