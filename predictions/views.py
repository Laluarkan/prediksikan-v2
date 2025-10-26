# predictions/views.py

import json
import pandas as pd
import numpy as np
import os
import joblib
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings

# Impor model
from .models import PredictionHistory

# Impor decorator admin
from .decorators import admin_required

# Impor SEMUA fungsi logika
from .logic import * # ==========================================================
# ROUTES HALAMAN (YANG MERENDER HTML)
# ==========================================================

def home_page(request):
    """ Gantikan @app.route('/') """
    return render(request, 'home.html')

def index(request):
    """ Gantikan @app.route('/index') """
    # Login tidak lagi wajib (seperti di app.py Anda)
    leagues = list_leagues()
    context = {'leagues': leagues}
    return render(request, 'predictions/index.html', context)

def stats_page(request):
    """ Gantikan @app.route('/stats') """
    return render(request, 'predictions/stats.html')

@login_required
@admin_required
def add_data_page(request):
    """ Gantikan @app.route('/add_data') """
    leagues = list_leagues()
    context = {'leagues': leagues}
    return render(request, 'predictions/add_data.html', context)

# Halaman login, logout, dll ditangani oleh django-allauth
# (lihat urls.py utama)

# ==========================================================
# ROUTES API (YANG MENGEMBALIKAN JSON)
# ==========================================================

# Catatan: @csrf_exempt diperlukan jika Anda memanggil API ini dari
# JavaScript tanpa menyertakan token CSRF.
# Cara yang lebih baik adalah menyertakan token CSRF di header AJAX Anda.

def api_leagues(request):
    """ Gantikan @app.route('/api/leagues') """
    leagues = list_leagues()
    return JsonResponse({'status': 'ok', 'leagues': leagues})

@csrf_exempt
@require_POST
def api_team_stats(request):
    """ Gantikan @app.route('/api/team_stats', methods=['POST']) """
    try:
        body = json.loads(request.body)
        league = body.get('league')
        team = body.get('team')
    except json.JSONDecodeError:
        return JsonResponse({'status':'error','message':'Invalid JSON'}, status=400)

    if not all([league, team]):
        return JsonResponse({'status':'error','message':'league and team required'}, status=400)

    try:
        if league == ALL_LEAGUES:
            df, _ = find_team_league_and_df(team)
            if df is None:
                raise FileNotFoundError(f"Tim '{team}' tidak ditemukan di liga manapun.")
        else:
            df = load_league_dataset_by_name(league)
            
        stats = recent_stats_for_team(df, team)
        last_elo = None
        if 'HomeTeamElo' in df.columns and 'AwayTeamElo' in df.columns:
            tmp = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
            if not tmp.empty:
                row = tmp.sort_values('Date', ascending=False).iloc[0]
                last_elo = row['HomeTeamElo'] if row['HomeTeam'] == team else row['AwayTeamElo']
        
        return JsonResponse({'status': 'ok', 'stats': {'recent': stats, 'last_elo': last_elo}})
    
    except FileNotFoundError as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=404)

def api_teams(request):
    """ Gantikan @app.route('/api/teams') """
    league = request.GET.get('league')
    if not league:
        return JsonResponse({'status': 'error', 'message': 'parameter league diperlukan'}, status=400)

    try:
        if league == ALL_LEAGUES:
            all_teams = set()
            files = glob.glob(os.path.join(DATASET_DIR, '*.csv'))
            for f in files:
                try:
                    df = pd.read_csv(f)
                    all_teams.update(df['HomeTeam'])
                    all_teams.update(df['AwayTeam'])
                except Exception:
                    continue
            teams = sorted(list(all_teams))
            return JsonResponse({'status': 'ok', 'teams': teams})
        else:
            df = load_league_dataset_by_name(league)
            teams = sorted(set(df['HomeTeam']).union(set(df['AwayTeam'])))
            return JsonResponse({'status': 'ok', 'teams': teams})
    except FileNotFoundError as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=404)

@csrf_exempt
@require_POST
def api_features(request):
    """ Gantikan @app.route('/api/features', methods=['POST']) """
    try:
        body = json.loads(request.body)
        league = body.get('league')
        home = body.get('home')
        away = body.get('away')
    except json.JSONDecodeError:
        return JsonResponse({'status':'error','message':'Invalid JSON'}, status=400)

    if not all([league, home, away]):
        return JsonResponse({'status': 'error', 'message': 'league, home, away dibutuhkan'}, status=400)

    try:
        if league == ALL_LEAGUES:
            feats = compute_features_all_leagues(home, away)
        else:
            df = load_league_dataset_by_name(league)
            feats = compute_features_from_dataset(df, home, away)
        return JsonResponse({'status': 'ok', 'features': feats})
    except FileNotFoundError as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f"Gagal menghitung fitur: {e}"}, status=500)

@csrf_exempt
@require_POST
def api_predict(request):
    """ Gantikan @app.route('/api/predict', methods=['POST']) """
    try:
        body = json.loads(request.body)
        league = body.get('league')
        features = body.get('features')
        home_team = body.get('home_team')
        away_team = body.get('away_team')
    except json.JSONDecodeError:
        return JsonResponse({'status':'error','message':'Invalid JSON'}, status=400)

    if not all([league, features, home_team, away_team]):
        return JsonResponse({'status': 'error', 'message': 'Data liga, fitur, dan tim diperlukan'}, status=400)

    try:
        league_to_use = league
        if league == ALL_LEAGUES:
            _, found_league_name = find_team_league_and_df(home_team)
            if found_league_name is None:
                return JsonResponse({'status': 'error', 'message': f'Tidak dapat menemukan liga untuk tim {home_team} guna memuat model.'}, status=404)
            league_to_use = found_league_name
        
        league_dir = os.path.join(MODEL_DIR, league_to_use.lower().replace(' ', '_'))
        model_hda = joblib.load(os.path.join(league_dir, 'model_hda.pkl'))
        model_ou25 = joblib.load(os.path.join(league_dir, 'model_ou25.pkl'))
        model_btts = joblib.load(os.path.join(league_dir, 'model_btts.pkl'))
        scaler = joblib.load(os.path.join(league_dir, 'scaler.pkl'))
        le_ftr = joblib.load(os.path.join(league_dir, 'le_ftr.pkl'))
        le_ou = joblib.load(os.path.join(league_dir, 'le_ou.pkl'))
        le_btts = joblib.load(os.path.join(league_dir, 'le_btts.pkl'))
        
        df_features = pd.DataFrame([features])[FEATURE_COLUMNS]
        X_scaled = scaler.transform(df_features)
        
        probs_hda = model_hda.predict_proba(X_scaled)[0]
        probs_ou = model_ou25.predict_proba(X_scaled)[0]
        probs_btts = model_btts.predict_proba(X_scaled)[0]
        
        probs_hda_dict = {le_ftr.classes_[i]: float(probs_hda[i]) for i in range(len(probs_hda))}
        probs_ou_dict = {le_ou.classes_[i]: float(probs_ou[i]) for i in range(len(probs_ou))}
        probs_btts_dict = {le_btts.classes_[i]: float(probs_btts[i]) for i in range(len(probs_btts))}
        
        pred_hda = le_ftr.classes_[np.argmax(probs_hda)]
        pred_ou = le_ou.classes_[np.argmax(probs_ou)]
        pred_btts = le_btts.classes_[np.argmax(probs_btts)]
        
        result = {
            'HDA': {'label': pred_hda, 'probs': probs_hda_dict},
            'OU25': {'label': pred_ou, 'probs': probs_ou_dict},
            'BTTS': {'label': pred_btts, 'probs': probs_btts_dict}
        }

        # Panggil fungsi logic, kirimkan 'request.user'
        add_prediction_to_history(request.user, {
            'league': league,
            'home_team': home_team,
            'away_team': away_team,
            'prediction': result
        })
        
        return JsonResponse({'status': 'ok', 'prediction': result})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


@login_required
def api_history(request):
    """ Gantikan @app.route('/api/history') """
    # Gunakan Django ORM (QuerySet)
    histories_query = PredictionHistory.objects.filter(user=request.user).order_by('timestamp')[:20]
    
    history_list = []
    for item in histories_query:
        history_list.append({
            'league': item.league,
            'home_team': item.home_team,
            'away_team': item.away_team,
            'prediction': item.prediction_data,
            'timestamp': item.timestamp.isoformat()
        })
    return JsonResponse({'status': 'ok', 'history': history_list})

@csrf_exempt
@login_required
@require_POST
def api_clear_history(request):
    """ Gantikan @app.route('/api/clear_history', methods=['POST']) """
    try:
        PredictionHistory.objects.filter(user=request.user).delete()
        return JsonResponse({'status': 'ok', 'message': 'Riwayat dibersihkan'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Gagal membersihkan riwayat: {str(e)}'}, status=500)


# ==========================================================
# ROUTES ADMIN (ADD DATA)
# ==========================================================

@csrf_exempt # <-- Hati-hati menggunakan ini, pastikan admin terotentikasi
@login_required
@admin_required
@require_POST
def api_upload_csv(request):
    """ Gantikan @app.route('/api/upload_csv', methods=['POST']) """
    # Gantikan request.form dan request.files
    league = request.POST.get('league')
    file = request.FILES.get('file')

    if not all([league, file]):
        return JsonResponse({'status': 'error', 'message': 'Liga dan file CSV diperlukan'}, status=400)
    
    try:
        df_new = pd.read_csv(file)
        
        # Logika pembersihan dari app.py
        score_cols = ['FTHG', 'FTAG']
        for col in score_cols:
            if col in df_new.columns:
                df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
                df_new[col] = df_new[col].fillna(0).astype(int)

        df_existing = load_league_dataset_by_name(league)
        
        # Logika filter dari app.py
        mask_new = ~df_new.apply(lambda r: ((df_existing['Date'] == r['Date']) & (df_existing['HomeTeam'] == r['HomeTeam']) & (df_existing['AwayTeam'] == r['AwayTeam'])).any(), axis=1)
        df_new_only = df_new[mask_new].copy()

        if df_new_only.empty:
            return JsonResponse({'status': 'ok', 'message': 'Tidak ada pertandingan baru'}, status=200)

        df_new_full = update_elo_and_features(df_existing, df_new_only)
        df_output = df_new_full.copy()

        # Logika pemformatan dari app.py
        if 'Date' in df_output.columns:
            df_output['Date'] = pd.to_datetime(df_output['Date'], errors='coerce')
            df_output['Date'] = df_output['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        cols_to_format = list(df_output.columns)
        cols_skip = ['HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'Div', 'Date']
        for col in cols_to_format:
            if col not in cols_skip and pd.api.types.is_numeric_dtype(df_output[col]):
                df_output[col] = df_output[col].apply(format_float_clean)

        return JsonResponse({'status': 'ok', 'matches': df_output.to_dict(orient='records')})
    
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
@login_required
@admin_required
@require_POST
def api_save_new_matches(request):
    """ Gantikan @app.route('/api/save_new_matches', methods=['POST']) """
    try:
        body = json.loads(request.body)
        league = body.get('league')
        matches = body.get('matches')
    except json.JSONDecodeError:
        return JsonResponse({'status':'error','message':'Invalid JSON'}, status=400)

    try:
        df_existing = load_league_dataset_by_name(league)
        df_new = pd.DataFrame(matches)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
        league_file = file_name_from_pretty(league)
        path = os.path.join(DATASET_DIR, f"{league_file}.csv")
        df_combined.to_csv(path, index=False)
        
        return JsonResponse({'status': 'ok', 'message': 'Pertandingan baru berhasil disimpan'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)