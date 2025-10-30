# predictions/logic.py

import pandas as pd
import numpy as np
import os
import joblib
import glob
import functools
from datetime import datetime, timezone
from django.db.models import Q
from django.conf import settings
from .models import PredictionHistory 

# ==========================================================
# KONSTANTA 
# ==========================================================
DATASET_DIR = settings.DATASET_DIR
MODEL_DIR = settings.MODEL_DIR
FEATURE_COLUMNS = settings.FEATURE_COLUMNS
ALL_LEAGUES = settings.ALL_LEAGUES
INITIAL_ELO = settings.INITIAL_ELO

# ==========================================================
# FUNGSI UTILITAS (Tidak Berubah)
# ==========================================================
def pretty_league_name(file_name):
    # ... (kode Anda tidak berubah) ...
    name = file_name.replace('dataset_', '').replace('.csv', '')
    name = name.replace('_1', '')
    special_cases = { 'seriea': 'Serie A', 'laliga': 'La Liga', 'premierleague': 'Premier League', 'bundesliga': 'Bundesliga', 'ligue1': 'Ligue 1' }
    key = name.lower().replace('_', '')
    return special_cases.get(key, name.replace('_', ' ').title())

def file_name_from_pretty(league_display):
    # ... (kode Anda tidak berubah) ...
    league_lower = league_display.lower().replace(' ', '')
    files = glob.glob(os.path.join(DATASET_DIR, '*.csv'))
    for f in files:
        fname = os.path.splitext(os.path.basename(f))[0].lower().replace('_', '')
        if league_lower in fname:
            return os.path.splitext(os.path.basename(f))[0]
    return league_lower

def list_leagues():
    # ... (kode Anda tidak berubah) ...
    files = glob.glob(os.path.join(DATASET_DIR, '*.csv'))
    league_names = sorted([pretty_league_name(os.path.splitext(os.path.basename(p))[0]) for p in files])
    return [ALL_LEAGUES] + league_names

def load_league_dataset_by_name(league_display):
    # ... (kode Anda tidak berubah) ...
    league_lower = league_display.lower().replace(' ', '')
    files = glob.glob(os.path.join(DATASET_DIR, '*.csv'))
    matched_file = None
    for f in files:
        fname = os.path.splitext(os.path.basename(f))[0].lower().replace('_', '')
        if league_lower in fname:
            matched_file = f
            break
    if not matched_file:
        print("Available dataset files:", [os.path.basename(f) for f in files])
        raise FileNotFoundError(f"Dataset '{league_display}' tidak ditemukan di server.")
    df = pd.read_csv(matched_file)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

@functools.lru_cache(maxsize=128)
def find_team_league_and_df(team_name):
    # ... (kode Anda tidak berubah) ...
    all_league_names = [lg for lg in list_leagues() if lg != ALL_LEAGUES]
    for league_name in all_league_names:
        try:
            df = load_league_dataset_by_name(league_name)
            if team_name in set(df['HomeTeam']).union(set(df['AwayTeam'])):
                return df, league_name
        except FileNotFoundError:
            continue
    return None, None

def h2h_stats_all_leagues(home_team, away_team, window=5):
    # ... (kode Anda tidak berubah) ...
    all_matches = []
    files = glob.glob(os.path.join(DATASET_DIR, '*.csv'))
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            else: continue
            mask = ((df['HomeTeam']==home_team)&(df['AwayTeam']==away_team)) | ((df['HomeTeam']==away_team)&(df['AwayTeam']==home_team))
            all_matches.append(df[mask])
        except Exception as e: print(f"Gagal memproses file {f} untuk H2H: {e}")
    if not all_matches: return {'HTH_HomeWins':0,'HTH_AwayWins':0,'HTH_Draws':0, 'HTH_AvgHomeGoals':0,'HTH_AvgAwayGoals':0}
    combined_hth = pd.concat(all_matches).dropna(subset=['Date'])
    if combined_hth.empty: return {'HTH_HomeWins':0,'HTH_AwayWins':0,'HTH_Draws':0, 'HTH_AvgHomeGoals':0,'HTH_AvgAwayGoals':0}
    hth_sorted = combined_hth.sort_values('Date', ascending=False).head(window)
    hth_home_wins=hth_away_wins=hth_draws=0
    home_goals=[]; away_goals=[]
    for _, row in hth_sorted.iterrows():
        try:
            fthg = pd.to_numeric(row['FTHG'], errors='coerce'); ftag = pd.to_numeric(row['FTAG'], errors='coerce')
            if pd.isna(fthg) or pd.isna(ftag): continue
            if row['HomeTeam']==home_team: h_goals,a_goals=fthg,ftag
            else: h_goals,a_goals=ftag,fthg
            home_goals.append(h_goals); away_goals.append(a_goals)
            if h_goals>a_goals: hth_home_wins+=1
            elif h_goals<a_goals: hth_away_wins+=1
            else: hth_draws+=1
        except Exception: continue
    avg_home_goals = float(np.mean(home_goals)) if home_goals else 0
    avg_away_goals = float(np.mean(away_goals)) if away_goals else 0
    return {'HTH_HomeWins':hth_home_wins,'HTH_AwayWins':hth_away_wins,'HTH_Draws':hth_draws, 'HTH_AvgHomeGoals':avg_home_goals,'HTH_AvgAwayGoals':avg_away_goals}

def compute_features_all_leagues(home_team, away_team, window=5):
    # ... (kode Anda tidak berubah) ...
    df_home, _ = find_team_league_and_df(home_team)
    df_away, _ = find_team_league_and_df(away_team)
    if df_home is None: raise FileNotFoundError(f"Data tim {home_team} tidak ditemukan.")
    if df_away is None: raise FileNotFoundError(f"Data tim {away_team} tidak ditemukan.")
    last_home_elo=INITIAL_ELO
    if 'HomeTeamElo' in df_home.columns and 'AwayTeamElo' in df_home.columns:
        tmp_h=df_home[(df_home['HomeTeam']==home_team)|(df_home['AwayTeam']==home_team)]
        if not tmp_h.empty:
            row=tmp_h.sort_values('Date', ascending=False).iloc[0]
            last_home_elo=row['HomeTeamElo'] if row['HomeTeam']==home_team else row['AwayTeamElo']
    last_away_elo=INITIAL_ELO
    if 'HomeTeamElo' in df_away.columns and 'AwayTeamElo' in df_away.columns:
        tmp_a=df_away[(df_away['HomeTeam']==away_team)|(df_away['AwayTeam']==away_team)]
        if not tmp_a.empty:
            row=tmp_a.sort_values('Date', ascending=False).iloc[0]
            last_away_elo=row['HomeTeamElo'] if row['HomeTeam']==away_team else row['AwayTeamElo']
    home_stats=recent_stats_for_team(df_home, home_team)
    away_stats=recent_stats_for_team(df_away, away_team)
    hth=h2h_stats_all_leagues(home_team, away_team, window)
    return {'AvgH':'','AvgD':'','AvgA':'','Avg>2.5':'','Avg<2.5':'','HomeTeamElo':last_home_elo,'AwayTeamElo':last_away_elo,'EloDifference':last_home_elo-last_away_elo,'Home_AvgGoalsScored':home_stats['AvgGoalsScored'],'Home_AvgGoalsConceded':home_stats['AvgGoalsConceded'],'Home_Wins':home_stats['Wins'],'Home_Draws':home_stats['Draws'],'Home_Losses':home_stats['Losses'],'Away_AvgGoalsScored':away_stats['AvgGoalsScored'],'Away_AvgGoalsConceded':away_stats['AvgGoalsConceded'],'Away_Wins':away_stats['Wins'],'Away_Draws':away_stats['Draws'],'Away_Losses':away_stats['Losses'],'HTH_HomeWins':hth['HTH_HomeWins'],'HTH_AwayWins':hth['HTH_AwayWins'],'HTH_Draws':hth['HTH_Draws'],'HTH_AvgHomeGoals':hth['HTH_AvgHomeGoals'],'HTH_AvgAwayGoals':hth['HTH_AvgAwayGoals']}


# ==========================================================
# FUNGSI RIWAYAT PREDIKSI (DIPERBARUI)
# ==========================================================

# ▼▼▼ PERBARUI FUNGSI INI ▼▼▼
# Tambahkan parameter 'input_features'
def add_prediction_to_history(user, prediction_dict, input_features=None):
    if not user.is_authenticated:
        return None # <-- Kembalikan None jika tidak login
    try:
        new_history = PredictionHistory.objects.create(
            user = user,
            league = prediction_dict.get('league'),
            home_team = prediction_dict.get('home_team'),
            away_team = prediction_dict.get('away_team'),
            prediction_data = prediction_dict.get('prediction'),
            input_features = input_features
        )
        return new_history # <<< KEMBALIKAN OBJEK YANG BARU DIBUAT
    except Exception as e:
        print(f"Gagal menyimpan riwayat ke DB: {e}")
        return None # Kembalikan None jika gagal
# ▲▲▲ AKHIR PERUBAHAN ▲▲▲

# ==========================================================
# FUNGSI HITUNG STATISTIK (Tidak Berubah)
# ==========================================================

def expected_score(rating_a, rating_b):
    # ... (kode Anda tidak berubah) ...
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def recent_stats_for_team(df, team, window=5):
    # ... (kode Anda tidak berubah) ...
    mask = (df['HomeTeam'] == team) | (df['AwayTeam'] == team)
    team_games = df[mask].sort_values('Date', ascending=False) if 'Date' in df.columns else df[mask]
    if team_games.empty: return {'AvgGoalsScored': 0, 'AvgGoalsConceded': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0}
    recent = team_games.head(window)
    def gs(row): return row['FTHG'] if row['HomeTeam']==team else row['FTAG']
    def gc(row): return row['FTAG'] if row['HomeTeam']==team else row['FTHG']
    scored = recent.apply(gs, axis=1)
    conceded = recent.apply(gc, axis=1)
    def result(row):
        h,a = (row['FTHG'],row['FTAG']) if row['HomeTeam']==team else (row['FTAG'],row['FTHG'])
        return 'W' if h>a else 'D' if h==a else 'L'
    res = recent.apply(result, axis=1)
    return {'AvgGoalsScored': float(scored.mean()),'AvgGoalsConceded': float(conceded.mean()),'Wins': int((res=='W').sum()),'Draws': int((res=='D').sum()),'Losses': int((res=='L').sum())}

def h2h_stats(df, home_team, away_team, window=5):
    # ... (kode Anda tidak berubah) ...
    mask = ((df['HomeTeam']==home_team)&(df['AwayTeam']==away_team)) | ((df['HomeTeam']==away_team)&(df['AwayTeam']==home_team))
    hth = df[mask].sort_values('Date', ascending=False).head(window)
    hth_home_wins=hth_away_wins=hth_draws=0
    home_goals=[]; away_goals=[]
    for _, row in hth.iterrows():
        if row['HomeTeam']==home_team: h_goals,a_goals=row['FTHG'],row['FTAG']
        else: h_goals,a_goals=row['FTAG'],row['FTHG']
        home_goals.append(h_goals); away_goals.append(a_goals)
        if h_goals>a_goals: hth_home_wins+=1
        elif h_goals<a_goals: hth_away_wins+=1
        else: hth_draws+=1
    avg_home_goals = float(np.mean(home_goals)) if home_goals else 0
    avg_away_goals = float(np.mean(away_goals)) if away_goals else 0
    return {'HTH_HomeWins':hth_home_wins,'HTH_AwayWins':hth_away_wins,'HTH_Draws':hth_draws, 'HTH_AvgHomeGoals':avg_home_goals,'HTH_AvgAwayGoals':avg_away_goals}

def compute_features_from_dataset(df, home_team, away_team, window=5):
    # ... (kode Anda tidak berubah) ...
    last_home_elo=last_away_elo=INITIAL_ELO
    if 'HomeTeamElo' in df.columns and 'AwayTeamElo' in df.columns:
        tmp_h=df[(df['HomeTeam']==home_team)|(df['AwayTeam']==home_team)]
        if not tmp_h.empty:
            row=tmp_h.sort_values('Date', ascending=False).iloc[0]
            last_home_elo=row['HomeTeamElo'] if row['HomeTeam']==home_team else row['AwayTeamElo']
        tmp_a=df[(df['HomeTeam']==away_team)|(df['AwayTeam']==away_team)]
        if not tmp_a.empty:
            row=tmp_a.sort_values('Date', ascending=False).iloc[0]
            last_away_elo=row['HomeTeamElo'] if row['HomeTeam']==away_team else row['AwayTeamElo']
    home_stats=recent_stats_for_team(df, home_team)
    away_stats=recent_stats_for_team(df, away_team)
    hth=h2h_stats(df, home_team, away_team, window)
    return {'AvgH':'','AvgD':'','AvgA':'','Avg>2.5':'','Avg<2.5':'','HomeTeamElo':last_home_elo,'AwayTeamElo':last_away_elo,'EloDifference':last_home_elo-last_away_elo,'Home_AvgGoalsScored':home_stats['AvgGoalsScored'],'Home_AvgGoalsConceded':home_stats['AvgGoalsConceded'],'Home_Wins':home_stats['Wins'],'Home_Draws':home_stats['Draws'],'Home_Losses':home_stats['Losses'],'Away_AvgGoalsScored':away_stats['AvgGoalsScored'],'Away_AvgGoalsConceded':away_stats['AvgGoalsConceded'],'Away_Wins':away_stats['Wins'],'Away_Draws':away_stats['Draws'],'Away_Losses':away_stats['Losses'],'HTH_HomeWins':hth['HTH_HomeWins'],'HTH_AwayWins':hth['HTH_AwayWins'],'HTH_Draws':hth['HTH_Draws'],'HTH_AvgHomeGoals':hth['HTH_AvgHomeGoals'],'HTH_AvgAwayGoals':hth['HTH_AvgAwayGoals']}

def format_float_clean(number):
    # ... (kode Anda tidak berubah) ...
    if number is None or pd.isna(number) or str(number).strip() == '': return ""
    try:
        rounded_num = round(float(number), 2)
        return f"{rounded_num:g}"
    except (ValueError, TypeError): return str(number)

def update_elo_and_features(df_existing, df_new, window=5, K=30, initial_elo=INITIAL_ELO):
    # ... (kode Anda tidak berubah) ...
    if 'Date' in df_existing.columns: df_existing['Date'] = pd.to_datetime(df_existing['Date'], errors='coerce')
    if 'Date' in df_new.columns: df_new['Date'] = pd.to_datetime(df_new['Date'], errors='coerce')
    df_combined = pd.concat([df_existing, df_new], ignore_index=True).sort_values('Date').reset_index(drop=True)
    elo = {}
    new_rows = []
    if not df_existing.empty:
        df_existing_last = df_existing.sort_values('Date', ascending=False)
        all_teams = set(df_existing_last['HomeTeam']).union(set(df_existing_last['AwayTeam']))
        for team in all_teams:
            last_game = df_existing_last[(df_existing_last['HomeTeam'] == team) | (df_existing_last['AwayTeam'] == team)]
            if not last_game.empty:
                row = last_game.iloc[0]
                last_elo = row['HomeTeamElo'] if row['HomeTeam']==team else row['AwayTeamElo']
                elo[team] = last_elo
    start_idx = len(df_existing)
    for idx, row in df_combined.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        h_elo_pre = elo.get(home, initial_elo); a_elo_pre = elo.get(away, initial_elo)
        E_h = 1/(1+10**((a_elo_pre-h_elo_pre)/400)); E_a = 1-E_h
        if row['FTHG']>row['FTAG']: S_h,S_a=1,0
        elif row['FTHG']<row['FTAG']: S_h,S_a=0,1
        else: S_h,S_a=0.5,0.5
        h_elo_new=h_elo_pre+K*(S_h-E_h); a_elo_new=a_elo_pre+K*(S_a-E_a)
        elo[home],elo[away]=h_elo_new,a_elo_new
        if idx >= start_idx:
            df_past=df_combined.iloc[:idx]
            home_stats=recent_stats_for_team(df_past, home, window)
            away_stats=recent_stats_for_team(df_past, away, window)
            hth_mask=((df_past['HomeTeam']==home)&(df_past['AwayTeam']==away))|((df_past['HomeTeam']==away)&(df_past['AwayTeam']==home))
            hth=df_past[hth_mask].sort_values('Date', ascending=False).head(window)
            hth_home_wins=hth_away_wins=hth_draws=0
            home_goals=[]; away_goals=[]
            if not hth.empty:
                for _, r in hth.iterrows():
                    if r['HomeTeam']==home: h_g,a_g=r['FTHG'],r['FTAG']
                    else: h_g,a_g=r['FTAG'],r['FTHG']
                    home_goals.append(h_g); away_goals.append(a_g)
                    if h_g>a_g: hth_home_wins+=1
                    elif h_g<a_g: hth_away_wins+=1
                    else: hth_draws+=1
                hth_avg_home_goals=float(np.mean(home_goals))
                hth_avg_away_goals=float(np.mean(away_goals))
            else: hth_avg_home_goals=hth_avg_away_goals=0
            row_full = row.copy()
            odds_values = {}; odds_cols = ['AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5']
            for col in odds_cols:
                if col in row and pd.notna(row[col]): odds_values[col] = row[col]
                else: odds_values[col] = ''
            row_full.update({'HomeTeamElo':h_elo_new,'AwayTeamElo':a_elo_new,'EloDifference':h_elo_new-a_elo_new,'Home_AvgGoalsScored':home_stats['AvgGoalsScored'],'Home_AvgGoalsConceded':home_stats['AvgGoalsConceded'],'Home_Wins':home_stats['Wins'],'Home_Draws':home_stats['Draws'],'Home_Losses':home_stats['Losses'],'Away_AvgGoalsScored':away_stats['AvgGoalsScored'],'Away_AvgGoalsConceded':away_stats['AvgGoalsConceded'],'Away_Wins':away_stats['Wins'],'Away_Draws':away_stats['Draws'],'Away_Losses':away_stats['Losses'],'HTH_HomeWins':hth_home_wins,'HTH_AwayWins':hth_away_wins,'HTH_Draws':hth_draws,'HTH_AvgHomeGoals':hth_avg_home_goals,'HTH_AvgAwayGoals':hth_avg_away_goals,**odds_values})
            new_rows.append(row_full)
    return pd.DataFrame(new_rows)

def check_prediction_results(new_matches_df):
    """
    Menerima DataFrame dari pertandingan yang baru di-upload (dengan hasil),
    mencari prediksi user yang relevan di database, dan menandai
    apakah tebakan mereka benar (W) atau salah (L).
    """
    print(f"🔄 Memulai pengecekan {len(new_matches_df)} hasil pertandingan...")
    updated_count = 0

    # Pastikan kolom hasil ada (FTHG, FTAG, FTR)
    required_cols = ['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG']
    if not all(col in new_matches_df.columns for col in required_cols):
        print("❌ Pengecekan hasil dibatalkan: DataFrame kekurangan kolom hasil (FTHG/FTAG/FTR).")
        return 0

    # Loop melalui setiap pertandingan yang baru diselesaikan dari CSV
    for index, row in new_matches_df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Tentukan hasil sebenarnya dari data CSV
        actual_ftr = row['FTR']
        actual_ou = 'Over' if (row['FTHG'] + row['FTAG']) > 2.5 else 'Under'
        actual_btts = 'Yes' if (row['FTHG'] > 0 and row['FTAG'] > 0) else 'No'

        # Cari semua prediksi di database yang belum selesai untuk pertandingan ini
        pending_predictions = PredictionHistory.objects.filter(
            home_team=home_team,
            away_team=away_team,
            is_match_completed=False,
            is_preferred_choice=True # Hanya cek jika user membuat pilihan
        )
        
        # Jika tidak ada yang menebak pertandingan ini, lanjutkan
        if not pending_predictions.exists():
            continue

        # Update semua prediksi yang ditemukan
        for pred in pending_predictions:
            # 1. Cek HDA
            if pred.hda_chosen == actual_ftr:
                pred.hda_result = 'W' # Win
            elif pred.hda_chosen != 'N':
                pred.hda_result = 'L' # Lose
            
            # 2. Cek O/U
            if pred.over_under_chosen == actual_ou:
                pred.ou_result = 'W'
            elif pred.over_under_chosen != 'N':
                pred.ou_result = 'L'

            # 3. Cek BTTS
            if pred.btts_chosen == actual_btts:
                pred.btts_result = 'W'
            elif pred.btts_chosen != 'N':
                pred.btts_result = 'L'

            pred.is_match_completed = True
            pred.save()
            updated_count += 1

    print(f"✅ Pengecekan selesai. {updated_count} tebakan pengguna telah diperbarui.")
    return updated_count