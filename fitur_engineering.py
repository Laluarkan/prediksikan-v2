import pandas as pd
import numpy as np
from tqdm import tqdm  # Untuk progress bar, install dengan: pip install tqdm

# --- KONFIGURASI ---
INPUT_FILE = "dataset/data_gabungan.csv"
OUTPUT_FILE = "dataset/dataset_Liga_Turki_1.csv"

# Parameter dari logika Anda
INITIAL_ELO = 1500
K_FACTOR = 30
WINDOW = 5  # Jumlah pertandingan terakhir untuk statistik (form/H2H)
# --- AKHIR KONFIGURASI ---


def expected_score(rating_a, rating_b):
    """
    Menghitung skor Elo yang diharapkan untuk Pemain A.
    Diambil dari logic.py Anda.
    """
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def recent_stats_for_team(df_past, team, window=5):
    """
    Menghitung statistik performa terakhir untuk satu tim (form).
    Diadaptasi dari logic.py Anda (recent_stats_for_team).
    """
    # Hanya ambil data DARI MASA LALU (df_past)
    mask = (df_past['HomeTeam'] == team) | (df_past['AwayTeam'] == team)
    team_games = df_past[mask].sort_values('Date', ascending=False)
    
    if team_games.empty:
        return {'AvgGoalsScored': 0, 'AvgGoalsConceded': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0}

    # Ambil 5 pertandingan terakhir
    recent = team_games.head(window)

    def gs(row): 
        return row['FTHG'] if row['HomeTeam'] == team else row['FTAG']
    def gc(row): 
        return row['FTAG'] if row['HomeTeam'] == team else row['FTHG']

    scored = recent.apply(gs, axis=1)
    conceded = recent.apply(gc, axis=1)

    def result(row):
        h, a = (row['FTHG'], row['FTAG']) if row['HomeTeam'] == team else (row['FTAG'], row['FTHG'])
        if pd.isna(h) or pd.isna(a):
            return 'D'  # Asumsikan draw jika data skor tidak ada
        return 'W' if h > a else 'D' if h == a else 'L'

    res = recent.apply(result, axis=1)

    return {
        'AvgGoalsScored': float(scored.mean()) if not scored.empty else 0,
        'AvgGoalsConceded': float(conceded.mean()) if not conceded.empty else 0,
        'Wins': int((res == 'W').sum()),
        'Draws': int((res == 'D').sum()),
        'Losses': int((res == 'L').sum())
    }


def h2h_stats(df_past, home_team, away_team, window=5):
    """
    Menghitung statistik Head-to-Head (H2H) dari data masa lalu.
    Diadaptasi dari logic.py Anda (h2h_stats).
    """
    # Hanya ambil data DARI MASA LALU (df_past)
    mask = ((df_past['HomeTeam'] == home_team) & (df_past['AwayTeam'] == away_team)) | \
           ((df_past['HomeTeam'] == away_team) & (df_past['AwayTeam'] == home_team))
    
    hth = df_past[mask].sort_values('Date', ascending=False).head(window)

    hth_home_wins = 0
    hth_away_wins = 0
    hth_draws = 0
    home_goals = []
    away_goals = []

    if hth.empty:
        return {
            'HTH_HomeWins': 0, 'HTH_AwayWins': 0, 'HTH_Draws': 0,
            'HTH_AvgHomeGoals': 0, 'HTH_AvgAwayGoals': 0
        }

    for _, row in hth.iterrows():
        if pd.isna(row['FTHG']) or pd.isna(row['FTAG']):
            continue  # Lewati jika data skor tidak ada

        if row['HomeTeam'] == home_team:
            h_goals, a_goals = row['FTHG'], row['FTAG']
        else:
            h_goals, a_goals = row['FTAG'], row['FTHG']

        home_goals.append(h_goals)
        away_goals.append(a_goals)

        if h_goals > a_goals:
            hth_home_wins += 1
        elif h_goals < a_goals:
            hth_away_wins += 1
        else:
            hth_draws += 1

    avg_home_goals = float(np.mean(home_goals)) if home_goals else 0
    avg_away_goals = float(np.mean(away_goals)) if away_goals else 0

    return {
        'HTH_HomeWins': hth_home_wins,
        'HTH_AwayWins': hth_away_wins,
        'HTH_Draws': hth_draws,
        'HTH_AvgHomeGoals': avg_home_goals,
        'HTH_AvgAwayGoals': avg_away_goals
    }


def add_features_and_elo(df):
    """
    Fungsi utama untuk mengiterasi DataFrame dan menambahkan semua fitur.
    """
    
    # Pastikan FTHG dan FTAG adalah angka, ganti error/NaN dengan np.nan
    df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
    df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')

    # Kamus untuk menyimpan rating Elo terakhir setiap tim
    elo_ratings = {}
    
    # List untuk menampung data baris baru (fitur-fitur yang dihitung)
    new_features_list = []

    print("Memulai proses feature engineering (ini mungkin perlu waktu)...")
    
    # Gunakan tqdm untuk melihat progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Menghitung Fitur"):
        
        # 1. Tentukan data "masa lalu" (semua baris SEBELUM baris ini)
        df_past = df.iloc[:idx]
        
        # 2. Ambil nama tim
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        # 3. Ambil Elo SEBELUM pertandingan (atau ELO awal jika tim baru)
        home_elo_pre = elo_ratings.get(home_team, INITIAL_ELO)
        away_elo_pre = elo_ratings.get(away_team, INITIAL_ELO)

        # 4. Hitung Statistik Form (berdasarkan df_past)
        home_stats = recent_stats_for_team(df_past, home_team, window=WINDOW)
        away_stats = recent_stats_for_team(df_past, away_team, window=WINDOW)

        # 5. Hitung Statistik H2H (berdasarkan df_past)
        hth = h2h_stats(df_past, home_team, away_team, window=WINDOW)

        # 6. Hitung Elo SETELAH pertandingan (berdasarkan hasil di 'row' ini)
        E_h = expected_score(home_elo_pre, away_elo_pre)
        E_a = 1 - E_h

        # Tentukan hasil aktual (S_h, S_a)
        if pd.isna(row['FTHG']) or pd.isna(row['FTAG']):
            # Jika skor tidak ada, anggap draw, Elo tidak banyak berubah
            S_h, S_a = 0.5, 0.5 
        elif row['FTHG'] > row['FTAG']:
            S_h, S_a = 1, 0  # Home win
        elif row['FTHG'] < row['FTAG']:
            S_h, S_a = 0, 1  # Away win
        else:
            S_h, S_a = 0.5, 0.5 # Draw

        # Hitung Elo baru
        home_elo_new = home_elo_pre + K_FACTOR * (S_h - E_h)
        away_elo_new = away_elo_pre + K_FACTOR * (S_a - E_a)

        # 7. Update Elo di kamus utama untuk pertandingan berikutnya
        elo_ratings[home_team] = home_elo_new
        elo_ratings[away_team] = away_elo_new

        # 8. Kumpulkan semua fitur baru dalam satu kamus
        # ▼▼▼ PERUBAHAN DI SINI ▼▼▼
        # Kita bulatkan semua nilai desimal menjadi 2 angka di belakang koma
        feature_row = {
            'HomeTeamElo': round(home_elo_new, 2),
            'AwayTeamElo': round(away_elo_new, 2),
            'EloDifference': round(home_elo_new - away_elo_new, 2),
            
            'Home_AvgGoalsScored': round(home_stats['AvgGoalsScored'], 2),
            'Home_AvgGoalsConceded': round(home_stats['AvgGoalsConceded'], 2),
            'Home_Wins': home_stats['Wins'],  # Ini integer, tidak perlu round
            'Home_Draws': home_stats['Draws'], # Ini integer, tidak perlu round
            'Home_Losses': home_stats['Losses'],# Ini integer, tidak perlu round
            
            'Away_AvgGoalsScored': round(away_stats['AvgGoalsScored'], 2),
            'Away_AvgGoalsConceded': round(away_stats['AvgGoalsConceded'], 2),
            'Away_Wins': away_stats['Wins'],  # Ini integer, tidak perlu round
            'Away_Draws': away_stats['Draws'], # Ini integer, tidak perlu round
            'Away_Losses': away_stats['Losses'],# Ini integer, tidak perlu round
            
            'HTH_HomeWins': hth['HTH_HomeWins'], # Ini integer, tidak perlu round
            'HTH_AwayWins': hth['HTH_AwayWins'], # Ini integer, tidak perlu round
            'HTH_Draws': hth['HTH_Draws'],       # Ini integer, tidak perlu round
            'HTH_AvgHomeGoals': round(hth['HTH_AvgHomeGoals'], 2),
            'HTH_AvgAwayGoals': round(hth['HTH_AvgAwayGoals'], 2)
        }
        # ▲▲▲ AKHIR PERUBAHAN ▲▲▲
        
        # 9. Tambahkan kamus fitur ke list
        new_features_list.append(feature_row)

    # 10. Setelah loop selesai, gabungkan fitur baru dengan DataFrame asli
    print("\nProses perhitungan selesai. Menggabungkan data...")
    
    # Buat DataFrame dari list kamus
    df_features = pd.DataFrame(new_features_list, index=df.index)
    
    # Gabungkan df asli dengan df fitur baru
    df_final = pd.concat([df, df_features], axis=1)
    
    return df_final


# --- FUNGSI UTAMA UNTUK MENJALANKAN SKRIP ---
if __name__ == "__main__":
    try:
        # 1. Baca file CSV gabungan yang sudah urut
        print(f"Membaca file: {INPUT_FILE}...")
        df = pd.read_csv(INPUT_FILE)
        
        # 2. Pastikan kolom 'Date' adalah datetime dan data terurut
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)
        print(f"Ditemukan {len(df)} baris data.")

        # 3. Jalankan fungsi utama untuk menambah fitur
        df_hasil = add_features_and_elo(df)

        # 4. Simpan hasil akhir ke file baru
        df_hasil.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\n🎉 Berhasil! File dengan fitur lengkap telah disimpan di: {OUTPUT_FILE}")
        print("Contoh 5 baris terakhir dari data baru:")
        print(df_hasil.tail(5))

    except FileNotFoundError:
        print(f"❌ ERROR: File '{INPUT_FILE}' tidak ditemukan.")
        print("Pastikan file tersebut ada di folder yang sama dengan skrip ini.")
    except Exception as e:
        print(f"❌ ERROR: Terjadi kesalahan -> {e}")