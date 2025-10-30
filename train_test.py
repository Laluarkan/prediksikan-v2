import pandas as pd
import numpy as np
import warnings
import os
import joblib
import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# --- Impor SEMUA Model Klasik yang Diperlukan ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ==============================================================================
# KONFIGURASI
# ==============================================================================
DATASET_DIR = 'dataset_train'
MODEL_DIR = 'models' 

warnings.filterwarnings('ignore')

# --- Fitur 'Date' tidak digunakan ---
FEATURE_COLUMNS = [
    'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5',
    'HomeTeamElo', 'AwayTeamElo', 'EloDifference',
    'Home_AvgGoalsScored', 'Home_AvgGoalsConceded', 'Home_Wins', 'Home_Draws', 'Home_Losses',
    'Away_AvgGoalsScored', 'Away_AvgGoalsConceded', 'Away_Wins', 'Away_Draws', 'Away_Losses',
    'HTH_HomeWins', 'HTH_AwayWins', 'HTH_Draws',
    'HTH_AvgHomeGoals', 'HTH_AvgAwayGoals'
]

# ==============================================================================
# --- DEFINISI MODEL TERBAIK (BERDASARKAN LOG F1-SCORE ANDA) ---
# ==============================================================================

# 1. Peta dari String Nama Model ke Kelas Modelnya
MODEL_CLASSES = {
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
    "SVC": SVC,
    "KNN": KNeighborsClassifier,
    "DecisionTree": DecisionTreeClassifier,
}

# 2. Parameter Default untuk konsistensi
MODEL_DEFAULTS = {
    "LogisticRegression": {'random_state': 42, 'max_iter': 2000, 'n_jobs': -1},
    "RandomForest": {'random_state': 42, 'n_jobs': -1},
    "SVC": {'random_state': 42, 'probability': True},
    "KNN": {'n_jobs': -1},
    "DecisionTree": {'random_state': 42},
}

# 3. Peta Spesifikasi Model Terbaik (diekstrak dari log F1-Score Anda)
BEST_MODEL_SPECS = {
    'liga_belanda': { # Belanda
        'H/D/A': ('RandomForest', {'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 300}),
        'BTTS': ('KNN', {'n_neighbors': 11, 'p': 2, 'weights': 'uniform'}),
        'O/U 2.5': ('DecisionTree', {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 5})
    },
    'liga_belgia': {
        'H/D/A': ('LogisticRegression', {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'}),
        'BTTS': ('RandomForest', {'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 100}),
        'O/U 2.5': ('RandomForest', {'max_depth': 20, 'min_samples_leaf': 10, 'n_estimators': 100}) # Tie, chose RF
    },
    'liga_inggris': { # Inggris
        'H/D/A': ('RandomForest', {'max_depth': 10, 'min_samples_leaf': 5, 'n_estimators': 300}),
        'BTTS': ('RandomForest', {'max_depth': None, 'min_samples_leaf': 5, 'n_estimators': 300}),
        'O/U 2.5': ('LogisticRegression', {'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear'})
    },
    'liga_itali': { # Itali
        'H/D/A': ('LogisticRegression', {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}),
        'BTTS': ('RandomForest', {'max_depth': 5, 'min_samples_leaf': 10, 'n_estimators': 300}),
        'O/U 2.5': ('DecisionTree', {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 5})
    },
    'liga_jerman': { # Jerman
        'H/D/A': ('SVC', {'C': 0.1, 'kernel': 'linear'}),
        'BTTS': ('LogisticRegression', {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}),
        'O/U 2.5': ('RandomForest', {'max_depth': 5, 'min_samples_leaf': 5, 'n_estimators': 100})
    },
    'liga_perancis': { # Perancis
        'H/D/A': ('SVC', {'C': 1, 'kernel': 'linear'}),
        'BTTS': ('RandomForest', {'max_depth': 10, 'min_samples_leaf': 5, 'n_estimators': 300}),
        'O/U 2.5': ('DecisionTree', {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 2})
    },
    'liga_portugal': {
        'H/D/A': ('LogisticRegression', {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'}),
        'BTTS': ('RandomForest', {'max_depth': 5, 'min_samples_leaf': 5, 'n_estimators': 100}),
        'O/U 2.5': ('LogisticRegression', {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear'})
    },
    'liga_skotlandia': {
        'H/D/A': ('RandomForest', {'max_depth': 5, 'min_samples_leaf': 2, 'n_estimators': 300}),
        'BTTS': ('DecisionTree', {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 2}),
        'O/U 2.5': ('LogisticRegression', {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'})
    },
    'liga_spanyol': { # Spanyol
        'H/D/A': ('RandomForest', {'max_depth': 10, 'min_samples_leaf': 10, 'n_estimators': 100}),
        'BTTS': ('LogisticRegression', {'C': 1, 'penalty': 'l2', 'solver': 'liblinear'}),
        'O/U 2.5': ('KNN', {'n_neighbors': 21, 'p': 1, 'weights': 'uniform'})
    },
    'liga_turki': {
        'H/D/A': ('SVC', {'C': 1, 'kernel': 'linear'}),
        'BTTS': ('KNN', {'n_neighbors': 21, 'p': 1, 'weights': 'uniform'}),
        'O/U 2.5': ('RandomForest', {'max_depth': 5, 'min_samples_leaf': 10, 'n_estimators': 300})
    }
}


# ==============================================================================
# UTILITAS
# ==============================================================================

def get_model_from_spec(spec):
    """
    Membuat instance model dari spesifikasi (nama, params).
    """
    model_name, params = spec
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Model class untuk '{model_name}' not found.")
    
    model_class = MODEL_CLASSES[model_name]
    
    # Ambil parameter default dan gabungkan
    default_params = MODEL_DEFAULTS.get(model_name, {}).copy()
    final_params = {**default_params, **params}
    
    # Atur 'solver' untuk LogisticRegression jika tidak ada
    if model_name == 'LogisticRegression' and 'solver' not in final_params:
        final_params['solver'] = 'liblinear'
        
    return model_class(**final_params)

def pretty_league_name(file_name):
    """
    Mengubah nama file dataset menjadi nama liga yang rapi untuk tampilan.
    """
    name = file_name.replace('dataset_', '').replace('.csv', '')
    name = name.replace('_1', '')

    special_cases = {
        'eredivisie': 'Eredivisie',
        'seriea': 'Serie A',
        'laliga': 'La Liga',
        'premierleague': 'Premier League',
        'bundesliga': 'Bundesliga',
        'ligue1': 'Ligue 1',
        'liga_inggris': 'Premier League',
        'liga_spanyol': 'La Liga',
        'liga_itali': 'Serie A',
        'liga_jerman': 'Bundesliga',
        'liga_perancis': 'Ligue 1',
        'liga_belanda': 'Eredivisie',
        'liga_belgia': 'Liga Belgia',
        'liga_portugal': 'Liga Portugal',
        'liga_skotlandia': 'Liga Skotlandia',
        'liga_turki': 'Liga Turki'
    }

    key = name.lower().replace('_', '')
    if key in special_cases:
        return special_cases[key]
    
    return name.replace('_', ' ').title()

# ==============================================================================
# FUNGSI UTAMA UNTUK MELATIH DAN MENYIMPAN MODEL FINAL
# ==============================================================================
def train_and_save_final_models():
    print("🚀 Memulai proses training FINAL untuk semua liga...")
    print("🔥 Menggunakan model KLASIK TERBAIK spesifik per liga dan kategori.")
    
    dataset_paths = glob.glob(os.path.join(DATASET_DIR, '*.csv'))
    
    if not dataset_paths:
        print(f"❌ ERROR: Tidak ada file dataset .csv yang ditemukan di folder '{DATASET_DIR}'.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    for path in dataset_paths:
        try:
            filename_with_ext = os.path.basename(path)
            filename_base = os.path.splitext(filename_with_ext)[0]
            pretty_name = pretty_league_name(filename_base)
            league_folder_name = pretty_name.lower().replace(' ', '_') # e.g., 'la_liga'
            
            print(f"\n{'='*20} PROCESSING LEAGUE: {pretty_name.upper()} {'='*20}")

            # --- Logika Baru: Dapatkan spesifikasi model untuk liga ini ---
            if league_folder_name not in BEST_MODEL_SPECS:
                print(f"⚠️  WARNING: Tidak ada spesifikasi model terbaik untuk '{league_folder_name}'. Melewati liga ini.")
                continue
            
            specs = BEST_MODEL_SPECS[league_folder_name]

            # --- Logika Baru: Buat instance model secara dinamis ---
            model_hda_final = get_model_from_spec(specs['H/D/A'])
            model_btts_final = get_model_from_spec(specs['BTTS'])
            model_ou25_final = get_model_from_spec(specs['O/U 2.5'])

            print(f"   - Model H/D/A:   {specs['H/D/A'][0]}")
            print(f"   - Model BTTS:    {specs['BTTS'][0]}")
            print(f"   - Model O/U 2.5: {specs['O/U 2.5'][0]}")
            # --- Akhir Logika Baru ---

            league_model_dir = os.path.join(MODEL_DIR, league_folder_name)
            os.makedirs(league_model_dir, exist_ok=True)

            # --- Memuat data (tanpa parse_dates) ---
            data = pd.read_csv(path)
            print(f"✅ Berhasil memuat file: {filename_with_ext}")

            # --- Persiapan Target ---
            data['TotalGoals'] = data['FTHG'] + data['FTAG']
            data['OverUnder2.5'] = np.where(data['TotalGoals'] > 2.5, 'Over', 'Under')
            data['BTTS'] = np.where((data['FTHG'] > 0) & (data['FTAG'] > 0), 'Yes', 'No')

            # Cek fitur
            if not all(col in data.columns for col in FEATURE_COLUMNS):
                missing_cols = [col for col in FEATURE_COLUMNS if col not in data.columns]
                print(f"⚠️  WARNING: Fitur berikut tidak ditemukan: {missing_cols}. Melewati liga ini.")
                continue

            # --- Persiapan Data untuk Training Final (Menggunakan 100% Data) ---
            X = data[FEATURE_COLUMNS]
            y_ftr = data['FTR']
            y_ou = data['OverUnder2.5']
            y_btts = data['BTTS']

            # Handle NaN
            if X.isnull().sum().sum() > 0:
                print("   - Mengisi nilai NaN yang hilang dengan rata-rata...")
                X = X.fillna(X.mean())

            # --- Logika pemotongan data awal musim ---
            ROWS_TO_DROP = 10
            if len(X) <= ROWS_TO_DROP:
                print(f"⚠️  WARNING: Data tidak cukup (<= {ROWS_TO_DROP} baris). Melewati liga ini.")
                continue
                
            X = X.iloc[ROWS_TO_DROP:]
            y_ftr = y_ftr.iloc[ROWS_TO_DROP:]
            y_ou = y_ou.iloc[ROWS_TO_DROP:]
            y_btts = y_btts.iloc[ROWS_TO_DROP:]
            
            if X.empty:
                print(f"⚠️  WARNING: Tidak ada data tersisa setelah filtering. Melewati liga ini.")
                continue

            print(f"✅ Data disiapkan. Melatih model pada {len(X)} sampel...")

            # --- Scaling dan Encoding (pada 100% data) ---
            scaler_final = StandardScaler()
            X_scaled_final = scaler_final.fit_transform(X) # Data yang di-scale

            le_ftr_final, le_ou_final, le_btts_final = LabelEncoder(), LabelEncoder(), LabelEncoder()
            y_ftr_encoded_final = le_ftr_final.fit_transform(y_ftr)
            y_ou_encoded_final = le_ou_final.fit_transform(y_ou)
            y_btts_encoded_final = le_btts_final.fit_transform(y_btts)

            # --- Melatih Model Final ---
            # (Tidak perlu cek GaussianNB karena sudah kita keluarkan)
            
            print(f"   - Melatih model H/D/A ({specs['H/D/A'][0]})...")
            model_hda_final.fit(X_scaled_final, y_ftr_encoded_final)
            
            print(f"   - Melatih model BTTS ({specs['BTTS'][0]})...")
            model_btts_final.fit(X_scaled_final, y_btts_encoded_final) 
            
            print(f"   - Melatih model O/U 2.5 ({specs['O/U 2.5'][0]})...")
            model_ou25_final.fit(X_scaled_final, y_ou_encoded_final)
            
            print("✅ Model final berhasil dilatih.")

            # --- Menyimpan Semua Artefak ---
            print("   - Menyimpan artefak model...")
            joblib.dump(model_hda_final, os.path.join(league_model_dir, 'model_hda.pkl'))
            joblib.dump(model_btts_final, os.path.join(league_model_dir, 'model_btts.pkl'))
            joblib.dump(model_ou25_final, os.path.join(league_model_dir, 'model_ou25.pkl'))
            
            joblib.dump(scaler_final, os.path.join(league_model_dir, 'scaler.pkl'))
            joblib.dump(le_ftr_final, os.path.join(league_model_dir, 'le_ftr.pkl'))
            joblib.dump(le_ou_final, os.path.join(league_model_dir, 'le_ou.pkl'))
            joblib.dump(le_btts_final, os.path.join(league_model_dir, 'le_btts.pkl'))

            print(f"✨ Model final untuk {pretty_name.upper()} telah disimpan di '{league_model_dir}'.")

        except Exception as e:
            print(f"❌ GAGAL memproses {filename_with_ext}. Error: {e}")
            continue

    print(f"\n\n✨✨✨ Proses Selesai! Semua model final telah dilatih dan disimpan.")


if __name__ == '__main__':
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"Membuat folder dummy '{DATASET_DIR}'.")
        print(f"Pastikan untuk mengisinya dengan file .csv Anda sebelum menjalankan.")
    
    # Ganti nama fungsi yang dipanggil dari template
    train_and_save_final_models()

