import pandas as pd
import numpy as np
import warnings
import os
import joblib
import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Impor SEMUA model yang kita butuhkan ---
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier # Diperlukan untuk Liga Perancis

# ==============================================================================
# KONFIGURASI
# ==============================================================================
DATASET_DIR = 'dataset'
MODEL_DIR = 'models' 

warnings.filterwarnings('ignore')

FEATURE_COLUMNS = [
    'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5',
    'HomeTeamElo', 'AwayTeamElo', 'EloDifference',
    'Home_AvgGoalsScored', 'Home_AvgGoalsConceded', 'Home_Wins', 'Home_Draws', 'Home_Losses',
    'Away_AvgGoalsScored', 'Away_AvgGoalsConceded', 'Away_Wins', 'Away_Draws', 'Away_Losses',
    'HTH_HomeWins', 'HTH_AwayWins', 'HTH_Draws',
    'HTH_AvgHomeGoals', 'HTH_AvgAwayGoals'
]

# ==============================================================================
# === KONFIGURASI MODEL TERBAIK (BERDASARKAN HASIL find_best_models.py) ===
# ==============================================================================
# Kunci dictionary (e.g., 'liga_belanda') harus cocok dengan output 
# dari 'pretty_name.lower().replace(' ', '_')' di fungsi utama.

BEST_MODELS_CONFIG = {
    'liga_belanda': {
        'H/D/A': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
        ),
        'BTTS': (
            SVC(random_state=42, probability=True),
            {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}
        ),
        'O/U 2.5': (
            LogisticRegression(random_state=42, max_iter=1000),
            {'C': 0.1, 'solver': 'lbfgs'}
        )
    },
    'liga_belgia': {
        'H/D/A': (
            XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, objective='multi:softmax', eval_metric='mlogloss'),
            {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100}
        ),
        'BTTS': (
            SVC(random_state=42, probability=True),
            {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}
        ),
        'O/U 2.5': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200}
        )
    },
    'liga_inggris': {
        'H/D/A': (
            XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, objective='multi:softmax', eval_metric='mlogloss'),
            {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200}
        ),
        'BTTS': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
        ),
        'O/U 2.5': (
            LogisticRegression(random_state=42, max_iter=1000),
            {'C': 0.1, 'solver': 'liblinear'}
        )
    },
    'liga_itali': {
        'H/D/A': (
            LogisticRegression(random_state=42, max_iter=1000),
            {'C': 0.1, 'solver': 'liblinear'}
        ),
        'BTTS': (
            KNeighborsClassifier(n_jobs=-1),
            {'n_neighbors': 3, 'weights': 'uniform'}
        ),
        'O/U 2.5': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
        )
    },
    'liga_jerman': {
        'H/D/A': (
            LogisticRegression(random_state=42, max_iter=1000),
            {'C': 1, 'solver': 'liblinear'}
        ),
        'BTTS': (
            GaussianNB(),
            {'var_smoothing': 1e-09}
        ),
        'O/U 2.5': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
        )
    },
    'liga_perancis': {
        'H/D/A': (
            XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, objective='multi:softmax', eval_metric='mlogloss'),
            {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100}
        ),
        'BTTS': (
            XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'),
            {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200}
        ),
        'O/U 2.5': (
            MLPClassifier(random_state=42, max_iter=1000),
            {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (100,), 'solver': 'adam'}
        )
    },
    'liga_portugal': {
        'H/D/A': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100}
        ),
        'BTTS': (
            GaussianNB(),
            {'var_smoothing': 1e-09}
        ),
        'O/U 2.5': (
            LogisticRegression(random_state=42, max_iter=1000),
            {'C': 0.1, 'solver': 'lbfgs'}
        )
    },
    'liga_skotlandia': {
        'H/D/A': (
            XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, objective='multi:softmax', eval_metric='mlogloss'),
            {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100}
        ),
        'BTTS': (
            LogisticRegression(random_state=42, max_iter=1000),
            {'C': 1, 'solver': 'liblinear'}
        ),
        'O/U 2.5': (
            LogisticRegression(random_state=42, max_iter=1000),
            {'C': 0.1, 'solver': 'lbfgs'}
        )
    },
    'liga_spanyol': {
        'H/D/A': (
            XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, objective='multi:softmax', eval_metric='mlogloss'),
            {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200}
        ),
        'BTTS': (
            LogisticRegression(random_state=42, max_iter=1000),
            {'C': 0.1, 'solver': 'liblinear'}
        ),
        'O/U 2.5': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200}
        )
    },
    'liga_turki': {
        'H/D/A': (
            LogisticRegression(random_state=42, max_iter=1000),
            {'C': 0.1, 'solver': 'liblinear'}
        ),
        'BTTS': (
            SVC(random_state=42, probability=True),
            {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}
        ),
        'O/U 2.5': (
            GaussianNB(),
            {'var_smoothing': 1e-09}
        )
    }
}


# ==============================================================================
# UTILITAS
# ==============================================================================
def pretty_league_name(file_name):
    """
    Mengubah nama file dataset menjadi nama liga yang rapi untuk tampilan.
    DIEDIT agar sesuai dengan output Anda (misal 'liga_jerman' -> 'Liga Jerman')
    """
    name = file_name.replace('dataset_', '').replace('.csv', '')
    name = name.replace('_1', '') # Menghapus _1 jika ada
    
    # Menggunakan logika sederhana untuk mengubah 'liga_jerman' -> 'Liga Jerman'
    return name.replace('_', ' ').title()

# ==============================================================================
# FUNGSI UTAMA (DIMODIFIKASI)
# ==============================================================================
def train_and_evaluate_all_leagues():
    print("🚀 Memulai proses training dan evaluasi untuk semua liga...")
    
    dataset_paths = glob.glob(os.path.join(DATASET_DIR, '*.csv'))
    all_results = []

    if not dataset_paths:
        print(f"❌ ERROR: Tidak ada file dataset .csv yang ditemukan di folder '{DATASET_DIR}'.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    for path in dataset_paths:
        try:
            filename_with_ext = os.path.basename(path)
            filename_base = os.path.splitext(filename_with_ext)[0]
            pretty_name = pretty_league_name(filename_base)
            league_folder_name = pretty_name.lower().replace(' ', '_')
            
            print(f"\n{'='*20} PROCESSING LEAGUE: {pretty_name.upper()} {'='*20}")

            # --- Ambil Konfigurasi Model Terbaik untuk Liga Ini ---
            league_config = BEST_MODELS_CONFIG.get(league_folder_name)
            if not league_config:
                print(f"⚠️  WARNING: Tidak ada konfigurasi model di BEST_MODELS_CONFIG untuk '{league_folder_name}'. Melewati liga ini.")
                continue
            
            print(f"✅ Konfigurasi model untuk {pretty_name.upper()} ditemukan.")

            league_model_dir = os.path.join(MODEL_DIR, league_folder_name)
            os.makedirs(league_model_dir, exist_ok=True)

            data = pd.read_csv(path)
            
            data['TotalGoals'] = data['FTHG'] + data['FTAG']
            data['OverUnder2.5'] = np.where(data['TotalGoals'] > 2.5, 'Over', 'Under')
            data['BTTS'] = np.where((data['FTHG'] > 0) & (data['FTAG'] > 0), 'Yes', 'No')

            if not all(col in data.columns for col in FEATURE_COLUMNS):
                print(f"⚠️  WARNING: Tidak semua fitur ditemukan di {filename_with_ext}. Melewati liga ini.")
                continue

            X = data[FEATURE_COLUMNS]
            y_ftr = data['FTR']
            y_ou = data['OverUnder2.5']
            y_btts = data['BTTS']

            if X.isnull().sum().sum() > 0:
                X = X.fillna(X.mean())

            WINDOW = 5
            X = X.iloc[WINDOW:]
            y_ftr = y_ftr.iloc[WINDOW:]
            y_ou = y_ou.iloc[WINDOW:]
            y_btts = y_btts.iloc[WINDOW:]
            
            if X.empty:
                print(f"⚠️  WARNING: Tidak ada data tersisa setelah di-windowing. Melewati.")
                continue
                
            X_train, X_test, y_ftr_train, y_ftr_test = train_test_split(X, y_ftr, test_size=0.2, shuffle=False)
            _, _, y_ou_train, y_ou_test = train_test_split(X, y_ou, test_size=0.2, shuffle=False)
            _, _, y_btts_train, y_btts_test = train_test_split(X, y_btts, test_size=0.2, shuffle=False)
            print("✅ Data berhasil dipisah menjadi data latih (80%) dan uji (20%).")

            scaler_eval = StandardScaler()
            X_train_scaled = scaler_eval.fit_transform(X_train)
            X_test_scaled = scaler_eval.transform(X_test)

            le_ftr_eval, le_ou_eval, le_btts_eval = LabelEncoder(), LabelEncoder(), LabelEncoder()
            y_ftr_train_encoded = le_ftr_eval.fit_transform(y_ftr_train)
            y_ou_train_encoded = le_ou_eval.fit_transform(y_ou_train)
            y_btts_train_encoded = le_btts_eval.fit_transform(y_btts_train)
            
            # --- Penyesuaian untuk XGBoost H/D/A ---
            # Pastikan num_class sesuai dengan data H/D/A yang di-encode
            if 'H/D/A' in league_config and isinstance(league_config['H/D/A'][0], XGBClassifier):
                num_classes_actual = len(le_ftr_eval.classes_)
                # Set num_class pada model instance
                league_config['H/D/A'][0].set_params(num_class=num_classes_actual)
                print(f"   (Info: Menyesuaikan num_class XGBoost H/D/A menjadi {num_classes_actual})")

            print("\n--- Mengevaluasi Model (dari Konfigurasi Terbaik) pada Data Uji ---")
            
            # --- Model H/D/A ---
            model_hda_base, params_hda = league_config['H/D/A']
            model_hda = model_hda_base.set_params(**params_hda) # Terapkan parameter terbaik
            model_hda.fit(X_train_scaled, y_ftr_train_encoded)
            y_pred_hda = model_hda.predict(X_test_scaled)
            acc_hda = accuracy_score(le_ftr_eval.transform(y_ftr_test), y_pred_hda)
            print(f" -> Akurasi H/D/A ({type(model_hda).__name__}): {acc_hda:.2%}")
            
            # --- Model BTTS ---
            model_btts_base, params_btts = league_config['BTTS']
            model_btts = model_btts_base.set_params(**params_btts) # Terapkan parameter terbaik
            model_btts.fit(X_train_scaled, y_btts_train_encoded)
            y_pred_btts = model_btts.predict(X_test_scaled)
            acc_btts = accuracy_score(le_btts_eval.transform(y_btts_test), y_pred_btts)
            print(f" -> Akurasi BTTS ({type(model_btts).__name__}): {acc_btts:.2%}")

            # --- Model O/U 2.5 ---
            model_ou25_base, params_ou25 = league_config['O/U 2.5']
            model_ou25 = model_ou25_base.set_params(**params_ou25) # Terapkan parameter terbaik
            model_ou25.fit(X_train_scaled, y_ou_train_encoded)
            y_pred_ou25 = model_ou25.predict(X_test_scaled)
            acc_ou25 = accuracy_score(le_ou_eval.transform(y_ou_test), y_pred_ou25)
            print(f" -> Akurasi O/U 2.5 ({type(model_ou25).__name__}): {acc_ou25:.2%}")

            all_results.append({
                'Liga': pretty_name.upper(), 
                f'H/D/A ({type(model_hda).__name__})': acc_hda, 
                f'BTTS ({type(model_btts).__name__})': acc_btts, 
                f'O/U 2.5 ({type(model_ou25).__name__})': acc_ou25
            })

            print("\n--- Melatih Ulang Model pada Keseluruhan Data ---")
            scaler_final = StandardScaler()
            X_scaled_final = scaler_final.fit_transform(X)

            le_ftr_final, le_ou_final, le_btts_final = LabelEncoder(), LabelEncoder(), LabelEncoder()
            y_ftr_encoded_final = le_ftr_final.fit_transform(y_ftr)
            y_ou_encoded_final = le_ou_final.fit_transform(y_ou)
            y_btts_encoded_final = le_btts_final.fit_transform(y_btts)
            
            # --- Penyesuaian num_class untuk model FINAL XGBoost H/D/A ---
            if 'H/D/A' in league_config and isinstance(league_config['H/D/A'][0], XGBClassifier):
                num_classes_final = len(le_ftr_final.classes_)
                model_hda.set_params(num_class=num_classes_final)
                print(f"   (Info: Menyesuaikan num_class FINAL XGBoost H/D/A menjadi {num_classes_final})")

            # Latih ulang model final dengan parameter terbaik
            model_hda.fit(X_scaled_final, y_ftr_encoded_final)
            model_btts.fit(X_scaled_final, y_btts_encoded_final) 
            model_ou25.fit(X_scaled_final, y_ou_encoded_final)
            print("✅ Model final berhasil dilatih ulang.")

            joblib.dump(model_hda, os.path.join(league_model_dir, 'model_hda.pkl'))
            joblib.dump(model_btts, os.path.join(league_model_dir, 'model_btts.pkl'))
            joblib.dump(model_ou25, os.path.join(league_model_dir, 'model_ou25.pkl'))
            joblib.dump(scaler_final, os.path.join(league_model_dir, 'scaler.pkl'))
            joblib.dump(le_ftr_final, os.path.join(league_model_dir, 'le_ftr.pkl'))
            joblib.dump(le_ou_final, os.path.join(league_model_dir, 'le_ou.pkl'))
            joblib.dump(le_btts_final, os.path.join(league_model_dir, 'le_btts.pkl'))

            print(f"✨ Model final untuk {pretty_name.upper()} telah disimpan di '{league_model_dir}'.")

        except Exception as e:
            print(f"❌ GAGAL memproses {filename_with_ext}. Error: {e}")
            import traceback
            traceback.print_exc() # Cetak error detail
            continue

    if all_results:
        print(f"\n\n{'='*25} PERBANDINGAN AKURASI AKHIR (MODEL TERBAIK) {'='*25}")
        results_df = pd.DataFrame(all_results).set_index('Liga')
        
        # Mengambil semua kolom kecuali 'Liga' (yang sekarang jadi index)
        formatters = {col: '{:.2%}'.format for col in results_df.columns}
        
        print(results_df.to_string(formatters=formatters))

    print(f"\n\n✨✨✨ Proses Selesai! Semua model TERBAIK telah dievaluasi, dilatih ulang, dan disimpan.")


if __name__ == '__main__':
    train_and_evaluate_all_leagues()