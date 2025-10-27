import pandas as pd
import numpy as np
import warnings
import os
import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# --- Impor Model untuk Diuji ---
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier  # --- IMPORT BARU ---

# ==============================================================================
# KONFIGURASI
# ==============================================================================
DATASET_DIR = 'dataset'
# Salin fitur yang sama persis dari skrip train.py Anda
FEATURE_COLUMNS = [
    'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5',
    'HomeTeamElo', 'AwayTeamElo', 'EloDifference',
    'Home_AvgGoalsScored', 'Home_AvgGoalsConceded', 'Home_Wins', 'Home_Draws', 'Home_Losses',
    'Away_AvgGoalsScored', 'Away_AvgGoalsConceded', 'Away_Wins', 'Away_Draws', 'Away_Losses',
    'HTH_HomeWins', 'HTH_AwayWins', 'HTH_Draws',
    'HTH_AvgHomeGoals', 'HTH_AvgAwayGoals'
]

# Konfigurasi GridSearchCV
# Set CV=3 untuk kecepatan. Naikkan ke 5 untuk keandalan yang lebih baik (tapi lebih lama)
CV_FOLDS = 5 
# Set n_jobs=-1 untuk menggunakan semua core CPU
N_JOBS = -1

warnings.filterwarnings('ignore')

# ==============================================================================
# UTILITAS (Dibutuhkan untuk penamaan)
# ==============================================================================
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
        'ligue1': 'Ligue 1'
    }
    key = name.lower().replace('_', '')
    if key in special_cases:
        return special_cases[key]
    return name.replace('_', ' ').title()

# ==============================================================================
# DEFINISI MODEL & PARAMETER GRID UNTUK TUNING
# ==============================================================================
def get_models_and_grids(target_name, label_encoder):
    """
    Mendefinisikan semua model dan parameter grid yang akan diuji.
    Dibutuhkan `target_name` dan `label_encoder` untuk menangani
    kasus khusus multiclass XGBoost.
    """
    
    # Grid parameter yang "luas" untuk dicoba
    grids = {
        'LogisticRegression': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'RandomForest': {
            'n_estimators': [100, 200], # Lebih sedikit dari 300 untuk tuning lebih cepat
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 4]
        },
        'GaussianNB': {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        # --- GRID BARU UNTUK NEURAL NETWORK ---
        # Catatan: Ini akan SANGAT lambat untuk dituning
        'MLP': {
            # 'hidden_layer_sizes' adalah arsitektur NN.
            # (100,) = 1 layer dengan 100 neuron
            # (50, 50) = 2 layer dengan masing-masing 50 neuron
            'hidden_layer_sizes': [(100,), (50, 50)],
            'activation': ['relu'], # 'relu' hampir selalu jadi pilihan terbaik
            'alpha': [0.001, 0.01], # 'alpha' adalah regularisasi L2 (mencegah overfitting)
            'solver': ['adam'] # 'adam' adalah optimizer default yang sangat baik
        }
    }

    # Model dasar
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'KNN': KNeighborsClassifier(n_jobs=N_JOBS),
        'SVM': SVC(random_state=42, probability=True),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=N_JOBS),
        'GaussianNB': GaussianNB(),
        # --- MODEL BARU UNTUK NEURAL NETWORK ---
        # max_iter=1000 agar memberi NN cukup waktu untuk konvergensi
        'MLP': MLPClassifier(random_state=42, max_iter=1000)
    }
    
    # Penanganan khusus untuk XGBoost (Multiclass vs Binary)
    if target_name == 'H/D/A':
        # Ini adalah klasifikasi multiclass
        num_classes = len(label_encoder.classes_)
        models['XGBoost'] = XGBClassifier(
            random_state=42, n_jobs=N_JOBS, 
            objective='multi:softmax', num_class=num_classes,
            use_label_encoder=False, eval_metric='mlogloss'
        )
    else:
        # Ini adalah klasifikasi biner (BTTS, O/U)
        models['XGBoost'] = XGBClassifier(
            random_state=42, n_jobs=N_JOBS,
            use_label_encoder=False, eval_metric='logloss'
        )
        
    # Mengembalikan tuple (model, grid)
    return {name: (models[name], grids[name]) for name in models}

# ==============================================================================
# FUNGSI EVALUASI UTAMA
# ==============================================================================
def find_best_models():
    print("🚀 Memulai pencarian model terbaik...")
    print(f"Menggunakan CV={CV_FOLDS} folds dan N_JOBS={N_JOBS}.")
    print("⚠️  PERINGATAN: Menambahkan MLPClassifier (Neural Network) akan membuat proses ini JAUH LEBIH LAMA.\n")
    
    dataset_paths = glob.glob(os.path.join(DATASET_DIR, '*.csv'))
    all_results = []

    if not dataset_paths:
        print(f"❌ ERROR: Tidak ada file dataset .csv yang ditemukan di folder '{DATASET_DIR}'.")
        return

    for path in dataset_paths:
        try:
            filename_with_ext = os.path.basename(path)
            filename_base = os.path.splitext(filename_with_ext)[0]
            pretty_name = pretty_league_name(filename_base)
            
            print(f"\n{'='*25} PROCESSING LEAGUE: {pretty_name.upper()} {'='*25}")

            data = pd.read_csv(path)

            # Buat target variables
            data['TotalGoals'] = data['FTHG'] + data['FTAG']
            data['OverUnder2.5'] = np.where(data['TotalGoals'] > 2.5, 'Over', 'Under')
            data['BTTS'] = np.where((data['FTHG'] > 0) & (data['FTAG'] > 0), 'Yes', 'No')

            if not all(col in data.columns for col in FEATURE_COLUMNS):
                print(f"⚠️  WARNING: Fitur tidak lengkap di {filename_with_ext}. Melewati.")
                continue

            X = data[FEATURE_COLUMNS]
            
            # Buat dictionary untuk semua target
            y_targets = {
                'H/D/A': data['FTR'],
                'O/U 2.5': data['OverUnder2.5'],
                'BTTS': data['BTTS']
            }

            if X.isnull().sum().sum() > 0:
                X = X.fillna(X.mean())

            # Terapkan WINDOW offset (sama seperti di train.py)
            WINDOW = 5
            X = X.iloc[WINDOW:]
            for target_name in y_targets:
                y_targets[target_name] = y_targets[target_name].iloc[WINDOW:]
            
            if X.empty:
                print(f"⚠️  WARNING: Tidak ada data tersisa setelah di-windowing. Melewati.")
                continue

            # Pisahkan data HANYA SEKALI
            # Kita hanya perlu memisahkan X, y akan mengikuti indeks yang sama
            X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            print(f"✅ Data {pretty_name} dimuat dan diproses (Train: {len(X_train)}, Test: {len(X_test)})")

            # --- Loop untuk setiap target prediksi ---
            for target_name, y_full in y_targets.items():
                print(f"\n--- Menganalisis Target: {target_name} ---")
                
                # Pisahkan y sesuai dengan X
                y_train = y_full.loc[X_train.index]
                y_test = y_full.loc[X_test.index]
                
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(y_train)
                y_test_encoded = le.transform(y_test)
                
                # Dapatkan semua model dan grid untuk diuji
                models_to_tune = get_models_and_grids(target_name, le)
                
                # --- Loop untuk setiap algoritma ---
                for model_name, (model, grid) in models_to_tune.items():
                    print(f"  Tuning {model_name}...")
                    try:
                        search = GridSearchCV(
                            estimator=model,
                            param_grid=grid,
                            cv=CV_FOLDS,
                            scoring='accuracy',
                            n_jobs=N_JOBS,
                            verbose=0 # Set ke 1 jika ingin lihat progres detail
                        )
                        
                        # Latih pada data training
                        search.fit(X_train_scaled, y_train_encoded)
                        
                        best_model = search.best_estimator_
                        
                        # Evaluasi pada data test
                        y_pred_test = best_model.predict(X_test_scaled)
                        test_accuracy = accuracy_score(y_test_encoded, y_pred_test)
                        
                        all_results.append({
                            'Liga': pretty_name.upper(),
                            'Target': target_name,
                            'Algorithm': model_name,
                            'Test Accuracy': test_accuracy,
                            'Best CV Score': search.best_score_,
                            'Best Params': search.best_params_
                        })
                        
                        print(f"    -> Selesai. Akurasi Test: {test_accuracy:.2%}")
                        
                    except Exception as e:
                        print(f"    -> ❌ GAGAL Tuning {model_name}: {e}")
                        all_results.append({
                            'Liga': pretty_name.upper(),
                            'Target': target_name,
                            'Algorithm': model_name,
                            'Test Accuracy': 0.0,
                            'Best CV Score': 0.0,
                            'Best Params': str(e)
                        })

        except Exception as e:
            print(f"❌ GAGAL memproses {filename_with_ext}. Error: {e}")
            continue

    # --- Setelah semua selesai, tampilkan hasilnya ---
    if not all_results:
        print("Tidak ada hasil yang didapat.")
        return

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by=['Liga', 'Target', 'Test Accuracy'], ascending=[True, True, False])

    print(f"\n\n{'='*35} HASIL LENGKAP EVALUASI MODEL {'='*35}")
    with pd.option_context('display.max_rows', None, 'display.width', 1000):
        print(results_df.to_string(
            formatters={'Test Accuracy': '{:.2%}'.format, 'Best CV Score': '{:.2%}'.format},
            columns=['Liga', 'Target', 'Algorithm', 'Test Accuracy', 'Best CV Score', 'Best Params']
        ))

    print(f"\n\n{'='*30} MODEL TERBAIK PER KATEGORI (BERDASARKAN AKURASI TEST) {'='*30}")
    # Cari baris dengan 'Test Accuracy' tertinggi untuk setiap grup 'Liga' dan 'Target'
    best_models_summary = results_df.loc[results_df.groupby(['Liga', 'Target'])['Test Accuracy'].idxmax()]
    
    with pd.option_context('display.max_rows', None, 'display.width', 1000):
        print(best_models_summary.to_string(
            formatters={'Test Accuracy': '{:.2%}'.format, 'Best CV Score': '{:.2%}'.format},
            columns=['Liga', 'Target', 'Algorithm', 'Test Accuracy', 'Best Params']
        ))
        
    print("\n\n✨✨✨ Proses Pencarian Selesai! ✨✨✨")


if __name__ == '__main__':
    find_best_models()