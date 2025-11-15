import pandas as pd
import numpy as np
import warnings
import os
import joblib
import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# [--- MULAI PERUBAHAN: Menyembunyikan log TensorFlow yang 'berisik' ---]
# Set log level TensorFlow sebelum mengimpornya
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 2 = Hanya tampilkan Error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
# [--- AKHIR PERUBAHAN ---]

# ==============================================================================
# KONFIGURASI
# ==============================================================================
DATASET_DIR = 'dataset_train/liga_inggris.csv' # Tetap bisa diganti ke folder 'dataset_train'
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
# UTILITAS 
# ==============================================================================
def pretty_league_name(file_name):
    """
    Mengubah nama file dataset menjadi nama liga yang rapi untuk tampilan.
    """
    name = file_name.replace('dataset_', '').replace('.csv', '')
    name = name.replace('_1', '')
    special_cases = {
        'eredivisie': 'Eredivisie', 'seriea': 'Serie A', 'laliga': 'La Liga',
        'premierleague': 'Premier League', 'bundesliga': 'Bundesliga', 'ligue1': 'Ligue 1'
    }
    key = name.lower().replace('_', '')
    return special_cases.get(key, name.replace('_', ' ').title())

# ==============================================================================
# FUNGSI HELPERS UNTUK MLP (Sama seperti sebelumnya)
# ==============================================================================
def create_mlp_model(input_dim, output_dim, output_activation):
    """
    Membuat model MLP dengan arsitektur standar + Dropout.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation=output_activation)
    ])
    return model

# Tentukan callback EarlyStopping sekali saja
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,            # <-- Tetap 1 agar menampilkan pesan "Early stopping"
    restore_best_weights=True
)

# ==============================================================================
# FUNGSI UTAMA UNTUK MELATIH DAN MENGGEVALUASI
# ==============================================================================
def train_and_evaluate_all_leagues():
    print("🚀 Memulai proses training dan evaluasi...")
    
    all_results = []
    
    # [--- MULAI PERUBAHAN: Logika untuk mendeteksi file atau folder ---]
    if os.path.isdir(DATASET_DIR):
        print(f"Mencari semua file .csv di folder: {DATASET_DIR}")
        dataset_paths = glob.glob(os.path.join(DATASET_DIR, '*.csv'))
    elif os.path.isfile(DATASET_DIR):
        print(f"Hanya memproses satu file: {DATASET_DIR}")
        dataset_paths = [DATASET_DIR]
    else:
        print(f"❌ ERROR: Path tidak ditemukan: {DATASET_DIR}")
        dataset_paths = []
    # [--- AKHIR PERUBAHAN ---]

    if not dataset_paths:
        print(f"❌ ERROR: Tidak ada file dataset .csv yang ditemukan.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    for path in dataset_paths:
        try:
            filename_with_ext = os.path.basename(path)
            filename_base = os.path.splitext(filename_with_ext)[0]
            pretty_name = pretty_league_name(filename_base)
            league_folder_name = pretty_name.lower().replace(' ', '_')
            
            # --- Perbaikan Typo ---
            print(f"\n{'='*20} PROCESSING LEAGUE: {pretty_name.upper()} {'='*20}")

            league_model_dir = os.path.join(MODEL_DIR, league_folder_name)
            os.makedirs(league_model_dir, exist_ok=True)

            data = pd.read_csv(path)
            print(f"✅ Berhasil memuat file: {filename_with_ext}")

            # --- Persiapan Fitur & Target ---
            data['TotalGoals'] = data['FTHG'] + data['FTAG']
            data['OverUnder2.5'] = np.where(data['TotalGoals'] > 2.5, 'Over', 'Under')
            data['BTTS'] = np.where((data['FTHG'] > 0) & (data['FTAG'] > 0), 'Yes', 'No')

            if not all(col in data.columns for col in FEATURE_COLUMNS):
                print(f"⚠️  WARNING: Tidak semua fitur ditemukan di {filename_with_ext}. Melewati liga ini.")
                continue

            X = data[FEATURE_COLUMNS]
            n_features = X.shape[1]
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
            
            # --- SPLIT DATA TRAIN/VALIDASI/TEST ---
            X_train_full, X_test, y_ftr_train_full, y_ftr_test = train_test_split(X, y_ftr, test_size=0.2, shuffle=False)
            _, _, y_ou_train_full, y_ou_test = train_test_split(X, y_ou, test_size=0.2, shuffle=False)
            _, _, y_btts_train_full, y_btts_test = train_test_split(X, y_btts, test_size=0.2, shuffle=False)

            X_train, X_val, y_ftr_train, y_ftr_val = train_test_split(X_train_full, y_ftr_train_full, test_size=0.25, shuffle=False)
            _, _, y_ou_train, y_ou_val = train_test_split(X_train_full, y_ou_train_full, test_size=0.25, shuffle=False)
            _, _, y_btts_train, y_btts_val = train_test_split(X_train_full, y_btts_train_full, test_size=0.25, shuffle=False)
            
            print(f"✅ Data berhasil dipisah: {len(X_train)} latih, {len(X_val)} validasi, {len(X_test)} uji.")

            # Scaling
            scaler_eval = StandardScaler()
            X_train_scaled = scaler_eval.fit_transform(X_train)
            X_val_scaled = scaler_eval.transform(X_val)
            X_test_scaled = scaler_eval.transform(X_test)

            # Label Encoding
            le_ftr_eval, le_ou_eval, le_btts_eval = LabelEncoder(), LabelEncoder(), LabelEncoder()
            y_ftr_train_encoded = le_ftr_eval.fit_transform(y_ftr_train)
            y_ftr_val_encoded = le_ftr_eval.transform(y_ftr_val)
            y_ftr_test_encoded = le_ftr_eval.transform(y_ftr_test)
            y_ou_train_encoded = le_ou_eval.fit_transform(y_ou_train)
            y_ou_val_encoded = le_ou_eval.transform(y_ou_val)
            y_ou_test_encoded = le_ou_eval.transform(y_ou_test)
            y_btts_train_encoded = le_btts_eval.fit_transform(y_btts_train)
            y_btts_val_encoded = le_btts_eval.transform(y_btts_val)
            y_btts_test_encoded = le_btts_eval.transform(y_btts_test)
            
            # One-Hot Encoding
            y_ftr_train_onehot = to_categorical(y_ftr_train_encoded, num_classes=3)
            y_ftr_val_onehot = to_categorical(y_ftr_val_encoded, num_classes=3)
            # --- AKHIR SPLIT DATA ---

            print("\n--- Mengevaluasi Model MLP pada Data Uji (dengan Early Stopping) ---")
            
            # --- Model H/D/A -> MLP ---
            print("--- Melatih Model H/D/A (MLP) ---")
            model_hda = create_mlp_model(n_features, 3, 'softmax')
            model_hda.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            history_hda = model_hda.fit(X_train_scaled, y_ftr_train_onehot,
                                        epochs=150,
                                        validation_data=(X_val_scaled, y_ftr_val_onehot), 
                                        callbacks=[early_stopping_callback],
                                        # [--- PERUBAHAN: Tampilkan log epoch ---]
                                        verbose=2) 
            
            best_epochs_hda = np.argmin(history_hda.history['val_loss']) + 1
            print(f"   - H/D/A (MLP) validation complete. Epochs terbaik: {best_epochs_hda}")
            
            # [--- PERUBAHAN: Sembunyikan progress bar predict ---]
            y_pred_hda_proba = model_hda.predict(X_test_scaled, verbose=0)
            y_pred_hda = np.argmax(y_pred_hda_proba, axis=1) 
            acc_hda = accuracy_score(y_ftr_test_encoded, y_pred_hda)
            print(f"   - Akurasi H/D/A (MLP): {acc_hda:.2%}")
            
            # --- Model BTTS -> MLP ---
            print("--- Melatih Model BTTS (MLP) ---")
            model_btts = create_mlp_model(n_features, 1, 'sigmoid')
            model_btts.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            history_btts = model_btts.fit(X_train_scaled, y_btts_train_encoded,
                                          epochs=150,
                                          validation_data=(X_val_scaled, y_btts_val_encoded),
                                          callbacks=[early_stopping_callback],
                                          # [--- PERUBAHAN: Tampilkan log epoch ---]
                                          verbose=2)
            
            best_epochs_btts = np.argmin(history_btts.history['val_loss']) + 1
            print(f"   - BTTS (MLP) validation complete. Epochs terbaik: {best_epochs_btts}")
            
            # [--- PERUBAHAN: Sembunyikan progress bar predict ---]
            y_pred_btts_proba = model_btts.predict(X_test_scaled, verbose=0)
            y_pred_btts = (y_pred_btts_proba > 0.5).astype(int)
            acc_btts = accuracy_score(y_btts_test_encoded, y_pred_btts)
            print(f"   - Akurasi BTTS (MLP): {acc_btts:.2%}")

            # --- Model O/U 2.5 -> MLP ---
            print("--- Melatih Model O/U 2.5 (MLP) ---")
            model_ou25 = create_mlp_model(n_features, 1, 'sigmoid')
            model_ou25.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            history_ou25 = model_ou25.fit(X_train_scaled, y_ou_train_encoded,
                                          epochs=150,
                                          validation_data=(X_val_scaled, y_ou_val_encoded),
                                          callbacks=[early_stopping_callback],
                                          # [--- PERUBAHAN: Tampilkan log epoch ---]
                                          verbose=2)
            
            best_epochs_ou25 = np.argmin(history_ou25.history['val_loss']) + 1
            print(f"   - O/U 2.5 (MLP) validation complete. Epochs terbaik: {best_epochs_ou25}")
            
            # [--- PERUBAHAN: Sembunyikan progress bar predict ---]
            y_pred_ou25_proba = model_ou25.predict(X_test_scaled, verbose=0)
            y_pred_ou25 = (y_pred_ou25_proba > 0.5).astype(int)
            acc_ou25 = accuracy_score(y_ou_test_encoded, y_pred_ou25)
            print(f"   - Akurasi O/U 2.5 (MLP): {acc_ou25:.2%}")
            
            # --- Perbaikan Typo ---
            all_results.append({'Liga': pretty_name.upper(), 'H/D/A (MLP)': acc_hda, 'BTTS (MLP)': acc_btts, 'O/U 2.5 (MLP)': acc_ou25})

            # --- BLOK RETRAINING FINAL ---
            print("\n--- Melatih Ulang Model MLP pada Keseluruhan Data (100%) ---")
            scaler_final = StandardScaler()
            X_scaled_final = scaler_final.fit_transform(X)

            le_ftr_final, le_ou_final, le_btts_final = LabelEncoder(), LabelEncoder(), LabelEncoder()
            y_ftr_encoded_final = le_ftr_final.fit_transform(y_ftr)
            y_ou_encoded_final = le_ou_final.fit_transform(y_ou)
            y_btts_encoded_final = le_btts_final.fit_transform(y_btts)
            
            y_ftr_final_onehot = to_categorical(y_ftr_encoded_final, num_classes=3)
            
            print(f"   - Melatih ulang H/D/A (MLP) untuk {best_epochs_hda} epochs...")
            model_hda_final = create_mlp_model(n_features, 3, 'softmax')
            model_hda_final.compile(loss='categorical_crossentropy', optimizer='adam')
            model_hda_final.fit(X_scaled_final, y_ftr_final_onehot, epochs=best_epochs_hda, verbose=0)
            
            print(f"   - Melatih ulang BTTS (MLP) untuk {best_epochs_btts} epochs...")
            model_btts_final = create_mlp_model(n_features, 1, 'sigmoid')
            model_btts_final.compile(loss='binary_crossentropy', optimizer='adam')
            model_btts_final.fit(X_scaled_final, y_btts_encoded_final, epochs=best_epochs_btts, verbose=0)
            
            print(f"   - Melatih ulang O/U 2.5 (MLP) untuk {best_epochs_ou25} epochs...")
            model_ou25_final = create_mlp_model(n_features, 1, 'sigmoid')
            model_ou25_final.compile(loss='binary_crossentropy', optimizer='adam')
            model_ou25_final.fit(X_scaled_final, y_ou_encoded_final, epochs=best_epochs_ou25, verbose=0)
            
            print("✅ Model MLP final berhasil dilatih ulang.")

            model_hda_final.save(os.path.join(league_model_dir, 'model_hda.h5'))
            model_btts_final.save(os.path.join(league_model_dir, 'model_btts.h5'))
            model_ou25_final.save(os.path.join(league_model_dir, 'model_ou25.h5'))
            
            joblib.dump(scaler_final, os.path.join(league_model_dir, 'scaler.pkl'))
            joblib.dump(le_ftr_final, os.path.join(league_model_dir, 'le_ftr.pkl'))
            joblib.dump(le_ou_final, os.path.join(league_model_dir, 'le_ou.pkl'))
            joblib.dump(le_btts_final, os.path.join(league_model_dir, 'le_btts.pkl'))

            # --- Perbaikan Typo ---
            print(f"✨ Model final untuk {pretty_name.upper()} telah disimpan di '{league_model_dir}'.")

        except Exception as e:
            print(f"❌ GAGAL memproses {filename_with_ext}. Error: {e}")
            continue

    # [--- MULAI PERUBAHAN: Hanya cetak tabel jika memproses > 1 file ---]
    if all_results and len(all_results) > 1:
        print(f"\n\n{'='*25} PERBANDINGAN AKURASI AKHIR {'='*25}")
        results_df = pd.DataFrame(all_results).set_index('Liga')
        
        print(results_df.to_string(formatters={
            'H/D/A (MLP)': '{:.2%}'.format,
            'BTTS (MLP)': '{:.2%}'.format,
            'O/U 2.5 (MLP)': '{:.2%}'.format
        }))
    # [--- AKHIR PERUBAHAN ---]

    print(f"\n\n✨✨✨ Proses Selesai!")


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    train_and_evaluate_all_leagues()