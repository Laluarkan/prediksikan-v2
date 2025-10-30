import pandas as pd

# ==============================
# 1️⃣ BACA DATASET
# ==============================
# Misal file kamu bernama 'data.csv'
df = pd.read_csv("dataset/dataset_Liga_Turki_1.csv")

# Pastikan kolom 'Date' dikenali sebagai datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# ==============================
# 2️⃣ HAPUS DATA DALAM RENTANG TANGGAL TERTENTU
# ==============================
# Contoh: menghapus semua data dari 2005-05-01 dan seterusnya
tanggal_batas = pd.to_datetime("2025-08")

# Simpan hanya data setelah tanggal tersebut
# (atau gunakan < jika kamu ingin buang yang sesudah)
df_filtered = df[df['Date'] < tanggal_batas]

# ==============================
# 3️⃣ SIMPAN HASIL FILTER KE FILE BARU (opsional)
# ==============================
df_filtered.to_csv("dataset_train/turki.csv", index=False)

print("Jumlah data sebelum:", len(df))
print("Jumlah data sesudah:", len(df_filtered))
print("Data berhasil difilter dan disimpan ke dataset_train/data_filtered.csv ✅")
