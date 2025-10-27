import pandas as pd

# --- KONFIGURASI ---

# 1. Masukkan nama 3 file CSV Anda di sini
file_list = [
    "dataset/T1 (1).csv",
    "dataset/T1 (2).csv",
    "dataset/T1.csv"
]

# 2. Nama file untuk menyimpan hasil gabungan
# (Saya ganti namanya agar jelas ini file yang sudah terurut)
output_file = "dataset/data_gabungan.csv"

# 3. Kolom yang ingin Anda ambil dari file asli
# Kita perlu 'Date' dan 'Time' untuk menggabungkannya
columns_to_read = [
    'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
    'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5'
]

# 4. Urutan kolom final di file output
final_columns_order = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
    'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5'
]

# --- AKHIR KONFIGURASI ---


def proses_data():
    """
    Fungsi utama untuk membaca, memformat, menggabungkan, dan mengurutkan file CSV.
    """
    # List untuk menampung data dari setiap file
    all_data_frames = []

    print("Mulai memproses file...")

    for file in file_list:
        try:
            # Langkah 1: Baca file CSV dan hanya ambil kolom yang kita perlukan
            df = pd.read_csv(file, usecols=columns_to_read)

            # Langkah 2: Hapus baris kosong (sering ada di akhir file data bola)
            df = df.dropna(subset=['Date', 'HomeTeam'])

            # Langkah 3: Gabungkan kolom 'Date' dan 'Time' menjadi satu string
            # Asumsi format input Date: dd/mm/YYYY dan Time: HH:MM
            datetime_str = df['Date'] + ' ' + df['Time']

            # Langkah 4: Konversi string gabungan menjadi objek datetime
            # pd.to_datetime akan otomatis menangani format 'dd/mm/YYYY HH:MM'
            # dengan 'dayfirst=True'
            datetime_obj = pd.to_datetime(datetime_str, dayfirst=True)

            # Langkah 5: Format ulang kolom 'Date' sesuai format yang Anda minta
            # (YYYY-MM-DD HH:MM:SS). Ini akan menimpa kolom 'Date' lama.
            df['Date'] = datetime_obj.dt.strftime('%Y-%m-%d %H:%M:%S')

            # Langkah 6: Pilih hanya kolom akhir yang diinginkan
            df_final = df[final_columns_order]

            # Langkah 7: Tambahkan data yang sudah bersih ke list
            all_data_frames.append(df_final)
            print(f"✅ Berhasil memproses: {file}")

        except FileNotFoundError:
            print(f"⚠️ PERINGATAN: File {file} tidak ditemukan. File dilewati.")
        except KeyError as e:
            print(f"⚠️ PERINGATAN: Kolom {e} tidak ada di {file}. File dilewati.")
        except Exception as e:
            print(f"❌ ERROR saat memproses {file}: {e}. File dilewati.")

    # Langkah 8: Gabungkan semua data dari list menjadi satu DataFrame
    if all_data_frames:
        print("\nMenggabungkan semua data...")
        combined_df = pd.concat(all_data_frames, ignore_index=True)

        # Langkah 9: Urutkan DataFrame berdasarkan kolom 'Date' (dari terlama ke terbaru)
        print("Mengurutkan data berdasarkan tanggal...")
        combined_df = combined_df.sort_values(by='Date', ascending=True)

        # Langkah 10: Simpan ke file CSV baru
        combined_df.to_csv(output_file, index=False)
        print(f"\n🎉 Berhasil! Data gabungan telah diurutkan dan disimpan di: {output_file}")
        print(f"Total {len(combined_df)} baris data diproses.")
    else:
        print("\nTidak ada data yang berhasil diproses. File output tidak dibuat.")


# Menjalankan skrip
if __name__ == "__main__":
    # Pastikan Anda sudah menginstal pandas: pip install pandas
    proses_data()