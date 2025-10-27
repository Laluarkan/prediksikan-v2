# 1. Gunakan base image Python yang stabil
FROM python:3.11.5-slim

# Set variabel environment
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 2. Instal dependensi sistem (HANYA git, TIDAK PERLU git-lfs)
RUN apt-get update && apt-get install -y \
    git \
 && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Kloning repositori Anda
# (Karena LFS sudah dihapus, ini akan mengkloning file .pkl besar secara langsung)
RUN git clone https://github.com/laluarkan21/prediksi-kan-1-django.git .

# 5. HAPUS SEMUA PERINTAH LFS
# RUN git lfs install  <-- DIHAPUS
# RUN git lfs pull   <-- DIHAPUS

# 6. Instal dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# 7. Kumpulkan file statis
RUN python manage.py collectstatic --noinput

# 8. Port yang akan diekspos
EXPOSE 8000

# 9. Perintah default untuk menjalankan server
# (Kita akan override ini di Render untuk menambahkan migrasi)
CMD ["gunicorn", "football_project.wsgi", "--bind", "0.0.0.0:8000"]