# Gunakan base image Python yang spesifik
FROM python:3.11.5-slim

# Set variabel environment
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 1. Instal dependensi sistem (termasuk git dan git-lfs)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    git-lfs \
 && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Instal dependensi Python (ini di-cache oleh Docker)
# (Hanya salin requirements.txt dulu agar langkah ini di-cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Salin seluruh kode proyek Anda
COPY . .

# 5. Tarik file LFS (model .pkl Anda)
# Ini sekarang akan berfungsi karena .git Anda sudah di-copy
RUN git lfs install
RUN git lfs pull

# 6. Kumpulkan file statis (CSS, JS, Ikon)
# Ini akan mengumpulkan file ke /app/staticfiles (sesuai settings.py Anda)
RUN python manage.py collectstatic --noinput

# 7. Port yang akan diekspos (Render akan mendeteksi ini)
EXPOSE 8000

# 8. Perintah untuk menjalankan server (bukan build.sh)
# Kita akan override ini di Render, tapi ini adalah default yang baik
CMD ["gunicorn", "football_project.wsgi"]