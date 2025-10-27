#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Installing requirements..."
pip install -r requirements.txt

echo "Installing Git LFS..."
git lfs install
echo "Pulling LFS files (models)..."
git lfs pull

echo "Running migrations..."
python manage.py migrate

echo "Collecting static files..."
python manage.py collectstatic --noinput

# ▼▼▼ TAMBAHKAN BARIS INI ▼▼▼
echo "Creating superuser..."
# Perintah ini akan menggunakan env var (DJANGO_SUPERUSER_*)
# '--no-input' berarti tidak interaktif
# '|| true' berarti build tidak akan gagal jika superuser sudah ada
python manage.py createsuperuser --no-input || true
# ▲▲▲ AKHIR TAMBAHAN ▲▲▲