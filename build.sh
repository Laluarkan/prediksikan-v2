#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Installing requirements..."
pip install -r requirements.txt

# ▼▼▼ TAMBAHKAN BLOK INI UNTUK MENGINSTAL GIT-LFS MANUAL ▼▼▼
echo "Updating package list and installing Git LFS..."
apt-get update
apt-get install -y git-lfs
# ▲▲▲ AKHIR TAMBAHAN ▲▲▲

echo "Setting up Git LFS..."
git lfs install
echo "Pulling LFS files (models)..."
git lfs pull

echo "Running migrations..."
python manage.py migrate

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Creating superuser..."
python manage.py createsuperuser --no-input || true