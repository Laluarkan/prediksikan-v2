#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Installing requirements..."
pip install -r requirements.txt

echo "Running migrations..."
python manage.py migrate

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Creating superuser..."
# Perintah ini akan menggunakan env var (DJANGO_SUPERUSER_*)
python manage.py createsuperuser --no-input || true