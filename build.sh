#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Installing requirements..."
pip install -r requirements.txt

echo "Running migrations..."
python manage.py migrate

echo "Creating/Updating Site ID 1..."
# Kita jalankan skrip Python kecil untuk ini
# Ganti 'prediksi-kan-1.onrender.com' JIKA NAMA DOMAIN ANDA BERBEDA
python manage.py shell <<EOF
from django.contrib.sites.models import Site
from django.conf import settings
Site.objects.update_or_create(
    id=settings.SITE_ID,
    defaults={
        'domain': 'prediksi-kan-1.onrender.com',
        'name': 'Prediksi-Kan'
    }
)
print('Site ID 1 updated successfully.')
EOF

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Creating superuser..."
# Perintah ini akan menggunakan env var (DJANGO_SUPERUSER_*)
python manage.py createsuperuser --no-input || true