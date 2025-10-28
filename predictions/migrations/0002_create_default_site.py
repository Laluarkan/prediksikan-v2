# predictions/migrations/0002_create_default_site.py

from django.db import migrations
from django.conf import settings

# Ganti 'prediksi-kan-1.onrender.com' dengan URL Render Anda
APP_DOMAIN = 'prediksi-kan-1.onrender.com' 

def create_site(apps, schema_editor):
    Site = apps.get_model('sites', 'Site')
    # Ini akan membuat ATAU memperbarui baris id=1
    Site.objects.update_or_create(
        id=settings.SITE_ID, # (SITE_ID = 1)
        defaults={
            'domain': APP_DOMAIN,
            'name': 'Prediksi-Kan'
        }
    )

class Migration(migrations.Migration):

    dependencies = [
        # Gantungkan pada migrasi pertama (0001) DAN migrasi 'sites'
        ('predictions', '0001_initial'), 
        ('sites', '0002_alter_domain_unique'), 
    ]

    operations = [
        migrations.RunPython(create_site),
    ]