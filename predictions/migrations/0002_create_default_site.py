# predictions/migrations/0002_create_default_site.py

from django.db import migrations
from django.conf import settings

# ▼▼▼ PENTING ▼▼▼
# Ganti 'prediksi-kan-1.onrender.com' dengan URL Render Anda yang benar
APP_DOMAIN = 'prediksi-kan-1.onrender.com' 
# ▲▲▲ --------- ▲▲▲

def create_site(apps, schema_editor):
    """
    Membuat atau memperbarui entri Site untuk SITE_ID = 1.
    """
    Site = apps.get_model('sites', 'Site')
    
    Site.objects.update_or_create(
        id=settings.SITE_ID, # (SITE_ID = 1 dari settings.py)
        defaults={
            'domain': APP_DOMAIN,
            'name': 'Prediksi-Kan' # Nama situs Anda
        }
    )

def remove_site(apps, schema_editor):
    """ Dibiarkan kosong, tidak perlu rollback data ini """
    pass

class Migration(migrations.Migration):

    dependencies = [
        # Bergantung pada migrasi pertama (0001) aplikasi 'predictions'
        ('predictions', '0001_initial'), 
        
        # Bergantung pada migrasi 'sites' bawaan Django
        ('sites', '0002_alter_domain_unique'), 
    ]

    operations = [
        # Menjalankan fungsi create_site saat migrate
        migrations.RunPython(create_site, remove_site),
    ]