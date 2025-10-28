# predictions/migrations/XXXX_create_default_site.py

from django.db import migrations
from django.conf import settings

# 
# PENTING: Ganti 'prediksi-kan-1.onrender.com' 
# dengan URL domain Render Anda yang benar.
#
APP_DOMAIN = 'prediksi-kan-1.onrender.com'

def create_site(apps, schema_editor):
    """Membuat atau memperbarui entri Site untuk SITE_ID = 1."""
    Site = apps.get_model('sites', 'Site')
    
    Site.objects.update_or_create(
        id=settings.SITE_ID,
        defaults={
            'domain': APP_DOMAIN,
            'name': 'Prediksi-Kan' # Nama situs Anda
        }
    )

def remove_site(apps, schema_editor):
    """ (Dibiarkan kosong, tidak perlu rollback data ini) """
    pass

class Migration(migrations.Migration):

    dependencies = [
        # ▼▼▼ GANTI INI dengan nama file migrasi terakhir Anda dari Langkah 2 ▼▼▼
        ('predictions', '0005_create_default_site'), 
        # ▲▲▲ -------------------------------------------------------- ▲▲▲
        
        # Migrasi ini juga bergantung pada migrasi bawaan 'sites'
        ('sites', '0002_alter_domain_unique'), 
    ]

    operations = [
        migrations.RunPython(create_site, remove_site),
    ]