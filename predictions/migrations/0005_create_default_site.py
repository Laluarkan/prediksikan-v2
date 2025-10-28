from django.db import migrations
from django.conf import settings

# ======================================
# PENTING: Ganti domain di bawah ini
# sesuai dengan URL Render kamu
# ======================================
APP_DOMAIN = 'prediksi-kan-1.onrender.com'

def create_site(apps, schema_editor):
    """Membuat atau memperbarui entri Site untuk SITE_ID = 1."""
    Site = apps.get_model('sites', 'Site')

    Site.objects.update_or_create(
        id=getattr(settings, 'SITE_ID', 1),
        defaults={
            'domain': APP_DOMAIN,
            'name': 'Prediksi-Kan'
        }
    )

def remove_site(apps, schema_editor):
    """Dibiarkan kosong, tidak perlu rollback data ini."""
    pass

class Migration(migrations.Migration):

    dependencies = [
        ('predictions', '0004_predictionhistory_btts_chosen_and_more'),
        ('sites', '0002_alter_domain_unique'),
    ]

    operations = [
        migrations.RunPython(create_site, remove_site),
    ]
