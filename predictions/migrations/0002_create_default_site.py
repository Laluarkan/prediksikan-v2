# predictions/migrations/0002_create_default_site.py
from django.db import migrations
from django.conf import settings

APP_DOMAIN = 'prediksi-kan-1.onrender.com' # Ganti jika nama Render Anda berbeda

def create_site(apps, schema_editor):
    Site = apps.get_model('sites', 'Site')
    Site.objects.update_or_create(
        id=settings.SITE_ID, # (SITE_ID = 1)
        defaults={'domain': APP_DOMAIN, 'name': 'Prediksi-Kan'}
    )

class Migration(migrations.Migration):
    dependencies = [
        ('predictions', '0001_initial'), # Bergantung pada migrasi pertama Anda
        ('sites', '0002_alter_domain_unique'), # Bergantung pada migrasi 'sites'
    ]
    operations = [
        migrations.RunPython(create_site),
    ]