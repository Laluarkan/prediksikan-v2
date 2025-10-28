# predictions/migrations/0003_create_google_social_app.py

from django.db import migrations
from django.conf import settings
import os # Import os

def create_social_app(apps, schema_editor):
    """
    Membuat entri SocialApplication untuk Google menggunakan
    Environment Variables.
    """
    Site = apps.get_model('sites', 'Site')
    SocialApp = apps.get_model('socialaccount', 'SocialApp')

    # 1. Dapatkan Kunci dari Environment Variables
    # (Pastikan ini ada di Env Var Render Anda)
    CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
    SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')

    if not CLIENT_ID or not SECRET:
        # Jangan lakukan apa-apa jika kunci tidak diatur
        print("WARNING: GOOGLE_CLIENT_ID atau SECRET tidak diatur di Environment.")
        return

    # 2. Dapatkan Site (yang sudah dibuat oleh migrasi 0002)
    try:
        site = Site.objects.get(id=settings.SITE_ID)
    except Site.DoesNotExist:
        print(f"ERROR: Site dengan ID={settings.SITE_ID} tidak ditemukan. Tidak dapat membuat SocialApp.")
        return

    # 3. Buat atau Perbarui SocialApp
    app, created = SocialApp.objects.update_or_create(
        provider='google',
        defaults={
            'name': 'Google Login',
            'client_id': CLIENT_ID,
            'secret': SECRET,
            'key': ''
        }
    )

    # 4. Hubungkan ke Situs Anda
    app.sites.set([site])
    app.save()

    if created:
        print('Aplikasi Sosial Google berhasil dibuat di database.')
    else:
        print('Aplikasi Sosial Google berhasil diperbarui di database.')

class Migration(migrations.Migration):

    dependencies = [
        # Gantungkan pada migrasi 'create_default_site'
        ('predictions', '0002_create_default_site'), 
        # Gantungkan pada migrasi 'socialaccount'
        ('socialaccount', '0001_initial'), 
    ]

    operations = [
        migrations.RunPython(create_social_app),
    ]