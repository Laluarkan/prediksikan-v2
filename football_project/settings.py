# football_project/settings.py

import os
from pathlib import Path
import dj_database_url
from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_URL = 'static/'
load_dotenv(os.path.join(BASE_DIR, '.env'))
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]

# --- 1. Gantikan Konfigurasi Flask Anda ---
SECRET_KEY = os.environ.get('FLASK_SECRET_KEY') # Ambil dari environment
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL') # Ambil dari environment

# Konfigurasi dari Flask app Anda
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FEATURE_COLUMNS = [
    'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5',
    'HomeTeamElo', 'AwayTeamElo', 'EloDifference',
    'Home_AvgGoalsScored', 'Home_AvgGoalsConceded', 'Home_Wins', 'Home_Draws', 'Home_Losses',
    'Away_AvgGoalsScored', 'Away_AvgGoalsConceded', 'Away_Wins', 'Away_Draws', 'Away_Losses',
    'HTH_HomeWins', 'HTH_AwayWins', 'HTH_Draws',
    'HTH_AvgHomeGoals', 'HTH_AvgAwayGoals'
]
ALL_LEAGUES = "All Leagues"
INITIAL_ELO = 1500
# ---------------------------------------------

DEBUG = os.environ.get('DEBUG', 'False') == 'True'

ALLOWED_HOSTS = [
    'prediksi-kan-1.onrender.com',  # Mengizinkan subdomain Render Anda
    '127.0.0.1',      # Tetap izinkan localhost
    'localhost', 
    '.onrender.com'
] # Sesuaikan untuk produksi (Render akan mengaturnya)
CSRF_TRUSTED_ORIGINS = [
    'https://*.onrender.com',                 # Wildcard
    'https://prediksi-kan-1.onrender.com'  # <-- TAMBAHKAN DOMAIN LENGKAP
]


# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'pwa',
    # Aplikasi Anda
    'predictions.apps.PredictionsConfig', # <-- Ubah ini agar signals.py dimuat

    # Otentikasi Google
    'django.contrib.sites',
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.google',
]

# ID Situs (diperlukan oleh allauth)
SITE_ID = 1

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'allauth.account.middleware.AccountMiddleware', # <-- Tambahkan ini
]

ROOT_URLCONF = 'football_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')], # <-- Arahkan ke folder templates root
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'football_project.wsgi.application'


# --- 2. Database (Gantikan SQLAlchemy) ---
# Menggunakan dj-database-url untuk membaca env DATABASE_URL (seperti di Flask)
DATABASES = {
    'default': dj_database_url.config(
        default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}",
        conn_max_age=600
    )
}
# ------------------------------------------

# --- 3. Otentikasi (Gantikan Flask-Login & Authlib) ---
AUTH_USER_MODEL = 'predictions.CustomUser' # <-- Arahkan ke model custom kita

AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
    'allauth.account.auth_backends.AuthenticationBackend',
]

# ==========================================================
# --- Konfigurasi Allauth (DIPERBARUI UNTUK MENGHILANGKAN WARNINGS) ---
# ==========================================================
ACCOUNT_LOGIN_METHODS = ['email']             
ACCOUNT_SIGNUP_FIELDS = ['email']             
ACCOUNT_USER_MODEL_USERNAME_FIELD = None      
ACCOUNT_USERNAME_REQUIRED = False             
ACCOUNT_UNIQUE_EMAIL = True
ACCOUNT_EMAIL_VERIFICATION = 'none' # Email sudah diverifikasi oleh Google         

# ▼▼▼ TAMBAHKAN BARIS INI ▼▼▼
SOCIALACCOUNT_AUTO_SIGNUP = True # Otomatis signup saat login sosial pertama kali
# ▲▲▲ AKHIR TAMBAHAN ▲▲▲

LOGIN_REDIRECT_URL = '/' 
LOGOUT_REDIRECT_URL = '/'
ACCOUNT_LOGOUT_ON_GET = True

# Konfigurasi Provider Google
SOCIALACCOUNT_PROVIDERS = {
    'google': {
        'APP': {
            'client_id': os.environ.get('GOOGLE_CLIENT_ID'),
            'secret': os.environ.get('GOOGLE_CLIENT_SECRET'),
            'key': ''
        },
        'SCOPE': [
            'profile',
            'email',
        ],
        'AUTH_PARAMS': {
            'access_type': 'online',
        }
    }
}
# ----------------------------------------------------

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]


# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I_18N = True # <-- Perbaikan typo kecil, 'USE_I18N' bukan 'USE_I18N'
USE_TZ = True


# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles') # Untuk deployment Render

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ==========================================================
# PWA SETTINGS (WAJIB ADA)
# ==========================================================
PWA_APP_NAME = 'Prediksi-Kan'
PWA_APP_DESCRIPTION = 'Aplikasi prediksi sepak bola bertenaga Machine Learning dan AI.'
PWA_APP_SCOPE = '/'
PWA_APP_START_URL = '/'
PWA_APP_DISPLAY = 'standalone'  # Menghilangkan address bar browser
PWA_APP_ORIENTATION = 'portrait'
PWA_APP_STATUS_BAR_COLOR = '#0b1220' # Warna tema Anda
PWA_APP_THEME_COLOR = '#06b6d4'   # Warna aksen

# URL Ikon (perlu disiapkan)
PWA_APP_ICONS = [
    {
        'src': '/static/images/prediksikan-logo.png', # 192x192
        'sizes': '192x192'
    },
    {
        'src': '/static/images/prediksikan-logo1.png', # 512x512
        'sizes': '512x512'
    }
]

# (Opsional) URL gambar Splash Screen (Android)
PWA_APP_SPLASH_SCREEN = [
    {
        'src': '/static/images/splash-2048x2732.png',
        'media': '(device-width: 1024px) and (device-height: 1366px) and (-webkit-device-pixel-ratio: 2)'
    }
]

# Pengaturan Cache (Service Worker)
PWA_SERVICE_WORKER_PATH = os.path.join(BASE_DIR, 'serviceworker.js')
PWA_APP_DEBUG = DEBUG # Gunakan mode debug Django