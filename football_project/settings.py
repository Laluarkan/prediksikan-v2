# football_project/settings.py

import os
from pathlib import Path
import dj_database_url
from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_URL = 'static/'
load_dotenv(os.path.join(BASE_DIR, '.env'))

# --- 1. Konfigurasi Kunci & Proyek ---
# ▼▼▼ PASTIKAN INI ADALAH 'SECRET_KEY' (bukan 'FLASK_SECRET_KEY') ▼▼▼
SECRET_KEY = os.environ.get('SECRET_KEY') 
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL') 
# ▲▲▲ -------------------------------------------------------- ▲▲▲

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
    '.onrender.com',
    '10.54.8.241'
] # Sesuaikan untuk produksi (Render akan mengaturnya)
CSRF_TRUSTED_ORIGINS = [
    'https://*.onrender.com',                 # Wildcard
    'https://prediksi-kan-1.onrender.com'  # <-- TAMBAHKAN DOMAIN LENGKAP
]


# Application definition
INSTALLED_APPS = [
    'predictions.apps.PredictionsConfig', # <-- Ubah ini agar signals.py dimuat
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.sitemaps',
    'pwa',
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
    'whitenoise.middleware.WhiteNoiseMiddleware', 
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'allauth.account.middleware.AccountMiddleware',
]

ROOT_URLCONF = 'football_project.urls'

# ▼▼▼ 2. GANTI BLOK TEMPLATES INI ▼▼▼
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')], # <-- Folder kustom Anda
        # 'APP_DIRS': True, # HAPUS BARIS INI
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
            # TAMBAHKAN 'loaders' INI SECARA EKSPLISIT
            'loaders': [
                # 1. Cari di 'DIRS' (templates/account/login.html) terlebih dahulu
                'django.template.loaders.filesystem.Loader', 
                # 2. BARU cari di folder aplikasi (allauth, admin, dll)
                'django.template.loaders.app_directories.Loader', 
            ],
        },
    },
]
# ▲▲▲ AKHIR PERUBAHAN BLOK ▲▲▲

WSGI_APPLICATION = 'football_project.wsgi.application'


# --- Database ---
DATABASES = {
    'default': dj_database_url.config(
        default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}",
        conn_max_age=600
    )
}
# ------------------------------------------

# --- Otentikasi ---
AUTH_USER_MODEL = 'predictions.CustomUser' 

AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
    'allauth.account.auth_backends.AuthenticationBackend',
]

# --- Konfigurasi Allauth (PERBAIKAN WARNING) ---
ACCOUNT_AUTHENTICATION_METHOD = 'email'
ACCOUNT_EMAIL_REQUIRED = True
ACCOUNT_UNIQUE_EMAIL = True
ACCOUNT_USER_MODEL_USERNAME_FIELD = None
ACCOUNT_USERNAME_REQUIRED = False
ACCOUNT_EMAIL_VERIFICATION = 'none'
SOCIALACCOUNT_AUTO_SIGNUP = True 
LOGIN_REDIRECT_URL = '/' 
LOGOUT_REDIRECT_URL = '/'
ACCOUNT_LOGOUT_ON_GET = True

# ▼▼▼ AKTIFKAN KEMBALI BLOK INI ▼▼▼
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
# ▲▲▲ -------------------------- ▲▲▲

# ... (Sisa file: Password validators, I18N, Static files, PWA, dll) ...
# ...
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I_18N = True
USE_TZ = True

STATIC_URL = 'static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles') 
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
PWA_APP_NAME = 'Prediksi-Kan'
PWA_APP_DESCRIPTION = 'Aplikasi prediksi sepak bola bertenaga Machine Learning dan AI.'
PWA_APP_SCOPE = '/'
PWA_APP_START_URL = '/'
PWA_APP_DISPLAY = 'standalone' 
PWA_APP_ORIENTATION = 'portrait'
PWA_APP_STATUS_BAR_COLOR = '#0b1220'
PWA_APP_THEME_COLOR = '#06b6d4'
PWA_APP_ICONS = [
    {
        'src': '/static/images/prediksikan-logo.png',
        'sizes': '192x192'
    },
    {
        'src': '/static/images/prediksikan-logo1.png',
        'sizes': '512x512'
    }
]
PWA_APP_SPLASH_SCREEN = [
    {
        'src': '/static/images/splash-2048x2732.png',
        'media': '(device-width: 1024px) and (device-height: 1366px) and (-webkit-device-pixel-ratio: 2)'
    }
]
PWA_SERVICE_WORKER_PATH = os.path.join(BASE_DIR, 'serviceworker.js')
PWA_APP_DEBUG = DEBUG