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

ALLOWED_HOSTS = ['*'] # Sesuaikan untuk produksi (Render akan mengaturnya)
CSRF_TRUSTED_ORIGINS = ['https://*.onrender.com'] # Sesuaikan untuk Render


# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
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
ACCOUNT_LOGIN_METHODS = ['email']             # <-- Menggantikan ACCOUNT_AUTHENTICATION_METHOD
ACCOUNT_SIGNUP_FIELDS = ['email']             # <-- Menggantikan ACCOUNT_EMAIL_REQUIRED
ACCOUNT_USER_MODEL_USERNAME_FIELD = None      # <-- Memberitahu allauth kita tidak pakai username
ACCOUNT_USERNAME_REQUIRED = False             # <-- Ini sekarang diabaikan, tapi kita set False
ACCOUNT_UNIQUE_EMAIL = True
ACCOUNT_EMAIL_VERIFICATION = 'none'           # Sederhanakan, tidak perlu verifikasi email
# ==========================================================

LOGIN_REDIRECT_URL = '/' # Ganti login_manager.login_view
LOGOUT_REDIRECT_URL = '/'
ACCOUNT_LOGOUT_ON_GET = True # Memudahkan logout

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