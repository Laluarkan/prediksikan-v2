# predictions/models.py

from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager # <-- TAMBAHKAN BaseUserManager
from django.conf import settings
from django.utils import timezone

# ==========================================================
# --- TAMBAHKAN MANAGER KUSTOM UNTUK CUSTOM USER ---
# ==========================================================
class CustomUserManager(BaseUserManager):
    """
    Manager kustom untuk model User kita yang tidak menggunakan username.
    """
    def create_user(self, email, password=None, **extra_fields):
        """
        Membuat dan menyimpan user biasa dengan email dan password.
        """
        if not email:
            raise ValueError('Email harus diisi')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        """
        Membuat dan menyimpan superuser dengan email dan password.
        """
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        
        # <-- TAMBAHKAN: Atur role ke 'admin' secara otomatis saat membuat superuser
        extra_fields.setdefault('role', 'admin') 

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser harus memiliki is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser harus memiliki is_superuser=True.')

        return self.create_user(email, password, **extra_fields)
# ==========================================================
# AKHIR BLOK TAMBAHAN
# ==========================================================


# --- 1. Gantikan User(db.Model, UserMixin) ---
class CustomUser(AbstractUser):
    # Hapus username, gunakan email sebagai gantinya
    username = None
    email = models.EmailField('email address', unique=True)
    
    # Tambahkan field 'role' dari Flask
    ROLE_CHOICES = (
        ('user', 'User'),
        ('admin', 'Admin'),
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = [] 

    # --- TAMBAHKAN BARIS INI ---
    # Memberitahu CustomUser untuk menggunakan Manager kustom yang baru kita buat
    objects = CustomUserManager() 
    # -------------------------

    def __str__(self):
        return self.email

# --- 2. Gantikan PredictionHistory(db.Model) ---
class PredictionHistory(models.Model):
    # Gunakan ForeignKey untuk terhubung ke CustomUser
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='histories'
    )
    # Gantikan default=lambda: datetime.now(timezone.utc)
    timestamp = models.DateTimeField(default=timezone.now)
    league = models.CharField(max_length=100)
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    
    # Django memiliki JSONField bawaan
    prediction_data = models.JSONField()

    def __str__(self):
        return f"{self.user.email} - {self.home_team} vs {self.away_team}"

    class Meta:
        ordering = ['-timestamp'] # Urutkan dari yang terbaru