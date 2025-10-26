# predictions/models.py

from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager 
from django.conf import settings
from django.utils import timezone

# ==========================================================
# --- MANAGER KUSTOM UNTUK CUSTOM USER ---
# ==========================================================
class CustomUserManager(BaseUserManager):
    """
    Manager kustom untuk model User kita yang tidak menggunakan username.
    """
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('Email harus diisi')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('role', 'admin') 

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser harus memiliki is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser harus memiliki is_superuser=True.')

        return self.create_user(email, password, **extra_fields)
# ==========================================================

# --- 1. Model CustomUser ---
class CustomUser(AbstractUser):
    username = None
    email = models.EmailField('email address', unique=True)
    
    ROLE_CHOICES = (
        ('user', 'User'),
        ('admin', 'Admin'),
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = [] 

    objects = CustomUserManager() 

    def __str__(self):
        return self.email

# --- 2. Model PredictionHistory ---
class PredictionHistory(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='histories'
    )
    timestamp = models.DateTimeField(default=timezone.now)
    league = models.CharField(max_length=100)
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    prediction_data = models.JSONField()
    
    # ▼▼▼ TAMBAHKAN FIELD INI ▼▼▼
    # Menyimpan fitur input yang digunakan saat prediksi
    input_features = models.JSONField(null=True, blank=True) 
    # ▲▲▲ AKHIR TAMBAHAN ▲▲▲

    def __str__(self):
        return f"{self.user.email} - {self.home_team} vs {self.away_team}"

    class Meta:
        ordering = ['-timestamp'] # Urutkan dari yang terbaru