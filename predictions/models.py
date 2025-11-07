# predictions/models.py

from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager 
from django.conf import settings
from django.utils import timezone
from django.utils.text import slugify
from django.urls import reverse

# ==========================================================
# --- MANAGER KUSTOM UNTUK CUSTOM USER (Tidak Berubah) ---
# ==========================================================
class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email: raise ValueError('Email harus diisi')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user
    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('role', 'admin') 
        if extra_fields.get('is_staff') is not True: raise ValueError('Superuser harus memiliki is_staff=True.')
        if extra_fields.get('is_superuser') is not True: raise ValueError('Superuser harus memiliki is_superuser=True.')
        return self.create_user(email, password, **extra_fields)
# ==========================================================

# --- 1. Model CustomUser (Tidak Berubah) ---
class CustomUser(AbstractUser):
    username = None
    email = models.EmailField('email address', unique=True)
    ROLE_CHOICES = (('user', 'User'), ('admin', 'Admin'))
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []
    objects = CustomUserManager()
    def __str__(self): return self.email

# --- 2. Model PredictionHistory (Diperbarui) ---
class PredictionHistory(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='histories')
    timestamp = models.DateTimeField(default=timezone.now)
    league = models.CharField(max_length=100)
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    prediction_data = models.JSONField()
    input_features = models.JSONField(null=True, blank=True)
    
    # ▼▼▼ TAMBAHKAN KOLOM BARU INI UNTUK PILIHAN USER ▼▼▼
    is_preferred_choice = models.BooleanField(default=False) # Flag Pilihan Terbaik
    hda_chosen = models.CharField(max_length=1, default='N') # H, D, A, atau N (None)
    over_under_chosen = models.CharField(max_length=5, default='N') # Over, Under, atau N
    btts_chosen = models.CharField(max_length=3, default='N') # Yes, No, atau N
    # ▲▲▲ AKHIR TAMBAHAN ▼▼▼

    # ▼▼▼ TAMBAHKAN KOLOM BARU UNTUK HASIL ▼▼▼
    # Status apakah pertandingan sudah selesai (berdasarkan upload CSV baru)
    is_match_completed = models.BooleanField(default=False)
    
    # Hasil tebakan user (W = Win, L = Lose, N/A = Not Played/Not Checked)
    hda_result = models.CharField(max_length=3, default='N/A')
    ou_result = models.CharField(max_length=3, default='N/A')
    btts_result = models.CharField(max_length=3, default='N/A')
    # ▲▲▲ AKHIR TAMBAHAN ▼▼▲


    def __str__(self):
        return f"{self.user.email} - {self.home_team} vs {self.away_team}"

    class Meta:
        ordering = ['-timestamp']

class Article(models.Model):
    STATUS_CHOICES = (
        ('draft', 'Draft'),
        ('published', 'Published'),
    )
    
    title = models.CharField(max_length=200, unique=True)
    slug = models.SlugField(max_length=200, unique=True, blank=True)
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='blog_posts')
    
    # Konten
    content = models.TextField() # Ini akan menjadi editor teks biasa
    excerpt = models.CharField(max_length=255, help_text="Ringkasan singkat (untuk SEO dan pratinjau).")
    
    # SEO Meta
    meta_description = models.CharField(max_length=160, blank=True, null=True, help_text="Deskripsi SEO (maks 160 karakter).")
    
    # Status & Waktu
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='draft')
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_on']

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        # Otomatis buat slug dari judul jika slug kosong
        if not self.slug:
            self.slug = slugify(self.title)
        super().save(*args, **kwargs)

    def get_absolute_url(self):
        # Ini adalah URL kanonis untuk setiap artikel (penting untuk SEO)
        return reverse('article_detail', kwargs={'slug': self.slug})