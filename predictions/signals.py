# predictions/signals.py

from django.dispatch import receiver
from allauth.account.signals import user_signed_up
from django.conf import settings

@receiver(user_signed_up)
def set_admin_role(sender, request, user, **kwargs):
    """
    Saat user baru mendaftar (via Google), cek apakah emailnya
    adalah email ADMIN.
    """
    if user.email == settings.ADMIN_EMAIL:
        user.role = 'admin'
        user.is_staff = True  # Izinkan login ke /admin
        user.is_superuser = True # Izinkan akses penuh ke /admin
        user.save()