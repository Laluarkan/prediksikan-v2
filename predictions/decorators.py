# predictions/decorators.py

from django.contrib.auth.decorators import user_passes_test
from django.shortcuts import redirect
from django.contrib import messages

def admin_required(function):
    """
    Decorator untuk view yang memeriksa apakah user adalah admin.
    """
    def check_admin(user):
        if user.is_authenticated and user.role == 'admin':
            return True
        messages.warning(user.request, "Hanya admin yang dapat mengakses halaman ini.")
        return False

    return user_passes_test(check_admin, login_url='/accounts/login/')(function)