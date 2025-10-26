# football_project/urls.py

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Arahkan semua URL otentikasi (login, logout, google) ke allauth
    path('accounts/', include('allauth.urls')), 
    
    # Arahkan semua URL aplikasi Anda ke predictions.urls
    path('', include('predictions.urls')),
    path('', include('pwa.urls')),
]