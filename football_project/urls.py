# football_project/urls.py

from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView
from django.contrib.sitemaps.views import sitemap
from predictions.sitemaps import ArticleSitemap, StaticViewSitemap

sitemaps = {
    'static': StaticViewSitemap,
    'articles': ArticleSitemap,
}

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Arahkan semua URL otentikasi (login, logout, google) ke allauth
    path('accounts/', include('allauth.urls')), 
    
    # Arahkan semua URL aplikasi Anda ke predictions.urls
    path('', include('predictions.urls')),
    path('', include('pwa.urls')),

    path('robots.txt', TemplateView.as_view(template_name="robots.txt", content_type="text/plain")),
    path('sitemap.xml', sitemap, {'sitemaps': sitemaps}, name='django.contrib.sitemaps.views.sitemap'),
    path('ads.txt', TemplateView.as_view(template_name="ads.txt", content_type="text/plain")),

]