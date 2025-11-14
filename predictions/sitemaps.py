# predictions/sitemaps.py

from django.contrib.sitemaps import Sitemap
from django.urls import reverse
from .models import Article

class ArticleSitemap(Sitemap):
    """
    Sitemap untuk semua artikel yang statusnya 'published'.
    """
    changefreq = "weekly" 
    priority = 0.9 

    def items(self):
        return Article.objects.filter(status='published')

    def lastmod(self, obj):
        return obj.updated_on

class StaticViewSitemap(Sitemap):
    """
    Sitemap untuk halaman statis (Home, Prediksi, Statistik).
    """
    priority = 0.7
    changefreq = "daily" 

    def items(self):
        return ['home', 'index', 'stats']

    def location(self, item):
        return reverse(item)