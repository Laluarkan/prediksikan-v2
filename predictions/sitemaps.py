# predictions/sitemaps.py

from django.contrib.sitemaps import Sitemap
from django.urls import reverse
from .models import Article

class ArticleSitemap(Sitemap):
    """
    Sitemap untuk semua artikel yang statusnya 'published'.
    """
    changefreq = "weekly" # Seberapa sering artikel berubah (weekly/monthly/never)
    priority = 0.9 # Prioritas relatif terhadap halaman lain (0.0 - 1.0)

    def items(self):
        # Ambil semua artikel yang sudah 'published'
        return Article.objects.filter(status='published')

    def lastmod(self, obj):
        # Gunakan tanggal artikel terakhir diupdate
        return obj.updated_on

class StaticViewSitemap(Sitemap):
    """
    Sitemap untuk halaman statis (Home, Prediksi, Statistik).
    """
    priority = 0.7
    changefreq = "daily" # Halaman prediksi mungkin berubah setiap hari

    def items(self):
        # Daftar nama URL dari 'predictions.urls'
        return ['home', 'index', 'stats']

    def location(self, item):
        # Dapatkan URL dari namanya
        return reverse(item)