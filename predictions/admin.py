# predictions/admin.py

from django.contrib import admin
from .models import CustomUser, PredictionHistory

# Daftarkan model CustomUser (Tidak Berubah)
@admin.register(CustomUser)
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ('email', 'first_name', 'role', 'is_staff')
    list_filter = ('role', 'is_staff')
    search_fields = ('email', 'first_name')

# Daftarkan model PredictionHistory (Diperbarui)
@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    # ▼▼▼ MODIFIKASI list_display ▼▼▼
    list_display = (
        'user', 
        'league', 
        'home_team', 
        'away_team', 
        'timestamp', 
        'get_prediction_hda',
        'get_prediction_ou',
        'get_prediction_btts',
        'hda_chosen',            # Pilihan User HDA
        'over_under_chosen',     # Pilihan User O/U
        'btts_chosen',           # Pilihan User BTTS
        'is_preferred_choice'    # Flag Pilihan Terbaik
    )
    # ▲▲▲ AKHIR MODIFIKASI list_display ▼▼▲
    
    list_filter = ('league', 'timestamp', 'is_preferred_choice', 'hda_chosen', 'over_under_chosen', 'btts_chosen') # Tambah filter
    search_fields = ('home_team', 'away_team', 'user__email')
    
    # Fungsi kustom untuk H/D/A (Model)
    def get_prediction_hda(self, obj):
        return obj.prediction_data.get('HDA', {}).get('label', 'N/A')
    get_prediction_hda.short_description = 'Model H/D/A'

    # Tambahkan Fungsi baru untuk O/U dan BTTS (Model)
    def get_prediction_ou(self, obj):
        return obj.prediction_data.get('OU25', {}).get('label', 'N/A')
    get_prediction_ou.short_description = 'Model O/U'

    def get_prediction_btts(self, obj):
        return obj.prediction_data.get('BTTS', {}).get('label', 'N/A')
    get_prediction_btts.short_description = 'Model BTTS'