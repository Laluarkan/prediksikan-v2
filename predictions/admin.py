# predictions/admin.py

from django.contrib import admin
from .models import CustomUser, PredictionHistory, Article
from .models import CustomUser, PredictionHistory, Article, SystemPerformanceStats

@admin.register(CustomUser)
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ('email', 'first_name', 'role', 'is_staff')
    list_filter = ('role', 'is_staff')
    search_fields = ('email', 'first_name')

@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = (
        'user', 
        'league', 
        'home_team', 
        'away_team', 
        'timestamp', 
        'get_prediction_hda',
        'get_prediction_ou',
        'get_prediction_btts',
        'hda_chosen',          
        'over_under_chosen',     
        'btts_chosen',           
        'is_preferred_choice'    
    )
    
    list_filter = ('league', 'timestamp', 'is_preferred_choice', 'hda_chosen', 'over_under_chosen', 'btts_chosen') 
    search_fields = ('home_team', 'away_team', 'user__email')

    def get_prediction_hda(self, obj):
        return obj.prediction_data.get('HDA', {}).get('label', 'N/A')
    get_prediction_hda.short_description = 'Model H/D/A'

    def get_prediction_ou(self, obj):
        return obj.prediction_data.get('OU25', {}).get('label', 'N/A')
    get_prediction_ou.short_description = 'Model O/U'

    def get_prediction_btts(self, obj):
        return obj.prediction_data.get('BTTS', {}).get('label', 'N/A')
    get_prediction_btts.short_description = 'Model BTTS'

@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'status', 'created_on')
    list_filter = ('status', 'created_on', 'author')
    search_fields = ('title', 'content')
    prepopulated_fields = {'slug': ('title',)} 

    def get_form(self, request, obj=None, **kwargs):
        form = super(ArticleAdmin, self).get_form(request, obj, **kwargs)
        form.base_fields['author'].queryset = CustomUser.objects.filter(is_staff=True)
        return form
    
@admin.register(SystemPerformanceStats)
class SystemPerformanceStatsAdmin(admin.ModelAdmin):
    list_display = (
        'timestamp', 
        'league_name', 
        'total_matches_processed', 
        'best_bet_win_rate', 
        'hda_win_rate', 
        'ou_win_rate', 
        'btts_win_rate'
    )
    list_filter = ('league_name', 'timestamp')
    # Buat read-only, karena ini dibuat otomatis
    readonly_fields = [f.name for f in SystemPerformanceStats._meta.get_fields()]

    def has_add_permission(self, request):
        return False # Tidak boleh menambah manual

    def has_change_permission(self, request, obj=None):
        return False # Tidak boleh mengubah manual