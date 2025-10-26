# predictions/urls.py

from django.urls import path
from . import views

# Ini mirip dengan @app.route(...)
urlpatterns = [
    # Halaman
    path('', views.home_page, name='home'),
    path('index/', views.index, name='index'),
    path('stats/', views.stats_page, name='stats'),
    path('add_data/', views.add_data_page, name='add_data'),
    
    # API
    path('api/leagues', views.api_leagues, name='api_leagues'),
    path('api/team_stats', views.api_team_stats, name='api_team_stats'),
    path('api/teams', views.api_teams, name='api_teams'),
    path('api/features', views.api_features, name='api_features'),
    path('api/predict', views.api_predict, name='api_predict'),
    
    # API (Perlu Login)
    path('api/history', views.api_history, name='api_history'),
    path('api/clear_history', views.api_clear_history, name='api_clear_history'),
    
    # API (Khusus Admin)
    path('api/upload_csv', views.api_upload_csv, name='api_upload_csv'),
    path('api/save_new_matches', views.api_save_new_matches, name='api_save_new_matches'),
]