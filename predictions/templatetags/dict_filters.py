# predictions/templatetags/dict_filters.py
from django import template
import math # Import math untuk pembulatan

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """
    Allows accessing dictionary keys with variables in Django templates.
    Usage: {{ my_dict|get_item:my_variable_key }}
    Returns the value, trying to convert numeric strings to float.
    """
    if not isinstance(dictionary, dict):
        return None
    
    val = dictionary.get(key)
    
    # Coba konversi ke float jika string terlihat seperti angka desimal
    # Jangan konversi string FTR atau yang tidak punya titik
    if isinstance(val, str) and '.' in val and key not in ['FTR', 'HomeTeam', 'AwayTeam', 'Date', 'league']:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass # Kembalikan string asli jika gagal
            
    # Coba konversi ke float jika string terlihat seperti integer (untuk perkalian nanti)
    # Jangan konversi FTR
    elif isinstance(val, str) and key not in ['FTR', 'HomeTeam', 'AwayTeam', 'Date', 'league']:
         try:
             # Cek apakah itu integer
             int_val = int(val)
             # Jika berhasil dan sama dgn float-nya, kembalikan float
             if float(val) == int_val:
                 return float(int_val) # Kembalikan sebagai float agar bisa dikalikan
         except (ValueError, TypeError):
             pass # Kembalikan string asli jika gagal

    return val # Kembalikan tipe asli (int, float, str, None, dll.)


# ▼▼▼ TAMBAHKAN FILTER BARU INI ▼▼▼
@register.filter
def multiply(value, arg):
    """
    Multiplies the value by the arg.
    Assumes value and arg can be converted to floats.
    Returns the ceiling integer of the result.
    """
    try:
        # Konversi value (avg goals) dan arg (count) ke float
        v = float(value)
        a = float(arg)
        # Hitung hasil perkalian
        result = v * a
        # Bulatkan ke atas ke integer terdekat
        # Gunakan math.ceil agar konsisten dengan JS di halaman prediksi (Math.round)
        # Atau gunakan round() jika pembulatan standar lebih diinginkan
        return math.ceil(result) 
    except (ValueError, TypeError, AttributeError):
        # Kembalikan 0 jika value atau arg tidak valid
        return 0 
# ▲▲▲ AKHIR FILTER BARU ▲▲▲