# predictions/templatetags/dict_filters.py
from django import template
import math 

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

    if isinstance(val, str) and '.' in val and key not in ['FTR', 'HomeTeam', 'AwayTeam', 'Date', 'league']:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass 

    elif isinstance(val, str) and key not in ['FTR', 'HomeTeam', 'AwayTeam', 'Date', 'league']:
         try:
             int_val = int(val)
             if float(val) == int_val:
                 return float(int_val) 
         except (ValueError, TypeError):
             pass 

    return val 

@register.filter
def multiply(value, arg):
    """
    Multiplies the value by the arg.
    Assumes value and arg can be converted to floats.
    Returns the ceiling integer of the result.
    """
    try:
        v = float(value)
        a = float(arg)
        result = v * a
        return math.ceil(result) 
    except (ValueError, TypeError, AttributeError):
        return 0 