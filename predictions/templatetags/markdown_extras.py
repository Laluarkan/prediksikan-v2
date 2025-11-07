# predictions/templatetags/markdown_extras.py

from django import template
from django.template.defaultfilters import stringfilter
import markdown

register = template.Library()

@register.filter
@stringfilter
def render_markdown(value):
    """
    Mengubah teks Markdown menjadi HTML.
    """
    return markdown.markdown(value, extensions=['fenced_code', 'tables'])