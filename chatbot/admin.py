

# Register your models here.
"""
Admin configuration
"""

from django.contrib import admin
from .models import ChatMessage

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['user_message', 'bot_response', 'timestamp']
    list_filter = ['timestamp']
    search_fields = ['user_message', 'bot_response']
