"""
Chatbot app URLs
"""

from django.urls import path
from . import views

app_name = 'chatbot'

urlpatterns = [
    # API endpoints only
    path('api/chat/', views.chat, name='api_chat'),
    path('api/health/', views.health_check, name='health_check'),
]
