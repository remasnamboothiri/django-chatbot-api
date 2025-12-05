from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from .langchain_nvidia import get_nvidia_response
from .models import ChatMessage


@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    """API endpoint for chat with NVIDIA AI"""
    try:
        # Get user message
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        
        # Validate input
        if not user_message:
            return JsonResponse({
                'error': 'Please enter a message',
                'status': 'error'
            }, status=400)
        
        # Get AI response from NVIDIA
        bot_response = get_nvidia_response(user_message)
        
        # Save to database
        try:
            ChatMessage.objects.create(
                user_message=user_message,
                bot_response=bot_response
            )
        except Exception as db_error:
            print(f"Database error: {db_error}")
            # Continue even if save fails
        
        # Return response
        return JsonResponse({
            'response': bot_response,
            'status': 'success'
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON format',
            'status': 'error'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'status': 'error'
        }, status=500)


@require_http_methods(["GET"])
def health_check(request):
    """API health check endpoint"""
    return JsonResponse({
        'status': 'healthy',
        'message': 'Django API is running',
        'version': '1.0'
    })




# from django.shortcuts import render
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json
# from .langchain_nvidia import get_nvidia_response
# from .models import ChatMessage


# def home(request):
#     """Render the chat interface"""
#     return render(request, 'chatbot/home.html')


# @csrf_exempt
# def chat(request):
#     """Handle chat with NVIDIA AI"""
#     if request.method == 'POST':
#         try:
#             # Get user message
#             data = json.loads(request.body)
#             user_message = data.get('message', '').strip()
            
#             # Validate input
#             if not user_message:
#                 return JsonResponse({
#                     'error': 'Please enter a message',
#                     'status': 'error'
#                 }, status=400)
            
#             # Get AI response from NVIDIA
#             bot_response = get_nvidia_response(user_message)
            
#             # Save to database
#             try:
#                 ChatMessage.objects.create(
#                     user_message=user_message,
#                     bot_response=bot_response
#                 )
#             except Exception as db_error:
#                 print(f"Database error: {db_error}")
#                 # Continue even if save fails
            
#             # Return response
#             return JsonResponse({
#                 'response': bot_response,
#                 'status': 'success'
#             })
            
#         except json.JSONDecodeError:
#             return JsonResponse({
#                 'error': 'Invalid request',
#                 'status': 'error'
#             }, status=400)
#         except Exception as e:
#             return JsonResponse({
#                 'error': str(e),
#                 'status': 'error'
#             }, status=500)
    
#     return JsonResponse({
#         'error': 'Only POST allowed',
#         'status': 'error'
#     }, status=405)


