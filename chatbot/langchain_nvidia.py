import warnings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from decouple import config

# Suppress the specific warning
warnings.filterwarnings('ignore', message='.*type is unknown.*')

def get_nvidia_response(user_message):
    """
    Get AI response using NVIDIA's FREE API
    
    WORKING MODEL: nvidia/llama-3.1-nemotron-nano-8b-v1
    This is THE BEST model for chatbots - perfect balance!
    """
    try:
        # Get API key from .env file
        nvidia_api_key = config('NVIDIA_API_KEY', default='')
        
        # Validate API key
        if not nvidia_api_key:
            return "⚠️ NVIDIA API key not found. Please add NVIDIA_API_KEY=nvapi-your-key to your .env file!"
        
        #if not nvidia_api_key.startswith('nvapi-'):
            #return "⚠️ Invalid NVIDIA API key format. Key should start with 'nvapi-'"
        
        # Initialize ChatNVIDIA with CORRECT MODEL NAME!
        # FORMAT: publisher/model-name (MUST match exactly!)
        llm = ChatNVIDIA(
            model="nvidia/llama-3.1-nemotron-nano-8b-v1",  # ✅ CORRECT FORMAT!
            api_key=nvidia_api_key,
            temperature=0.7,
            max_tokens=500,
        )
        
        # Create messages
        messages = [
            SystemMessage(content="You are a helpful, friendly AI assistant. "),
            HumanMessage(content=user_message)
        ]
        
        # Get AI response
        response = llm.invoke(messages)
        
        # Extract content
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
            
    except Exception as e:
        error_msg = str(e).lower()
        
        # User-friendly error messages
        if "401" in error_msg or "unauthorized" in error_msg:
            return "❌ API Key Error: Your NVIDIA API key is invalid. Get a new one from build.nvidia.com"
        
        elif "404" in error_msg or "not found" in error_msg:
            return "❌ Model not found. Using: nvidia/llama-3.1-nemotron-nano-8b-v1"
        
        elif "rate limit" in error_msg or "429" in str(e):
            return "❌ Too many requests! Please wait 30 seconds and try again."
        
        elif "timeout" in error_msg:
            return "❌ Request timed out. Please try a shorter message."
        
        else:
            return f"❌ Error: {str(e)[:150]}. Please check your API key and internet connection."