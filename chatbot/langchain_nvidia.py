import warnings
import requests  #  For calling Weather API
import json      # For handling JSON data
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import Tool, initialize_agent, AgentType  # : For tools
from decouple import config

# Suppress the specific warning
warnings.filterwarnings('ignore', message='.*type is unknown.*')



# ============================================
# SECTION 2: WEATHER FUNCTION (NEW)
# ============================================

def get_weather(city_name):
    """
    This function gets real weather data from OpenWeatherMap API
    
    How it works:
    1. Takes city name as input (e.g., "London")
    2. Calls Weather API with the city name
    3. Gets weather data (temperature, condition, etc.)
    4. Returns formatted weather information
    
    Example:
    Input: "London"
    Output: "Temperature: 15°C, Condition: Cloudy, Humidity: 65%"
    """
    try:
        # Get Weather API key from .env file
        weather_api_key = config('WEATHER_API_KEY')
        # Check if API key exists
        if not weather_api_key:
            return "Weather API key not found. Please add WEATHER_API_KEY to .env file"
        
        # Build the API URL
        # This URL asks OpenWeatherMap for weather data
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city_name,           # City name (e.g., "London")
            'appid': weather_api_key, # Your API key
            'units': 'metric'         # Use Celsius (not Fahrenheit)
        }
        
        # Call the Weather API
        response = requests.get(base_url, params=params, timeout=10)
        
        # Check if API call was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Extract weather information
            temperature = data['main']['temp']           # Temperature in Celsius
            feels_like = data['main']['feels_like']      # Feels like temperature
            humidity = data['main']['humidity']          # Humidity percentage
            description = data['weather'][0]['description']  # Weather condition
            wind_speed = data['wind']['speed']           # Wind speed
            
            # Format the weather information nicely
            weather_info = f"""
Weather in {city_name}:
- Temperature: {temperature}°C (Feels like: {feels_like}°C)
- Condition: {description.capitalize()}
- Humidity: {humidity}%
- Wind Speed: {wind_speed} m/s
            """
            
            return weather_info.strip()
        
        elif response.status_code == 404:
            return f"City '{city_name}' not found. Please check the spelling."
        
        else:
            return f"Could not fetch weather data. Error code: {response.status_code}"
    
    except requests.exceptions.Timeout:
        return "Weather API request timed out. Please try again."
    
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Weather API: {str(e)}"
    
    except Exception as e:
        return f"Error getting weather: {str(e)}"

# def get_nvidia_response(user_message):
#     """
#     Get AI response using NVIDIA's FREE API
    
#     WORKING MODEL: nvidia/llama-3.1-nemotron-nano-8b-v1
#     This is THE BEST model for chatbots - perfect balance!
#     """
#     try:
#         # Get API key from .env file
#         nvidia_api_key = config('NVIDIA_API_KEY', default='')
        
#         # Validate API key
#         if not nvidia_api_key:
#             return "⚠️ NVIDIA API key not found. Please add NVIDIA_API_KEY=nvapi-your-key to your .env file!"
        
#         #if not nvidia_api_key.startswith('nvapi-'):
#             #return "⚠️ Invalid NVIDIA API key format. Key should start with 'nvapi-'"
        
#         # Initialize ChatNVIDIA with CORRECT MODEL NAME!
#         # FORMAT: publisher/model-name (MUST match exactly!)
#         llm = ChatNVIDIA(
#             model="nvidia/llama-3.1-nemotron-nano-8b-v1",  # ✅ CORRECT FORMAT!
#             api_key=nvidia_api_key,
#             temperature=0.7,
#             max_tokens=500,
#         )
        
#         # Create messages
#         messages = [
#             SystemMessage(content="You are a helpful, friendly AI assistant. "),
#             HumanMessage(content=user_message)
#         ]
        
#         # Get AI response
#         response = llm.invoke(messages)
        
#         # Extract content
#         if hasattr(response, 'content'):
#             return response.content
#         else:
#             return str(response)
            
#     except Exception as e:
#         error_msg = str(e).lower()
        
#         # User-friendly error messages
#         if "401" in error_msg or "unauthorized" in error_msg:
#             return "❌ API Key Error: Your NVIDIA API key is invalid. Get a new one from build.nvidia.com"
        
#         elif "404" in error_msg or "not found" in error_msg:
#             return "❌ Model not found. Using: nvidia/llama-3.1-nemotron-nano-8b-v1"
        
#         elif "rate limit" in error_msg or "429" in str(e):
#             return "❌ Too many requests! Please wait 30 seconds and try again."
        
#         elif "timeout" in error_msg:
#             return "❌ Request timed out. Please try a shorter message."
        
#         else:
#             return f"❌ Error: {str(e)[:150]}. Please check your API key and internet connection."
        
        
        
        
# ============================================
# SECTION 3: AI RESPONSE FUNCTION (MODIFIED)
# ============================================

def get_nvidia_response(user_message):
    """
    Get AI response using NVIDIA's API with Weather Tool
    
    How it works:
    1. Creates a Weather Tool from the get_weather function
    2. Creates an AI agent that can use the Weather Tool
    3. When user asks about weather, AI automatically calls the tool
    4. Returns the complete response
    
    Example Flow:
    User: "What's the weather in Paris?"
    → AI detects "weather" keyword
    → AI calls get_weather("Paris")
    → AI gets weather data
    → AI formats nice response: "The weather in Paris is..."
    """
    try:
        # Get NVIDIA API key from .env file
        nvidia_api_key = config('NVIDIA_API_KEY', default='')
        
        # Validate API key
        if not nvidia_api_key:
            return "⚠️ NVIDIA API key not found. Please add NVIDIA_API_KEY to .env file!"
        
        # ============================================
        # STEP 1: Create the Weather Tool
        # ============================================
        # This tells LangChain about our weather function
        weather_tool = Tool(
            name="Weather",  # Tool name (AI will see this)
            func=get_weather,  # The function to call
            description="Useful for getting current weather information for a city. Input should be a city name."
            # This description helps AI know when to use this tool
        )
        
        # ============================================
        # STEP 2: Create the AI Model
        # ============================================
        llm = ChatNVIDIA(
            model="nvidia/llama-3.1-nemotron-nano-8b-v1",
            api_key=nvidia_api_key,
            temperature=0.7,
            max_tokens=500,
        )
        
        # ============================================
        # STEP 3: Create an Agent with Tools
        # ============================================
        # Agent = AI that can use tools
        # We give it the Weather Tool
        agent = initialize_agent(
            tools=[weather_tool],  # List of tools (we only have weather tool)
            llm=llm,               # The AI model
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Agent type
            verbose=False,         # Don't show debug info
            handle_parsing_errors=True  # Handle errors gracefully
        )
        
        # ============================================
        # STEP 4: Send Message to Agent
        # ============================================
        # The agent will:
        # - Read the user message
        # - Decide if it needs to use the weather tool
        # - Call the tool if needed
        # - Generate a response
        response = agent.run(user_message)
        
        return response
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # User-friendly error messages
        if "401" in error_msg or "unauthorized" in error_msg:
            return "❌ API Key Error: Your NVIDIA API key is invalid."
        
        elif "404" in error_msg or "not found" in error_msg:
            return "❌ Model not found. Please check the model name."
        
        elif "rate limit" in error_msg or "429" in str(e):
            return "❌ Too many requests! Please wait 30 seconds and try again."
        
        elif "timeout" in error_msg:
            return "❌ Request timed out. Please try a shorter message."
        
        else:
            return f"❌ Error: {str(e)[:150]}. Please check your API key and internet connection."