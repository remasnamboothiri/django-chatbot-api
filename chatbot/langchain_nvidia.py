import warnings
import requests
import re
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from decouple import config

# Suppress warnings
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
        response = requests.get(base_url, params=params, timeout=60)
        
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

# ============================================
# SECTION 3: HELPER FUNCTION - Extract City Name
# ============================================

def extract_city_name(user_message):
    """
    Extract city name from user message
    
    Examples:
    "weather in London" -> "London"
    "What's the temperature in Paris?" -> "Paris"
    "How's the climate in New York" -> "New York"
    """
    # Common patterns to find city names
    patterns = [
        r'weather in ([A-Za-z\s]+)',
        r'temperature in ([A-Za-z\s]+)',
        r'forecast for ([A-Za-z\s]+)',
        r'climate in ([A-Za-z\s]+)',
        r'weather of ([A-Za-z\s]+)',
        r'temperature of ([A-Za-z\s]+)',
    ]
    
    message_lower = user_message.lower()
    
    for pattern in patterns:
        match = re.search(pattern, message_lower)
        if match:
            city = match.group(1).strip()
            # Capitalize each word (e.g., "new york" -> "New York")
            return city.title()
    
    # If no pattern matches, try to find the last word(s) as city name
    # This handles cases like "weather London" or "temperature Paris"
    words = user_message.split()
    if len(words) >= 2:
        return words[-1].title()
    
    return None


# ============================================
# SECTION 4: MAIN AI RESPONSE FUNCTION
# ============================================

def get_nvidia_response(user_message):
    """
    Get AI response using NVIDIA's API
    
    LOGIC:
    1. Check if user is asking about weather
    2. If YES -> Extract city name and call get_weather()
    3. If NO -> Use simple ChatNVIDIA for normal conversation
    """
    try:
        # Get NVIDIA API key from .env file
        nvidia_api_key = config('NVIDIA_API_KEY', default='')
        
        # Validate API key
        if not nvidia_api_key:
            return "⚠️ NVIDIA API key not found. Please add NVIDIA_API_KEY to .env file!"
        
        # ============================================
        # STEP 1: Check if user is asking about weather
        # ============================================
        weather_keywords = ['weather', 'temperature', 'forecast', 'climate']
        message_lower = user_message.lower()
        
        is_weather_question = any(keyword in message_lower for keyword in weather_keywords)
        
        # ============================================
        # STEP 2: If weather question -> Call weather function
        # ============================================
        if is_weather_question:
            city_name = extract_city_name(user_message)
            
            if city_name:
                # Call the weather function
                weather_data = get_weather(city_name)
                return weather_data
            else:
                return "Please specify a city name. For example: 'weather in London'"
        
        # ============================================
        # STEP 3: If NOT weather question -> Normal AI chat
        # ============================================
        else:
            # Create the AI model
            llm = ChatNVIDIA(
                model="nvidia/llama-3.1-nemotron-nano-8b-v1",
                api_key=nvidia_api_key,
                temperature=0.7,
                max_tokens=500,
            )
            
            # Create messages
            messages = [
                SystemMessage(content="You are a helpful, friendly AI assistant."),
                HumanMessage(content=user_message)
            ]
            
            # Get AI response
            response = llm.invoke(messages)
            
            # Extract and return content
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        
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