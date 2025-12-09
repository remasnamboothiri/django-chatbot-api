import warnings
import requests
import re
import json
from openai import OpenAI  # NEW: For function calling
from decouple import config

# Suppress warnings
warnings.filterwarnings('ignore', message='.*type is unknown.*')


# ============================================
# SECTION 1: WEATHER FUNCTION (NO CHANGES)
# ============================================

def get_weather(city_name):
    """
    This function gets real weather data from OpenWeatherMap API
    """
    try:
        weather_api_key = config('WEATHER_API_KEY')
        if not weather_api_key:
            return "Weather API key not found. Please add WEATHER_API_KEY to .env file"
        
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city_name,
            'appid': weather_api_key,
            'units': 'metric'
        }
        
        response = requests.get(base_url, params=params, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            temperature = data['main']['temp']
            feels_like = data['main']['feels_like']
            humidity = data['main']['humidity']
            description = data['weather'][0]['description']
            wind_speed = data['wind']['speed']
            
            weather_info = f"""
Weather in {city_name}:
- Temperature: {temperature}°C (Feels like: {feels_like}°C)
- Condition: {description.capitalize()}
- Humidity: {humidity}%
- Wind Speed: {wind_speed} m/s
            """
            
            return weather_info.strip()
        
        elif response.status_code == 404:
            print(f"City '{city_name}' not found by name, trying geocoding...")
            coords = get_coordinates(city_name)
            
            if coords:
                print(f"Found coordinates: {coords['lat']}, {coords['lon']}")
                coord_params = {
                    'lat': coords['lat'],
                    'lon': coords['lon'],
                    'appid': weather_api_key,
                    'units': 'metric'
                }
                coord_response = requests.get(base_url, params=coord_params, timeout=60)
                
                if coord_response.status_code == 200:
                    data = coord_response.json()
                    temperature = data['main']['temp']
                    feels_like = data['main']['feels_like']
                    humidity = data['main']['humidity']
                    description = data['weather'][0]['description']
                    wind_speed = data['wind']['speed']
                    
                    weather_info = f"""
Weather in {coords['name']}:
- Temperature: {temperature}°C (Feels like: {feels_like}°C)
- Condition: {description.capitalize()}
- Humidity: {humidity}%
- Wind Speed: {wind_speed} m/s
                    """
                    return weather_info.strip()
            
            return f"City '{city_name}' not found. Please check the spelling or try a nearby major city."
        
        else:
            return f"Could not fetch weather data. Error code: {response.status_code}"
    
    except requests.exceptions.Timeout:
        return "Weather API request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Weather API: {str(e)}"
    except Exception as e:
        return f"Error getting weather: {str(e)}"


# ============================================
# SECTION 2: HELPER FUNCTIONS (NO CHANGES)
# ============================================

def get_coordinates(city_name):
    """Get latitude and longitude for a city"""
    try:
        weather_api_key = config('WEATHER_API_KEY')
        geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
        params = {
            'q': city_name,
            'limit': 1,
            'appid': weather_api_key
        }
        
        response = requests.get(geocoding_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                return {
                    'lat': data[0]['lat'],
                    'lon': data[0]['lon'],
                    'name': data[0].get('name', city_name)
                }
        return None
    except Exception as e:
        return None


# ============================================
# SECTION 3: MAIN AI FUNCTION (NEW - WITH FUNCTION CALLING)
# ============================================

def get_nvidia_response(user_message):
    """
    Get AI response using NVIDIA's API with Function Calling
    
    NEW APPROACH:
    - AI decides when to call weather function
    - AI extracts city name automatically
    - AI provides natural conversational responses
    """
    try:
        # Get NVIDIA API key
        nvidia_api_key = config('NVIDIA_API_KEY')
        
        if not nvidia_api_key:
            return "⚠️ NVIDIA API key not found. Please add NVIDIA_API_KEY to .env file!"
        
        # Create OpenAI client with NVIDIA endpoint
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key
        )
        
        # Define the weather function for AI
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Retrieves real-time weather data (temperature, humidity, conditions) for a specified city. MUST have BOTH requirements: 1) User message contains weather keywords (weather/temperature/rain/hot/cold/humidity/forecast) AND 2) User message contains a city name. Never call for: greetings (hi/hello), general chat, questions without city names.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g., London, Paris, New York"
                        }
                    },
                    "required": ["location"]
                }
            }
        }]
        
        # System prompt for natural conversation
        system_prompt = """You are a friendly AI assistant who can chat and provide weather information.

CRITICAL FUNCTION CALLING RULES:
1. ONLY call get_weather function when BOTH conditions are met:
   - User message has weather words: weather, temperature, rain, hot, cold, humidity, forecast
   - User message has a city name

2. NEVER call get_weather for:
   - Greetings: "hi", "hello", "good morning", "hey"
   - General questions: "how are you", "what can you do"
   - Chat without weather words or city names

3. Response guidelines:
   - Greetings → Respond warmly without mentioning weather
   - Weather + City → Call get_weather function
   - Other questions → Answer naturally

EXAMPLES:
❌ "Hello" → DO NOT call function → Say "Hello! How can I help you today?"
❌ "Good morning" → DO NOT call function → Say "Good morning! What would you like to know?"
✅ "Weather in London" → CALL get_weather("London")
✅ "Is it raining in Paris?" → CALL get_weather("Paris")
❌ "How are you?" → DO NOT call function → Say "I'm doing well, thanks!"
"""
        
        # First API call - Let AI decide if it needs to call function
        response = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-nano-8b-v1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            tools=tools,
            tool_choice="auto",  # AI decides when to use tools
            temperature=0.7,
            max_tokens=500
        )
        
        message = response.choices[0].message
        
        # Check if AI wants to call a function
        if message.tool_calls:
            # AI decided to call the weather function!
            tool_call = message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "get_weather":
                # Call our actual Python function
                location = function_args.get("location", "")
                weather_data = get_weather(location)
                
                # Send weather data back to AI for natural response
                second_response = client.chat.completions.create(
                    model="nvidia/llama-3.1-nemotron-nano-8b-v1",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": None, "tool_calls": [tool_call.model_dump()]},
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": weather_data
                        }
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                # Return AI's natural response
                return second_response.choices[0].message.content
        
        # No function call needed - return AI's direct response
        return message.content
        
    except Exception as e:
        error_msg = str(e).lower()
        
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







# import warnings
# import requests
# import re
# import json
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_core.messages import HumanMessage, SystemMessage
# from decouple import config

# # Suppress warnings
# warnings.filterwarnings('ignore', message='.*type is unknown.*')



# # ============================================
# # SECTION 2: WEATHER FUNCTION (NEW)
# # ============================================

# def get_weather(city_name):
#     """
#     This function gets real weather data from OpenWeatherMap API
    
#     How it works:
#     1. Takes city name as input (e.g., "London")
#     2. Calls Weather API with the city name
#     3. Gets weather data (temperature, condition, etc.)
#     4. Returns formatted weather information
    
#     Example:
#     Input: "London"
#     Output: "Temperature: 15°C, Condition: Cloudy, Humidity: 65%"
#     """
#     try:
#         # Get Weather API key from .env file
#         weather_api_key = config('WEATHER_API_KEY')
#         # Check if API key exists
#         if not weather_api_key:
#             return "Weather API key not found. Please add WEATHER_API_KEY to .env file"
        
#         # Build the API URL
#         # This URL asks OpenWeatherMap for weather data
#         base_url = "http://api.openweathermap.org/data/2.5/weather"
#         params = {
#             'q': city_name,           # City name (e.g., "London")
#             'appid': weather_api_key, # Your API key
#             'units': 'metric'         # Use Celsius (not Fahrenheit)
#         }
        
#         # Call the Weather API
#         response = requests.get(base_url, params=params, timeout=60)
        
#         # Check if API call was successful
#         if response.status_code == 200:
#             # Parse the JSON response
#             data = response.json()
            
#             # Extract weather information
#             temperature = data['main']['temp']           # Temperature in Celsius
#             feels_like = data['main']['feels_like']      # Feels like temperature
#             humidity = data['main']['humidity']          # Humidity percentage
#             description = data['weather'][0]['description']  # Weather condition
#             wind_speed = data['wind']['speed']           # Wind speed
            
#             # Format the weather information nicely
#             weather_info = f"""
# Weather in {city_name}:
# - Temperature: {temperature}°C (Feels like: {feels_like}°C)
# - Condition: {description.capitalize()}
# - Humidity: {humidity}%
# - Wind Speed: {wind_speed} m/s
#             """
            
#             return weather_info.strip()
        
#         elif response.status_code == 404:
#             #return f"City '{city_name}' not found. Please check the spelling."
#             # City not found by name, try geocoding
#             print(f"City '{city_name}' not found by name, trying geocoding...")
            
#             # Get coordinates for the city
#             coords = get_coordinates(city_name)
            
#             # If coordinates found, try weather API with coordinates
#             if coords:
#                 print(f"Found coordinates: {coords['lat']}, {coords['lon']}")
                
                
#                 # Build new request with coordinates
#                 coord_params = {
#                     'lat': coords['lat'],
#                     'lon': coords['lon'],
#                     'appid': weather_api_key,
#                     'units': 'metric'
#                 }
#                 # Try weather API with coordinates
#                 coord_response = requests.get(base_url, params=coord_params, timeout=60)
#                 # Check if successful
#                 if coord_response.status_code == 200:
#                     # Parse the response
#                     data = coord_response.json()
                    
                    
#                     # Extract weather information
#                     temperature = data['main']['temp']
#                     feels_like = data['main']['feels_like']
#                     humidity = data['main']['humidity']
#                     description = data['weather'][0]['description']
#                     wind_speed = data['wind']['speed']
                    
#                     # Format weather info (use the name from geocoding)
#                     weather_info = f"""
# Weather in {coords['name']}:
# - Temperature: {temperature}°C (Feels like: {feels_like}°C)
# - Condition: {description.capitalize()}
# - Humidity: {humidity}%
# - Wind Speed: {wind_speed} m/s
#                     """
                    
#                     return weather_info.strip()
                
#             # If geocoding also failed, show error
#             return f"City '{city_name}' not found. Please check the spelling or try a nearby major city."
        
#         else:
#             return f"Could not fetch weather data. Error code: {response.status_code}"
    
#     except requests.exceptions.Timeout:
#         return "Weather API request timed out. Please try again."
    
#     except requests.exceptions.RequestException as e:
#         return f"Error connecting to Weather API: {str(e)}"
    
#     except Exception as e:
#         return f"Error getting weather: {str(e)}"

# # ============================================
# # SECTION 3: HELPER FUNCTION - Extract City Name
# # ============================================

# def extract_city_name(user_message):
#     """
#     Extract city name from user message
    
#     Examples:
#     "weather in London" -> "London"
#     "What's the temperature in Paris?" -> "Paris"
#     "How's the climate in New York" -> "New York"
#     """
#     # Common patterns to find city names
#     patterns = [
#         r'weather in ([A-Za-z\s]+)',
#         r'temperature in ([A-Za-z\s]+)',
#         r'humidity in ([A-Za-z\s]+)',      # NEW
#         r'rain.*in ([A-Za-z\s]+)',         # NEW
#         r'hot.*in ([A-Za-z\s]+)',          # NEW
#         r'cold.*in ([A-Za-z\s]+)',         # NEW
#         r'forecast for ([A-Za-z\s]+)',
#         r'climate in ([A-Za-z\s]+)',
#         r'weather of ([A-Za-z\s]+)',
#         r'temperature of ([A-Za-z\s]+)',
#     ]
    
#     message_lower = user_message.lower()
    
#     for pattern in patterns:
#         match = re.search(pattern, message_lower)
#         if match:
#             city = match.group(1).strip()
#             # Capitalize each word (e.g., "new york" -> "New York")
#             return city.title()
    
#     # If no pattern matches, try to find the last word(s) as city name
#     # This handles cases like "weather London" or "temperature Paris"
#     words = user_message.split()
#     if len(words) >= 2:
#         return words[-1].title()
    
#     return None

# def get_coordinates(city_name):
#     """
#     Get latitude and longitude for a city using OpenWeatherMap Geocoding API
    
#     This function helps find small cities that the main weather API doesn't recognize.
    
#     Example:
#     Input: "Pulluvazhy"
#     Output: {"lat": 10.1, "lon": 76.4, "name": "Pulluvazhy"}
#     """
#     try:
#         # Get API key
#         weather_api_key = config('WEATHER_API_KEY')
        
#         # Geocoding API URL
#         geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
        
#         # Parameters
#         params = {
#             'q': city_name,           # City to search
#             'limit': 1,               # Get only 1 result (the best match)
#             'appid': weather_api_key  # Same API key
#         }
        
#         # Call Geocoding API
#         response = requests.get(geocoding_url, params=params, timeout=10)
        
#         # Check if successful
#         if response.status_code == 200:
#             data = response.json()
            
#             # Check if we got results
#             if data and len(data) > 0:
#                 return {
#                     'lat': data[0]['lat'],      # Latitude
#                     'lon': data[0]['lon'],      # Longitude
#                     'name': data[0].get('name', city_name)  # Actual city name
#                 }
        
#         # If failed, return None
#         return None
        
#     except Exception as e:
#         return None



# # ============================================
# # SECTION 4: MAIN AI RESPONSE FUNCTION
# # ============================================

# def get_nvidia_response(user_message):
#     """
#     Get AI response using NVIDIA's API
    
#     LOGIC:
#     1. Check if user is asking about weather
#     2. If YES -> Extract city name and call get_weather()
#     3. If NO -> Use simple ChatNVIDIA for normal conversation
#     """
#     try:
#         # Get NVIDIA API key from .env file
#         nvidia_api_key = config('NVIDIA_API_KEY', default='')
        
#         # Validate API key
#         if not nvidia_api_key:
#             return "⚠️ NVIDIA API key not found. Please add NVIDIA_API_KEY to .env file!"
        
#         # ============================================
#         # STEP 1: Check if user is asking about weather
#         # ============================================
#         #weather_keywords = ['weather', 'temperature', 'forecast', 'climate']
#         weather_keywords = [
#             'weather', 'temperature', 'forecast', 'climate',
#             'rain', 'raining', 'rainy', 'rainfall',
#             'humidity', 'humid',
#             'hot', 'cold', 'warm', 'cool',
#             'sunny', 'cloudy', 'cloud', 'clouds',
#             'wind', 'windy',
#             'storm', 'stormy',
#             'snow', 'snowing', 'snowy'
#         ]
#         message_lower = user_message.lower()
        
#         is_weather_question = any(keyword in message_lower for keyword in weather_keywords)
        
#         # ============================================
#         # STEP 2: If weather question -> Call weather function
#         # ============================================
#         if is_weather_question:
#             city_name = extract_city_name(user_message)
            
#             if city_name:
#                 # Call the weather function
#                 weather_data = get_weather(city_name)
#                 return weather_data
#             else:
#                 return "Please specify a city name. For example: 'weather in London'"
        
#         # ============================================
#         # STEP 3: If NOT weather question -> Normal AI chat
#         # ============================================
#         else:
#             # Create the AI model
#             llm = ChatNVIDIA(
#                 model="nvidia/llama-3.1-nemotron-nano-8b-v1",
#                 api_key=nvidia_api_key,
#                 temperature=0.7,
#                 max_tokens=500,
#             )
            
#             # Create messages
#             messages = [
#                 SystemMessage(content="You are a helpful, friendly AI assistant."),
#                 HumanMessage(content=user_message)
#             ]
            
#             # Get AI response
#             response = llm.invoke(messages)
            
#             # Extract and return content
#             if hasattr(response, 'content'):
#                 return response.content
#             else:
#                 return str(response)
        
#     except Exception as e:
#         error_msg = str(e).lower()
        
#         # User-friendly error messages
#         if "401" in error_msg or "unauthorized" in error_msg:
#             return "❌ API Key Error: Your NVIDIA API key is invalid."
        
#         elif "404" in error_msg or "not found" in error_msg:
#             return "❌ Model not found. Please check the model name."
        
#         elif "rate limit" in error_msg or "429" in str(e):
#             return "❌ Too many requests! Please wait 30 seconds and try again."
        
#         elif "timeout" in error_msg:
#             return "❌ Request timed out. Please try a shorter message."
        
#         else:
#             return f"❌ Error: {str(e)[:150]}. Please check your API key and internet connection."