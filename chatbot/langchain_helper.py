"""
LangChain integration helper
Uncomment and use this when you have an OpenAI API key
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage  # CHANGED THIS LINE
import os

def get_langchain_response(user_message):
    """
    Get response using LangChain and OpenAI
    """
    try:
        # Initialize the model
        chat = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Create messages
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=user_message)
        ]
        
        # Get response
        response = chat.invoke(messages)
        return response.content
        
    except Exception as e:
        return f"Error: {str(e)}"




# from langchain_openai import ChatOpenAI
# from langchain.schema import HumanMessage, SystemMessage
# import os
# from decouple import config

# def get_langchain_response(user_message):
#     """
#     Get response using LangChain and OpenAI
#     """
#     try:
#         # Initialize the model
#         chat = ChatOpenAI(
#             temperature=0.7,
#             model_name="gpt-3.5-turbo",
#             openai_api_key=config('OPENAI_API_KEY')
#         )
        
#         # Create messages
#         messages = [
#             SystemMessage(content="You are a helpful AI assistant."),
#             HumanMessage(content=user_message)
#         ]
        
#         # Get response
#         response = chat.invoke(messages)
#         return response.content
        
#     except Exception as e:
#         return f"Error: {str(e)}"