
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

# Read Gemini API Key
API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client 
#we make validation to check if the prompt is empty or not, if it is empty we raise a ValueError
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

client = genai.Client(api_key=API_KEY)

def generate_response(prompt):
    """Send a prompt to Gemini and return its response."""

    if not prompt or not prompt.strip():#none || space
        raise ValueError("Prompt cannot be empty.")

    response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=prompt,
    )

    return response.text
