"""Quick test script for Gemini API connection."""
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load .env from project directory
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
load_dotenv(env_path)

import google.generativeai as genai

# Configure with API key
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env file")
    sys.exit(1)

print(f"API Key loaded: {api_key[:10]}...{api_key[-4:]}")

genai.configure(api_key=api_key)

# Test chat model
print("\nTesting gemini-2.0-flash...")
model = genai.GenerativeModel('gemini-2.0-flash')
response = model.generate_content('Say "Hello, RCA system!" in exactly those words.')
print(f"Chat response: {response.text}")

# Test embedding model
print("\nTesting text-embedding-004...")
result = genai.embed_content(
    model="models/text-embedding-004",
    content="ERROR: Connection timeout to database service"
)
embedding = result['embedding']
print(f"Embedding dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

print("\n✓ Gemini API connection successful!")
