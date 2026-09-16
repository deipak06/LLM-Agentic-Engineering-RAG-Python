import os
from dotenv import load_dotenv
from pathlib import Path
from google import genai

load_dotenv(dotenv_path=Path(__file__).parent/".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Gemini_API_Key not found")

client = genai.Client(api_key=GEMINI_API_KEY)

print("Streaming response:")
for chunk in client.models.generate_content_stream(
    model="gemini-2.5-flash-lite",
    contents="Write a short 3 sentence story about a robot learning to paint."
):
    print(chunk.text,end="",flush=True)

print("\n---Done----")