import os
import httpx
from dotenv import load_dotenv
from pathlib import Path

#load_dotenv(dotenv_path=r"D:\python\Week7-LLM-Fundamentals\.env")
# Find .env relative to this script's location
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

conversation_history = []

def chat(user_message:str) -> str:
    #step 1: add user message to history
    conversation_history.append(
        {
            "role":"user",
            "parts":[{"text":user_message}]
        }
    )

    #step 2 : send entire history to API
    response = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={GEMINI_API_KEY}",
                headers ={"Content-Type":"application/json"},
                json={"contents":conversation_history},
                timeout=60.0
    )
    

    #step 3 : extract the reply
    
    if response.status_code != 200:
        print(f"API error: {response.json()}")
        return None
    
    reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    
    #step 4 :add assistant reply to history
    conversation_history.append(
        {
            "role":"model",
            "parts":[{"text":reply}]
        }
    )


    return reply

print(chat("My name is Nitin and I am learning AI engineering."))
print("---")
print(chat("What is my name?"))
print("---")
print(chat("What am I learning?"))
print("---")
print(chat("Summarize our conversation so far in one sentence."))