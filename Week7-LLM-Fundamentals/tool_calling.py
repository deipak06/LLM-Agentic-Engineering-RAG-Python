import os
import json
import httpx
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not  GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

# Step1 Define your actual Python funcation

def get_weather(city:str)->str:
    """Fake weather function - replace with real API later"""
    weather_data = {
         "london": "15°C, cloudy",
        "paris": "22°C, sunny",
        "tokyo": "28°C, humid",
        "new york": "18°C, partly cloudy"
    }

    return weather_data.get(city.lower(), f"Weather data not available for {city} ")

#Step2 : Describe the funcation to the model in JSON
tools= [
    {"function_declarations":[

{
    "name":"get_weather",
    "description":"Get the current weather for a city",
    "parameters": {
        "type":"object",
        "properties":{
            "city":{
                "type":"string",
                "description":"The name of the city"
            }
        },
        "required":["city"]
    }
}    
]
}
]




#Step3: Send user message + tool definition to model
messages =[
    {
        "role":"user",
        "parts":[{"text":"What is the weather like in Paris?"}]
    }
] 

response = httpx.post(
    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}",
    headers={"Content-Type": "application/json"},
    json={
        "contents": messages,
        "tools": tools
    },
    timeout=30.0
)
print("Step 1 - Model's first response:")
print(response.json())

response_data= response.json()
function_call= response_data["candidates"][0]["content"]["parts"][0]["functionCall"]
function_name= function_call["name"]
function_args=function_call["args"]

print(f"\nStep2 - MOdel want to call:{function_name}({function_args}) ")

#Step 3 : Execute the actual python function

if function_name == "get_weather":
    tool_result = get_weather(**function_args)

print(f"Step 3 - Tool result: {tool_result}")

#step 4 Send result back to model
messages.append({
    "role":"model",
    "parts":[{"functionCall":{"name":function_name,"args":function_args}}]
})
messages.append({
    "role":"user",
    "parts":[{"functionResponse":{
        "name":function_name,
        "response":{"result":tool_result}
    }}]
})

final_response = httpx.post(
    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}",
    headers={"Content-Type": "application/json"},
    json={
        "contents": messages,
        "tools": tools
    },
    timeout=30.0
)

print("\nStep 4 - Final answer:")
final_answer = final_response.json()["candidates"][0]["content"]["parts"][0]["text"]
print(final_answer)