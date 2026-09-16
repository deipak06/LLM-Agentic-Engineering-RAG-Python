import os
import httpx
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent /".env")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI API KEY not found")

#Tool 1: Weather
def get_weather(city:str) -> str:
    weather_data= {
        "london": "15°C, cloudy",
        "paris": "22°C, sunny",
        "tokyo": "28°C, humid",
        "new york": "18°C, partly cloudy"
    }
    return weather_data.get(city.lower(),f"No weather data for {city}")

#Tool 2 : Calculator
def calculator(expression:str)->str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Calculation error:{e}"

#Tool 3: Simple fact lookup
def get_fact(topic:str) -> str:

    facts= { 
        "python": "Python was created by Guido van Rossum in 1991.",
        "llm": "Large Language Models are trained on vast amounts of text data.",
        "paris": "Paris is the capital of France, with a population of about 2.1 million.",
        "tokyo": "Tokyo is the capital of Japan and the world's most populous city."
    }
    return facts.get(topic.lower(),f"No fact avaiable for {topic}")

#Tool registary - maps name to actual function
TOOLS = {
    "get_weather":get_weather,
    "calculate":calculator,
    "get_fact":get_fact

}

#Tool definitions for model
tool_definitions = [
{
    "function_declarations": [
            {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            },
            {
                "name": "calculate",
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression to evaluate e.g. '2 + 2'"}
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "get_fact",
                "description": "Get a fact about a topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic to get a fact about"}
                    },
                    "required": ["topic"]
                }
            }
        ]
    }
]

def run_agent(user_message:str) -> str:
    messages = [
        {
            "role":"user",
            "parts":[{"text":user_message}]
        }
    ]

# Loop until model stops requesting tools
    while True:
    # Step 1: call the API with current messages + tools
        response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": messages,
            "tools":tool_definitions
        },
        timeout=30.0
    )
        response_data=response.json()
        candidate = response_data["candidates"][0]
        part= candidate["content"]["parts"][0]
        

    #Step2 :check if model wants a tool call or is done
        if "functionCall" in part:
            function_name= part["functionCall"]["name"]
            function_args = part["functionCall"]["args"]
            tool_result=TOOLS[function_name](**function_args)
            print(f"Tool called: {function_name}({function_args}) → {tool_result}")

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
        else:
            return part["text"]

print(run_agent("What is the weather in Paris and Tokyo, and what is the sum of their temperatures in Celsius?"))
    
