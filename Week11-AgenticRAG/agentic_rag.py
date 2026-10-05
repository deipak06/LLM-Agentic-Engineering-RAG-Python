import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

#----------RAG setup(loaded ONCE at startup)----------
def load_and_chunk(filepath:str, chunk_size: int=3) -> list[str]:
    with open(filepath,"r", encoding="utf-8") as f:
        lines=[line.strip() for line in f.readlines() if line.strip()]
    chunks =[]
    for i in range(0,len(lines),chunk_size):
        chunks.append(" ".join(lines[i:i + chunk_size]))
    return chunks

def embed_chunks(chunks:list[str],model:SentenceTransformer) -> np.ndarray:
    return model.encode(chunks)

def retrieve(query:str,chunks:list[str],
             chunk_embeddings: np.ndarray, model:SentenceTransformer, top_k:int=3) -> list[str]:
    query_emb = model.encode([query])
    sims =np.dot(chunk_embeddings, query_emb.T).flatten()
    sims = sims/(np.linalg.norm(chunk_embeddings,axis=1)* np.linalg.norm(query_emb))
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_idx]

print("Loading embedding model.....")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Loading document...")
chunks = load_and_chunk("document.txt")
chunk_embeddings = embed_chunks(chunks, embedding_model)
print(f"Ready - {len(chunks)} chunks embedded\n")

#----- The RAG tool function--------------------------------------------------------------

def search_knowledge_base(query:str) -> str:
    """Search company HR documents and return relevant context"""
    retrieved=retrieve(query, chunks, chunk_embeddings,embedding_model)
    #Format retrieved chunk as numbered context

    context=""
    for i,chunk in enumerate(retrieved):
        context += f"[{i+1}]{chunk}\n\n"
    return f"Retrieved context for '{query}':\n\n{context}"

#----Tool registry and definitions
TOOLS={
    "search_knowledge_base": search_knowledge_base
}

tool_definitions = [
    {
        "function_declarations": [
            {
                "name": "search_knowledge_base",
                "description": "Search the company HR knowledge base for information about leave policies, remote work, expenses, performance reviews, and IT security. Use this tool when the user asks about company policies or HR-related topics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant HR policy information"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    }
]

#---Agent loop-----------------------------------

client = genai.Client(api_key=GEMINI_API_KEY)

# def run_agentic_rag(user_question: str) -> str:
#     print(f"\nQuestion: {user_question}")
#     messages = [
#         {
#             "role": "user",
#             "parts": [{"text": user_question}]
#         }
#     ]

#     while True:
#         response = client.models.generate_content(
#             model="gemini-3.5-flash-lite",
#             contents=messages,
#             config={"tools": tool_definitions}
#         )

#         candidate = response.candidates[0]
#         parts = candidate.content.parts
#         has_tool_call = False

#         for part in parts:
#             if hasattr(part, "function_call") and part.function_call:
#                 has_tool_call = True
#                 function_name = part.function_call.name
#                 function_args = dict(part.function_call.args)

#                 print(f"  → Searching: '{function_args.get('query', '')}'")
#                 tool_result = TOOLS[function_name](**function_args)
#                 print(f"  → Retrieved {len(tool_result)} chars of context")

#                 # messages.append({
#                 #     "role": "model",
#                 #     "parts": [{"function_call": {
#                 #         "name": function_name,
#                 #         "args": function_args
#                 #     }}]
#                 # })
#                 # With this — append the SDK's native content object:
#                 messages.append(candidate.content)
#                 messages.append({
#                     "role": "user",
#                     "parts": [{"function_response": {
#                         "name": function_name,
#                         "response": {"result": tool_result}
#                     }}]
#                 })

#         if not has_tool_call:
#             text_part = next(
#                 (p for p in parts if hasattr(p, "text") and p.text), None
#             )
#             if text_part:
#                 return text_part.text
#             return "No response generated"

# # ── Test questions ───────────────────────────────────────────

# if __name__ == "__main__":
#     questions = [
#         "How many days of annual leave do I get?",      # should search
#         "What is 15 multiplied by 4?",                   # should NOT search
#         "What are the remote work rules?",               # should search
#         "What is the capital of France?",                # should NOT search
#         "Do I need a receipt for a $30 business meal?",  # should search
#     ]

#     for q in questions:
#         answer = run_agentic_rag(q)
#         print(f"Answer: {answer}\n")
#         print("-" * 50)

def run_agentic_rag(user_question:str, history:list = None) -> tuple[str,list]:
    if history is None:
        history=[]
    print(f"\n Question:{user_question}")

    #Start with history + current question
    messages = history.copy()
    messages.append({
        "role":"user",
        "parts":[{"text":user_question}]
    })

    final_answer =""

    while True:
        response= client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=messages,
            config={"tools":tool_definitions}
        )
         

        candidate = response.candidates[0]
        parts= candidate.content.parts
        has_tool_call = False


        for i, part in enumerate(parts):
            print(f"  DEBUG part[{i}]: "
                  f"text={getattr(part, 'text', None)[:50] if getattr(part, 'text', None) else None}, "
                  f"function_call={getattr(part, 'function_call', None)}")



        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                has_tool_call = True
                function_name = part.function_call.name
                function_args= dict(part.function_call.args)

                print(f" -> Searching :'{function_args.get('query','')}'")

                tool_result = TOOLS[function_name](**function_args)
                print(f" -> Retrieved {len(tool_result)} chars of context")

                messages.append(candidate.content)
                messages.append({
                    "role":"user",
                    "parts":[{"function_response":
                              {
                                  "name":function_name,
                                  "response":{"result":tool_result}
                              }}]
                })

        if not has_tool_call:
            text_part = next(
                (p for p in parts if hasattr(p, "text") and p.text), None)
            if text_part:
                    final_answer = text_part.text
                    break
            
    
# Update history with ONLY user question + final answer
    updated_history = history.copy()
    updated_history.append({
        "role": "user",
        "parts": [{"text": user_question}]
    })
    updated_history.append({
        "role": "model",
        "parts": [{"text": final_answer}]
    })
    
    return final_answer, updated_history


# Test multi-turn conversation
if __name__ == "__main__":
    history = []
    
    answer1, history = run_agentic_rag(
        "How many days of annual leave do I get?", history)
    print(f"A1: {answer1}\n")
    print("-" * 50)
    
    answer2, history = run_agentic_rag(
        "Can I carry them over?", history)
    print(f"A2: {answer2}\n")
    print("-" * 50)
    
    answer3, history = run_agentic_rag(
        "What about sick leave?", history)
    print(f"A3: {answer3}\n")
    print("-" * 50)
    
    # Bonus: out-of-scope follow-up
    answer4, history = run_agentic_rag(
        "What is the capital of France?", history)
    print(f"A4: {answer4}\n")