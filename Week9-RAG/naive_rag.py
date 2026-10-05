import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path
from google import genai

load_dotenv(dotenv_path=Path(__file__).parent /".env")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI KEY not found")

#Step1 : Load and chunk the document

def load_and_chunk(filepath:str,chunk_size:int = 3) -> list[str]:
    """Split document into chunks of N lines each"""
    with open(filepath,"r",encoding="utf-8") as f:
        lines=[line.strip() for line in f.readlines() if line.strip()]
    chunks=[]
    for i in range(0,len(lines),chunk_size):
        chunk=" ".join(lines[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

#Step2 : Embed all chunks
def embed_chunks(chunks:list[str], model:SentenceTransformer) -> np.ndarray:
    """Convert chunks to vectors"""
    return model.encode(chunks)

#Step 3 : Find the most similar chunk to a query
def retrieve(query:str, chunks:list[str],chunk_embeddings:np.ndarray,
             model: SentenceTransformer, top_k:int=3) -> list[str]:
    """Find top_k most relevant chunks for the query"""
    query_embedding= model.encode([query])

    #Cosine similarity between query and all chunks
    similarities=np.dot(chunk_embeddings, query_embedding.T).flatten()
    similarities = similarities/(
        np.linalg.norm(chunk_embeddings,axis=1) * np.linalg.norm(query_embedding)
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

#Step 4 : Generate answer using retrived chunks
def generate_answer(query:str,context_chunks:list[str])->str:
    """Send query + context to LLM"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    context = "\n\n".join(context_chunks)
    # prompt = f"""Answer the question based on the context below.
    # if the answer is not in the context, say "I don't have that infromation."
    # Context:
    # {context}
    # Question:{query}
    
    # Answer:"""
    prompt = f"""You are a helpful HR assistant. Answer the question using ONLY the information provided in the context below.
    - Answer in complete sentences
    - Include all relevant details from the context
    - If you can directly answer or reasonably infer the answer from the context, provide it
    - If the answer cannot be found or inferred from the context, say exactly: "I don't have that information."
    - Do not add information not present in the context

    Context:
    {context}

    Question: {query}

    Answer:"""
    reponses=client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={"temperature":0}
    )
    return reponses.text

#Main RAG Pipeline

def ask(question:str,chunks:list[str], 
    chunk_embeddings:np.ndarray, model:SentenceTransformer)->str:
    print(f"\n Question:{question} ")
    retrieved= retrieve(question,chunks,chunk_embeddings,model)
    print(f"Retrieved chunks:")
    for i, chunk in enumerate(retrieved):
        print(f"[{i+1}]{chunk[:100]}...")
    answer = generate_answer(question,retrieved)
    print(f"Answer:{answer}")
    return answer



# Test questions
# ask("How many days of annual leave do employees get?", 
#    chunks, chunk_embeddings, embedding_model)

# ask("Can I work from home every day?", 
#    chunks, chunk_embeddings, embedding_model)

# ask("What happens if I lose a receipt for a $30 expense?", 
#    chunks, chunk_embeddings, embedding_model)

# ask("What is the weather like today?",  # intentional out-of-scope question
#    chunks, chunk_embeddings, embedding_model)

# #ask("What are ALL the different types of leave available?",
# #    chunks, chunk_embeddings, embedding_model)

# # ask("I submitted an expense 35 days ago without a receipt, what are the issues?",
# #     chunks, chunk_embeddings, embedding_model)

# # ask("What is the maximum number of days I can carry over AND what is the sick leave limit?",
# #     chunks, chunk_embeddings, embedding_model)

# # Failure mode 1: answer split across chunk boundary
# ask("What is the carry over limit and when must leave be requested by?",
#     chunks, chunk_embeddings, embedding_model)

# # Failure mode 2: vocabulary mismatch
# ask("What are the WFH rules?",   # "WFH" not in document, only "remote work"
#     chunks, chunk_embeddings, embedding_model)

# # Failure mode 3: ambiguous question with multiple possible matches
# ask("What are the limits?",   # too vague — limits on what?
#     chunks, chunk_embeddings, embedding_model)


# With:
if __name__ == "__main__":
    # Initialize
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading and chunking document...")
    chunks=load_and_chunk("document.txt")
    print(f"Created {len(chunks)} chunks")

    print("Embedding chunks...")
    chunk_embeddings = embed_chunks(chunks, embedding_model)
    print(f"Embedded {len(chunk_embeddings)} chunks")
    ask("How many days of annual leave do employees get?", 
        chunks, chunk_embeddings, embedding_model)
    ask("Can I work from home every day?", 
        chunks, chunk_embeddings, embedding_model)
    ask("What happens if I lose a receipt for a $30 expense?", 
        chunks, chunk_embeddings, embedding_model)
    ask("What is the weather like today?",
        chunks, chunk_embeddings, embedding_model)