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

