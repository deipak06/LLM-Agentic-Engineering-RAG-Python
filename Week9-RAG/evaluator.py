import os
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

#Import your RAG Pipeline

from naive_rag import load_and_chunk, embed_chunks, retrieve,generate_answer

#Eval Set
# EVAL_SET = [
#     {
#         "question": "How many days of annual leave do employees get?",
#         "expected": "20 days"
#     },
#     {
#         "question": "How many days can I carry over?",
#         "expected": "10 days"
#     },
#     {
#         "question": "Can I work from home every day?",
#         "expected": "No, maximum 3 days per week"
#     },
#     {
#         "question": "What are the WFH rules?",
#         "expected": "3 days per week, manager approval, available 10am-3pm"
#     },
#     {
#         "question": "What happens if I lose a receipt for a $30 expense?",
#         "expected": "Receipt required for expenses over $25"
#     },
#     {
#         "question": "How often are performance reviews?",
#         "expected": "Twice a year, June and December"
#     },
#     {
#         "question": "How often must I change my password?",
#         "expected": "Every 90 days"
#     },
#     {
#         "question": "What is the weather like today?",
#         "expected": "I don't have that information"
#     }
# ]

EVAL_SET = [
    {
        "question": "How many days of annual leave do employees get?",
        "expected": "Employees are entitled to 20 days of annual leave per year"
    },
    
    {
        "question": "How many days can I carry over?",
        "expected": "Unused leave can be carried over to the next year, up to a maximum of 10 days"
    },
    
    {
        "question": "Can I work from home every day?",
        "expected": "Employees may work remotely up to 3 days per week"
    },
    
    {
        "question": "What are the WFH rules?",
        "expected": "Remote work must be approved by the direct manager and employees must be available during core hours 10am to 3pm"
    },
    
    # {
    #     "question": "What happens if I lose a receipt for a $30 expense?",
    #     "expected": "Receipts are required for any expense over $25"
    # },
    {
    "question": "Do I need a receipt for a $30 expense?",
    "expected": "Receipts are required for any expense over $25"
    },
    
    {
        "question": "How often are performance reviews?",
        "expected": "Performance reviews are conducted twice a year in June and December"
    },
    
    {
        "question": "How often must I change my password?",
        "expected": "All passwords must be changed every 90 days"
    },
    
    {
        "question": "What is the weather like today?",
        "expected": "I don't have that information"
    }
    
]

def semantic_similarity(text1:str, text2:str, model:SentenceTransformer) ->float:
    """Compare two texts using cosine similarity"""
    embeddings = model.encode([text1,text2])
    similarity= np.dot(embeddings[0],embeddings[1])/(np.linalg.norm(embeddings[0])* np.linalg.norm(embeddings[1])
    )
    return float(similarity)

def run_eval(chunks, chunk_embeddings, embedding_model, threshold:float=0.7,label:str="Naive RAG"):
    print(f"\n{'='*50}")
    print(f"Evaluation:{label}")
    print(f"{'='*50}")

    passed=0
    results=[]

    for item in EVAL_SET:
        question=item["question"]
        expected=item["expected"]

        # Get RAG Answer
        retrieved= retrieve(question,chunks,chunk_embeddings,embedding_model)
        if "receipt" in question.lower():
            context = "\n\n".join(retrieved)
            print(f"\n--- FULL CONTEXT SENT TO LLM ---")
            print(context)
            print(f"--- END CONTEXT ---\n") 
        print(f"  Debug retrieved: {[c[:60] for c in retrieved]}") 
        actual=generate_answer(question,retrieved)

        #Score using semantic similarity
        score= semantic_similarity(expected,actual,embedding_model)
        passed_check=score>=threshold

        if passed_check:
            passed+=1

        results.append({
            "question":question,
            "expected":expected,
            "actual":actual.strip(),
            "score":score,
            "passed":passed_check
        })
        status=" PASS" if passed_check else "FAIL"
        print(f"\n{status}(score:{score:.2f})")
        print(f" Q:{question}")
        print(f" Expected:{expected}")
        print(f" Got:{actual.strip()[:150]}")
    print(f"\n{'='*50}")
    print(f"Score:{passed}/{len(EVAL_SET)}({passed/len(EVAL_SET)*100:.0f}%)")
    print(f"{'='*50}")

    return results

# Run evaluations

print("Loading models and documents....")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chunks= load_and_chunk("document.txt")
chunk_embeddings = embed_chunks(chunks,embedding_model)

run_eval(chunks,chunk_embeddings,embedding_model,label="Naive RAG Baseline")

print("\n--- All chunks ---")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}")
