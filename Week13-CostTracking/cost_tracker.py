import os
import time
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash-lite"

# Pricing (per million tokens)
INPUT_PRICE_PER_M = 0.075
OUTPUT_PRICE_PER_M = 0.30

# ── Pydantic models ──────────────────────────────────────────────────────────

class ApiCallLog(BaseModel):
    call_number: int
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float

class QuestionLog(BaseModel):
    question: str
    answer: str
    used_retrieval: bool
    api_calls: list[ApiCallLog]
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    latency_seconds: float

class SessionSummary(BaseModel):
    total_questions: int
    retrieval_questions: int
    direct_questions: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    avg_cost_per_question_usd: float
    avg_latency_seconds: float
    total_api_calls: int

# ── RAG setup ────────────────────────────────────────────────────────────────

def load_and_chunk(filepath: str, chunk_size: int = 3) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [" ".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = load_and_chunk("document.txt")
chunk_embeddings = embedding_model.encode(chunks)

def retrieve(query: str, top_k: int = 3) -> list[str]:
    query_emb = embedding_model.encode([query])
    sims = np.dot(chunk_embeddings, query_emb.T).flatten()
    sims = sims / (np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_idx]

# ── Cost helpers ─────────────────────────────────────────────────────────────

def calc_cost(input_tokens: int, output_tokens: int) -> tuple[float, float, float]:
    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
    return input_cost, output_cost, input_cost + output_cost

def log_api_call(response, call_number: int) -> ApiCallLog:
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    input_cost, output_cost, total_cost = calc_cost(input_tokens, output_tokens)
    return ApiCallLog(
        call_number=call_number,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=total_cost
    )

# ── Tool definition ───────────────────────────────────────────────────────────

TOOL_CONFIG = [types.Tool(function_declarations=[{
    "name": "search_knowledge_base",
    "description": "Search the HR knowledge base for company policies on leave, remote work, expenses, performance, and IT security. Do NOT use for general knowledge questions.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
}])]

# ── Agent with cost tracking ──────────────────────────────────────────────────

def run_agent(question: str, history: list = None) -> tuple[QuestionLog, list]:
    if history is None:
        history = []

    messages = history.copy()
    messages.append({"role": "user", "parts": [{"text": question}]})

    api_calls: list[ApiCallLog] = []
    used_retrieval = False
    final_answer = ""
    call_number = 0
    start_time = time.time()

    while True:
        call_number += 1
        response = client.models.generate_content(
            model=MODEL,
            contents=messages,
            config=types.GenerateContentConfig(tools=TOOL_CONFIG)
        )

        api_calls.append(log_api_call(response, call_number))

        candidate = response.candidates[0]
        parts = candidate.content.parts
        has_tool_call = False

        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                has_tool_call = True
                used_retrieval = True
                query = part.function_call.args["query"]
                results = retrieve(query)
                context = "\n\n".join(results)

                messages.append(candidate.content)
                messages.append({
                    "role": "user",
                    "parts": [{"function_response": {
                        "name": "search_knowledge_base",
                        "response": {"result": context}
                    }}]
                })

        if not has_tool_call:
            text_part = next((p for p in parts if hasattr(p, "text") and p.text), None)
            if text_part:
                final_answer = text_part.text
            break

    latency = time.time() - start_time
    total_input = sum(c.input_tokens for c in api_calls)
    total_output = sum(c.output_tokens for c in api_calls)
    _, _, total_cost = calc_cost(total_input, total_output)

    log = QuestionLog(
        question=question,
        answer=final_answer,
        used_retrieval=used_retrieval,
        api_calls=api_calls,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cost_usd=total_cost,
        latency_seconds=latency
    )

    updated_history = history.copy()
    updated_history.append({"role": "user", "parts": [{"text": question}]})
    updated_history.append({"role": "model", "parts": [{"text": final_answer}]})

    return log, updated_history

# ── Print helpers ─────────────────────────────────────────────────────────────

def print_question_log(log: QuestionLog):
    print(f"\n{'─'*60}")
    print(f"Q: {log.question}")
    print(f"A: {log.answer[:120]}...")
    print(f"   Retrieval used: {log.used_retrieval} | API calls: {len(log.api_calls)}")
    for c in log.api_calls:
        print(f"   Call {c.call_number}: {c.input_tokens} in / {c.output_tokens} out → ${c.total_cost_usd:.6f}")
    print(f"   TOTAL: {log.total_input_tokens} in / {log.total_output_tokens} out → ${log.total_cost_usd:.6f} | {log.latency_seconds:.2f}s")

def print_session_summary(logs: list[QuestionLog]):
    total_q = len(logs)
    retrieval_q = sum(1 for l in logs if l.used_retrieval)
    total_in = sum(l.total_input_tokens for l in logs)
    total_out = sum(l.total_output_tokens for l in logs)
    _, _, total_cost = calc_cost(total_in, total_out)
    avg_cost = total_cost / total_q if total_q > 0 else 0
    avg_latency = sum(l.latency_seconds for l in logs) / total_q if total_q > 0 else 0
    total_calls = sum(len(l.api_calls) for l in logs)

    summary = SessionSummary(
        total_questions=total_q,
        retrieval_questions=retrieval_q,
        direct_questions=total_q - retrieval_q,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_cost_usd=total_cost,
        avg_cost_per_question_usd=avg_cost,
        avg_latency_seconds=avg_latency,
        total_api_calls=total_calls
    )

    print(f"\n{'='*60}")
    print("SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"Questions:    {summary.total_questions} ({summary.retrieval_questions} with retrieval, {summary.direct_questions} direct)")
    print(f"API calls:    {summary.total_api_calls} total")
    print(f"Tokens:       {summary.total_input_tokens:,} input / {summary.total_output_tokens:,} output")
    print(f"Total cost:   ${summary.total_cost_usd:.6f}")
    print(f"Avg/question: ${summary.avg_cost_per_question_usd:.6f}")
    print(f"Avg latency:  {summary.avg_latency_seconds:.2f}s")
    print(f"\nAt 1,000 questions/day: ~${summary.avg_cost_per_question_usd * 1000:.4f}/day")
    print(f"At 1,000 questions/day: ~${summary.avg_cost_per_question_usd * 30000:.2f}/month")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    questions = [
        "How many days of annual leave do I get?",
        "Can I carry them over to next year?",
        "What is 25 multiplied by 4?",
        "What are the WFH rules?",
        "What is the capital of Japan?",
        "How often do I need to change my password?",
    ]

    logs = []
    history = []

    for q in questions:
        log, history = run_agent(q, history)
        print_question_log(log)
        logs.append(log)

    print_session_summary(logs)