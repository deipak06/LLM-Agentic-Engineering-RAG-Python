import os
import json
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL="gemini-3.5-flash-lite"

CONFIDENCE_THRESHOLD = 0.55 # below this -> escalate to human
ESCALATION_LOG = "escalations.json"

#---Pydantic models-------------------

class AgentResponse(BaseModel):
    question: str
    answer: str
    confidence_score: float
    escalated: bool
    escalation_reason: Optional[str] = None
    human_answer: Optional[str] = None       # filled in after human reviews
    used_retrieval: bool
    latency_seconds: float

class EscalationRecord(BaseModel):
    question: str
    agent_answer: str
    confidence_score: float
    reason: str
    timestamp: str
    status: str = "pending"                   # pending | resolved
    human_answer: Optional[str] = None

# RAG Setup---------------------------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_and_chunk(filepath: str, chunk_size: int = 3) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [" ".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]

chunks = load_and_chunk("document.txt")
chunk_embeddings = embedding_model.encode(chunks)

def retrieve(query: str, top_k: int = 3) -> list[str]:
    query_emb = embedding_model.encode([query])
    sims = np.dot(chunk_embeddings, query_emb.T).flatten()
    sims = sims / (np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_idx]

def semantic_similarity(text1: str, text2: str) -> float:
    embeddings = embedding_model.encode([text1, text2])
    return float(np.dot(embeddings[0], embeddings[1]) /
                 (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-10))

# ── Confidence scoring ────────────────────────────────────────────────────────

def score_answer_confidence(answer: str, retrieved_context: list[str], question: str) -> tuple[float, str]:
    """
    Score how confident we should be in the agent's answer.
    Returns (score 0-1, reason string).

    Three signals:
    1. If no retrieval was used: assume high confidence (general knowledge)
    2. Similarity between answer and retrieved context
    3. Agent saying "I don't have that information" = low but valid
    """
    if not retrieved_context:
        return 0.90, "direct_knowledge_no_retrieval"

    if "i don't have that information" in answer.lower():
        return 0.85, "appropriate_refusal"

    # Measure: does the answer actually use what was retrieved?
    context_text = " ".join(retrieved_context)
    answer_context_sim = semantic_similarity(answer, context_text)

    # Measure: is the answer relevant to the question?
    answer_question_sim = semantic_similarity(answer, question)

    # Combined score (weighted)
    combined = (answer_context_sim * 0.7) + (answer_question_sim * 0.3)
    reason = f"context_sim={answer_context_sim:.2f}, question_sim={answer_question_sim:.2f}"
    return combined, reason

# ── Escalation log ────────────────────────────────────────────────────────────

def save_escalation(record: EscalationRecord):
    records = load_escalations()
    records.append(record.model_dump())
    with open(ESCALATION_LOG, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n  ⚠️  ESCALATED → saved to {ESCALATION_LOG}")

def load_escalations() -> list[dict]:
    try:
        with open(ESCALATION_LOG, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def simulate_human_review():
    """
    Simulate a human reviewer resolving pending escalations.
    In production this would be a web UI or Slack bot.
    """
    records = load_escalations()
    pending = [r for r in records if r["status"] == "pending"]

    if not pending:
        print("\n  No pending escalations.")
        return

    print(f"\n{'='*60}")
    print(f"HUMAN REVIEW QUEUE ({len(pending)} pending)")
    print(f"{'='*60}")

    for i, record in enumerate(pending):
        print(f"\n[{i+1}] Question: {record['question']}")
        print(f"    Agent said: {record['agent_answer'][:100]}...")
        print(f"    Confidence: {record['confidence_score']:.2f} | Reason: {record['reason']}")
        human_answer = input("    Your answer (or Enter to skip): ").strip()

        if human_answer:
            record["status"] = "resolved"
            record["human_answer"] = human_answer
            print(f"    ✓ Resolved")

    # Save all back
    with open(ESCALATION_LOG, "w") as f:
        json.dump(records, f, indent=2)

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

# ---- Agent -----------------------------------------------------------------------

def run_agent(question:str, history:list=None) -> tuple[AgentResponse, list]:
    if history is None:
        history =[]

    messages = history.copy()
    messages.append({"role":"user","parts":[{"text":question}]})

    retrieved_context: list[str] = []
    used_retrieval = False
    final_answer = ""
    start_time=time.time()

    while True:
        response = client.models.generate_content(
            model=MODEL,
            contents=messages,
            config=types.GenerateContentConfig(tools=TOOL_CONFIG)
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts
        has_tool_call = False

        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                has_tool_call = True
                used_retrieval = True
                query = part.function_call.args["query"]
                results = retrieve(query)
                retrieved_context = results
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

        # ── Confidence scoring ─────────────────────────────────────────────────
    confidence, reason = score_answer_confidence(final_answer, retrieved_context, question)
    escalated = confidence < CONFIDENCE_THRESHOLD
    escalation_reason = reason if escalated else None

    if escalated:
        record = EscalationRecord(
            question=question,
            agent_answer=final_answer,
            confidence_score=confidence,
            reason=reason,
            timestamp=time.strftime("%Y-%m-%d %Human:%M:%S")
        )
        save_escalation(record)

    # ── Update conversation history ────────────────────────────────────────
    updated_history = history.copy()
    updated_history.append({"role": "user", "parts": [{"text": question}]})
    updated_history.append({"role": "model", "parts": [{"text": final_answer}]})

    return AgentResponse(
        question=question,
        answer=final_answer,
        confidence_score=confidence,
        escalated=escalated,
        escalation_reason=escalation_reason,
        used_retrieval=used_retrieval,
        latency_seconds=latency
    ), updated_history


# ── Print helper ──────────────────────────────────────────────────────────────

def print_response(r: AgentResponse):
    status = "⚠️  ESCALATED" if r.escalated else "✓  AUTO-ANSWERED"
    print(f"\n{'─'*60}")
    print(f"Q: {r.question}")
    print(f"A: {r.answer[:150]}...")
    print(f"   {status}")
    print(f"   Confidence: {r.confidence_score:.2f} | Retrieval: {r.used_retrieval} | {r.latency_seconds:.2f}s")
    if r.escalation_reason:
        print(f"   Reason: {r.escalation_reason}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Phase 1: Run agent on test questions
    questions = [
        "How many days of annual leave do I get?",
        "Can I carry them over to next year?",
        "What is the company's policy on drone usage at the office?",   # not in document
        "What are the WFH rules?",
        "What is 25 multiplied by 4?",
        "What happens if I violate the data breach policy?",            # vague, not in document
    ]

    print("PHASE 1: Agent running...")
    logs = []
    history = []
    for q in questions:
        response, history = run_agent(q, history)
        print_response(response)
        logs.append(response)

    escalated = [r for r in logs if r.escalated]
    print(f"\n\nSummary: {len(logs) - len(escalated)}/{len(logs)} auto-answered, {len(escalated)} escalated")

    # Phase 2: Human review of escalations
    if escalated:
        print("\n\nPHASE 2: Human review")
        simulate_human_review()

    # Phase 3: Show what was resolved
    print("\n\nPHASE 3: Resolved escalations")
    records = load_escalations()
    resolved = [r for r in records if r["status"] == "resolved"]
    if resolved:
        for r in resolved:
            print(f"\n  Q: {r['question']}")
            print(f"  Human answer: {r['human_answer']}")
    else:
        print("  None resolved yet.")