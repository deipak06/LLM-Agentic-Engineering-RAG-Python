import streamlit as st
import os
import json
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

#---Config-----------------------------------------------------------

load_dotenv(dotenv_path=Path(__file__).parent /".env")

API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY","")
MODEL = "gemini-3.5-flash-lite"
CONFIDENCE_THRESHOLD=0.55

# ── HR Knowledge Base ────────────────────────────────────────────────────────
HR_DOCUMENTS = [
    "Annual leave entitlement is 20 days per year for full-time employees.",
    "Sick leave allowance is 10 days per year. A doctor's certificate is required for absences longer than 3 consecutive days.",
    "The probation period for new employees is 3 months.",
    "Remote work policy: Employees may work from home up to 3 days per week with manager approval.",
    "Performance reviews are conducted twice a year: in June and December.",
    "The notice period for resignation is 1 month for employees under 2 years, and 2 months for employees over 2 years.",
    "Overtime pay is 1.5x the standard hourly rate for weekdays and 2x for weekends and public holidays.",
    "Training budget: Each employee receives $1,000 per year for professional development.",
    "Health insurance covers the employee and up to 3 dependants.",
    "The company observes 12 public holidays per year.",
    "Maternity leave is 16 weeks fully paid. Paternity leave is 2 weeks fully paid.",
    "Annual leave must be applied at least 5 working days in advance and approved by the line manager.",
    "Employees are entitled to carry over up to 5 unused leave days to the following year.",
    "The dress code is business casual Monday to Thursday. Casual Fridays are permitted.",
    "Disciplinary procedures follow a 3-step process: verbal warning, written warning, termination.",
]

#---Pydantic Models-----------------------------------------------

class AgentResponse(BaseModel):
    question:str
    answer:str
    confidence_score:float
    escalated:bool
    escalation_reason:Optional[str]=None
    used_retrieval:bool
    latency_seconds: float

class ChatMessage(BaseModel):
    role:str #"user" or "assistant"
    content:str

#----Init (cached so it runs once, not every rerun)-----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_kb_embeddings():
    embedder = load_embedder()
    return embedder.encode(HR_DOCUMENTS, convert_to_numpy=True)

@st.cache_resource
def get_client():
    return genai.Client(api_key=API_KEY)


# ── RAG Helpers ──────────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def retrieve_context(question: str, top_k: int = 3) -> tuple[list[str], bool]:
    embedder = load_embedder()
    kb_embeddings = load_kb_embeddings()
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    scores = [cosine_similarity(q_emb, doc_emb) for doc_emb in kb_embeddings]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_score = scores[top_indices[0]]
    if top_score < 0.3:
        return [], False
    return [HR_DOCUMENTS[i] for i in top_indices], True

def score_confidence(answer: str, context: list[str], question: str) -> tuple[float, str]:
    if not context:
        return 0.90, "direct_knowledge"
    if "i don't have" in answer.lower() or "i cannot" in answer.lower():
        return 0.85, "appropriate_refusal"
    embedder = load_embedder()
    ans_emb = embedder.encode([answer], convert_to_numpy=True)[0]
    ctx_text = " ".join(context)
    ctx_emb = embedder.encode([ctx_text], convert_to_numpy=True)[0]
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    ctx_sim = cosine_similarity(ans_emb, ctx_emb)
    q_sim = cosine_similarity(ans_emb, q_emb)
    score = (ctx_sim * 0.7) + (q_sim * 0.3)
    return round(score, 3), "retrieval_scored"

# ── Agent ────────────────────────────────────────────────────────────────────
def run_agent(question: str, history: list[ChatMessage]) -> AgentResponse:
    client = get_client()
    start = time.time()

    context_chunks, used_retrieval = retrieve_context(question)

    system_prompt = """You are an HR assistant for a company. Answer employee questions 
about HR policies clearly and concisely. If the context provided contains the answer, 
use it. If the question is outside HR policy scope, say so politely."""

    if context_chunks:
        context_block = "\n\nRelevant HR Policy:\n" + "\n".join(f"- {c}" for c in context_chunks)
    else:
        context_block = ""

    messages = []
    for msg in history[-6:]:   # last 3 turns to keep context manageable
        messages.append({"role": msg.role, "parts": [{"text": msg.content}]})
    messages.append({"role": "user", "parts": [{"text": question + context_block}]})

    response = client.models.generate_content(
        model=MODEL,
        contents=messages,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    answer = response.text.strip()
    confidence, reason = score_confidence(answer, context_chunks, question)
    escalated = confidence < CONFIDENCE_THRESHOLD

    return AgentResponse(
        question=question,
        answer=answer,
        confidence_score=confidence,
        escalated=escalated,
        escalation_reason=reason if escalated else None,
        used_retrieval=used_retrieval,
        latency_seconds=round(time.time() - start, 2),
    )

# ── Streamlit UI ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="HR Chatbot", page_icon="💼", layout="centered")
    st.title("💼 HR Policy Chatbot")
    st.caption("Ask me anything about company HR policies.")

    if not API_KEY:
        st.error("GEMINI_API_KEY not found. Add it to .env (local) or Streamlit secrets (cloud).")
        st.stop()

    # Session state — survives reruns for this user's session
    if "messages" not in st.session_state:
        st.session_state.messages = []   # list of ChatMessage
    if "escalations" not in st.session_state:
        st.session_state.escalations = []

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg.role):
            st.write(msg.content)

    # Chat input
    user_input = st.chat_input("Ask an HR question...")
    if user_input:
        # Show user message immediately
        st.session_state.messages.append(ChatMessage(role="user", content=user_input))
        with st.chat_message("user"):
            st.write(user_input)

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_agent(user_input, st.session_state.messages[:-1])

            st.write(result.answer)

            # Metadata row
            cols = st.columns(3)
            cols[0].caption(f"⏱ {result.latency_seconds}s")
            cols[1].caption(f"🎯 Confidence: {result.confidence_score:.0%}")
            cols[2].caption("🔍 RAG" if result.used_retrieval else "💡 Direct")

            if result.escalated:
                st.warning(f"⚠️ Low confidence — flagged for HR review ({result.escalation_reason})")
                st.session_state.escalations.append(user_input)

        st.session_state.messages.append(ChatMessage(role="assistant", content=result.answer))

    # Sidebar
    with st.sidebar:
        st.header("Session Stats")
        total = len([m for m in st.session_state.messages if m.role == "user"])
        st.metric("Questions asked", total)
        st.metric("Escalations", len(st.session_state.escalations))

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.session_state.escalations = []
            st.rerun()

        st.divider()
        st.caption("Week 16 — Streamlit Deployment")

if __name__ == "__main__":
    main()
