import os
import time
from dotenv import load_dotenv
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from google import genai
from typing import Optional

load_dotenv(dotenv_path=Path(__file__).parent /".env")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

#------- Pydantic models for eval---------------

class ToolCall(BaseModel):
    tool_name : str
    query : str
    chars_retrieved : int

class AgentRun(BaseModel):
    question: str
    answer : str
    tool_calls: list[ToolCall]
    tool_call_count: int
    duration_seconds: float
    success: bool
    error: Optional[str]=None

class TestCase(BaseModel):
    question : str
    expected_answer : str
    should_use_tool : bool
    max_tool_calls : int = 2
    topic: str = "general"

class EvalResult(BaseModel):
    test_case: TestCase
    run: AgentRun
    answer_score:float
    tool_use_correct:bool
    efficiency_ok:bool
    passed: bool

#--- RAG Setup --------------------------------------------

def load_and_chunk(filepath:str,chunk_size:int=3) -> list[str]:
    with open(filepath,"r",encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    chunks = []
    for i in range(0,len(lines),chunk_size):
        chunks.append(" ".join(lines[i:i + chunk_size]))
    return chunks

def embed_chunks(chunks:list[str], model:SentenceTransformer) -> np.ndarray:
    return model.encode(chunks)

def retrieve(query:str,chunks:list[str], chunk_embeddings:np.ndarray,
             model: SentenceTransformer, top_k:int =3) ->list[str]:
    query_emb = model.encode([query])
    sims = np.dot(chunk_embeddings,query_emb.T).flatten()
    sims = sims / (np.linalg.norm(chunk_embeddings,axis=1)*
                   np.linalg.norm(query_emb))
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_idx] 

print("Load embedding model....")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = load_and_chunk("document.txt")
chunk_embeddings = embed_chunks(chunks,embedding_model)
print(f"Ready -{len(chunks)} chunks embedded\n")

def search_knowledge_base(query:str) ->str:
    retrieved = retrieve(query,chunks,chunk_embeddings,embedding_model)
    context = " "
    for i,chunk in enumerate(retrieved):
        context += f"[{i+1}]{chunk}\n\n"
    return f"Retrieved context for '{query}':\n\n{context}"

TOOLS = {"search_knowledge_base":search_knowledge_base}

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

client = genai.Client(api_key=GEMINI_API_KEY)

#---- Instrumented agent run ------

def run_agent(question:str) -> AgentRun:
    """Run agent and capture full trace for evaluation"""
    messages = [{"role": "user", "parts": [{"text": question}]}]
    tool_calls_made = []
    start_time = time.time()
    final_answer = ""

    try:
        while True:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=messages,
                config={"tools": tool_definitions}
            )

            candidate = response.candidates[0]
            parts = candidate.content.parts
            has_tool_call = False

            for part in parts:
                if hasattr(part, "function_call") and part.function_call:
                    has_tool_call = True
                    function_name = part.function_call.name
                    function_args = dict(part.function_call.args)
                    query = function_args.get("query", "")

                    tool_result = TOOLS[function_name](**function_args)

                    tool_calls_made.append(ToolCall(
                        tool_name=function_name,
                        query=query,
                        chars_retrieved=len(tool_result)
                    ))

                    messages.append(candidate.content)
                    messages.append({
                        "role": "user",
                        "parts": [{"function_response": {
                            "name": function_name,
                            "response": {"result": tool_result}
                        }}]
                    })

            if not has_tool_call:
                print(f"  DEBUG final parts: {[(getattr(p,'text','')[:50] if getattr(p,'text',None) else None, bool(getattr(p,'function_call',None))) for p in parts]}")
                text_part = next(
                    (p for p in parts if hasattr(p, "text") and p.text), None
                )
                if text_part:
                    final_answer = text_part.text
                break

        return AgentRun(
            question=question,
            answer=final_answer,
            tool_calls=tool_calls_made,
            tool_call_count=len(tool_calls_made),
            duration_seconds=round(time.time() - start_time, 2),
            success=True
        )

    except Exception as e:
        return AgentRun(
            question=question,
            answer="",
            tool_calls=tool_calls_made,
            tool_call_count=len(tool_calls_made),
            duration_seconds=round(time.time() - start_time, 2),
            success=False,
            error=str(e)
        )

#--- Scoring -----------------------------------

def semantic_similarity(text1:str, text2:str, 
                        model:SentenceTransformer) -> float:
    embeddings = model.encode([text1,text2])
    return float(np.dot(embeddings[0],embeddings[1])/(
        np.linalg.norm(embeddings[0])* np.linalg.norm(embeddings[1])
    ))

def evaluate_run(test_case:TestCase, run:AgentRun) -> EvalResult:
    # 1. Answer quality
    answer_score = semantic_similarity(
        test_case.expected_answer,run.answer,embedding_model
    ) if run.answer else 0.0

    # 2. Tool use correctness 
    actually_used_tool = run.tool_call_count >0
    tool_use_correct = actually_used_tool == test_case.should_use_tool

    # 3 Efficiency
    efficiency_ok= run.tool_call_count <= test_case.max_tool_calls

    # 4 Overall pass
    passed = (
        answer_score >= 0.7 and
        tool_use_correct and
        efficiency_ok and
        run.success
    )

    return EvalResult(
        test_case=test_case,
        run=run,
        answer_score=answer_score,
        tool_use_correct=tool_use_correct,
        efficiency_ok=efficiency_ok,
        passed=passed
    )

# ── Test suite ────────────────────────────────────────────────
TEST_SUITE = [
    TestCase(
        question="How many days of annual leave do I get?",
        expected_answer="Employees are entitled to 20 days of annual leave per year",
        should_use_tool=True,
        max_tool_calls=1,
        topic="leave_policy"
    ),
    TestCase(
        question="Can I work from home every day?",
        expected_answer="Employees may work remotely up to 3 days per week",
        should_use_tool=True,
        max_tool_calls=1,
        topic="remote_work"
    ),
    TestCase(
        question="Do I need a receipt for a $30 expense?",
        expected_answer="Receipts are required for any expense over $25",
        should_use_tool=True,
        max_tool_calls=1,
        topic="expense_policy"
    ),
    TestCase(
        question="How often are performance reviews conducted?",
        expected_answer="Performance reviews are conducted twice a year in June and December",
        should_use_tool=True,
        max_tool_calls=1,
        topic="performance"
    ),
    TestCase(
        question="What is 25 multiplied by 4?",
        expected_answer="25 multiplied by 4 is 100",
        should_use_tool=False,
        max_tool_calls=0,
        topic="general_knowledge"
    ),
    TestCase(
        question="What is the capital of Japan?",
        expected_answer="The capital of Japan is Tokyo",
        should_use_tool=False,
        max_tool_calls=0,
        topic="general_knowledge"
    ),
]


# ── Run evaluation ────────────────────────────────────────────
def run_eval_suite(test_suite: list[TestCase]) -> list[EvalResult]:
    results = []
    passed = 0

    print(f"{'='*60}")
    print(f"Agent Eval Suite — {len(test_suite)} test cases")
    print(f"{'='*60}\n")

    for i, test_case in enumerate(test_suite):
        print(f"[{i+1}/{len(test_suite)}] {test_case.topic}: {test_case.question}")

        run = run_agent(test_case.question)
        result = evaluate_run(test_case, run)
        results.append(result)

        if result.passed:
            passed += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        print(f"  {status} | answer: {result.answer_score:.2f} | "
              f"tool_use: {'✓' if result.tool_use_correct else '✗'} | "
              f"efficiency: {'✓' if result.efficiency_ok else '✗'} | "
              f"time: {run.duration_seconds}s")

        if run.tool_calls:
            for tc in run.tool_calls:
                print(f"    → searched: '{tc.query}'")

        if not result.passed:
            print(f"    ⚠ answer: '{run.answer[:100]}'")
            print(f"    ⚠ expected: '{test_case.expected_answer}'")

    print(f"\n{'='*60}")
    print(f"Final Score: {passed}/{len(test_suite)} "
          f"({passed/len(test_suite)*100:.0f}%)")

    # Summary by topic
    topics = {}
    for r in results:
        t = r.test_case.topic
        if t not in topics:
            topics[t] = {"passed": 0, "total": 0}
        topics[t]["total"] += 1
        if r.passed:
            topics[t]["passed"] += 1

    print("\nBy topic:")
    for topic, counts in topics.items():
        print(f"  {topic}: {counts['passed']}/{counts['total']}")
    print(f"{'='*60}")

    return results

if __name__ == "__main__":
    run_eval_suite(TEST_SUITE)