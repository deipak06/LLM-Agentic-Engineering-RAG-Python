# AI Engineering Journey — Complete Summary
**Period:** July – September 2026  
**Starting point:** Python beginner, light LLM exposure  
**Hours/week:** 5-10  
**Current status:** Week 12 complete, Phase 3 in progress

---

## PHASE 0: Python Foundations (Weeks 1–6)

### Week 1: Python Basics — 10 Scripts
**What you built:** Word frequency counter (the capstone), plus 9 foundational scripts

**Key scripts:**
- FizzBuzz — learned `range(1, 101)` (not 0-100), if/elif/else ordering
- Name/age input — `int(input())` casting, f-strings
- Average function — first encounter with print vs return bug
- Word reversal — built manually without shortcuts
- Palindrome — `word == word[::-1]`, fixed truthy-string bug
- File reading — accumulator pattern, `encoding="utf-8"` for Unicode
- Word length dict — `{word: len(word)}`
- Manual zip vs `zip()` — built both, understood why `zip()` is safer
- `*args` basics — `def greet(greeting="Hello", *names):`
- **Word frequency counter** — full pipeline: read file → split → clean punctuation → count → sort by frequency

**Key bugs hit and fixed:**
- `print()` instead of `return` — hit 3 times, pattern recognized by end of week
- `range(0, 100)` vs `range(1, 101)` — off-by-one
- Ghost variable from loop scope bleeding into unrelated code
- `in dicttxt[word]` vs `in dicttxt` — membership check vs dict lookup
- `UnicodeDecodeError` — self-debugged from traceback, fixed with `encoding="utf-8"`

**Key concepts locked in:** accumulator pattern, return vs print, loop scope, membership checks, file encoding

---

### Week 2: Todo App — Dicts, JSON, Persistence
**What you built:** CLI todo list app with add/list/done/remove, persisted to JSON

**Architecture decisions made:**
- Dict keyed by ID (not name) — because names can duplicate, IDs are unique
- Never reuse deleted IDs — external logs reference old IDs, reuse creates ambiguity
- Auto-incrementing `uniquenumber` counter — starts at 0, increments before assigning

**Key features:**
```python
task = {}
uniquenumber = 0

def addtask(name, description, status=False):
    global uniquenumber
    uniquenumber += 1
    task[uniquenumber] = {"name": name, "description": description, "status": status}

def del_task(id_remove):
    if id_remove in task:  # membership check before access
        del task[id_remove]

def mark_done(status_id):
    if status_id in task:
        task[status_id]["status"] = True  # update ONE field, not replace whole dict

def save_task():
    with open("tasks.json", "w") as file:
        json.dump(task, file)

def load_task():
    global task, uniquenumber
    try:
        with open("tasks.json", "r") as file:
            loaded = json.load(file)
            task = {int(k): v for k, v in loaded.items()}  # string keys → int keys
            if task:
                uniquenumber = max(task.keys())  # recover counter after restart
    except FileNotFoundError:
        task = {}
        uniquenumber = 0
```

**Key bugs hit and fixed:**
- `mark_done` replacing whole dict instead of updating one field
- `uniquenumber` not surviving restart — `max(task.keys())` fix
- `FileExistsError` vs `FileNotFoundError` — wrong exception type
- `json.loads` vs `json.load` — string vs file object
- JSON string keys after load — `{int(k): v for k, v in loaded.items()}`

**Key concept:** file vs in-memory variable — `load_task()` only at startup, `save_task()` only on quit

---

### Week 3: OOP — Classes, Objects, Composition
**What you built:** Rebuilt todo app as `Task` + `TodoList` classes

```python
class Task:
    def __init__(self, task_id, name, description, status=False):
        self.id = task_id   # NOT: self.id = id (built-in collision!)
        self.name = name
        self.description = description
        self.status = status

    def mark_done(self, status=True):
        self.status = status  # no search needed — self IS the task

    def __str__(self):
        return f"{self.id} {self.name} {self.description} {self.status}"

class TodoList:
    def __init__(self):
        self.tasks = []       # list of Task objects
        self.next_id = 0      # counter lives on the object, not global

    def add_task(self, name, description):
        self.next_id += 1
        new_task = Task(self.next_id, name, description)
        self.tasks.append(new_task)

    def remove_task(self, task_id):
        for task in self.tasks:
            if task.id == task_id:   # search by attribute, not dict key
                self.tasks.remove(task)
                return

    def mark_done(self, task_id):
        for task in self.tasks:
            if task.id == task_id:
                task.mark_done(True)   # delegate to Task's own method
                return

    def save(self):
        tasks_as_dicts = []
        for task in self.tasks:
            tasks_as_dicts.append({
                "id": task.id, "name": task.name,
                "description": task.description, "status": task.status
            })
        with open("tasks.json", "w") as file:
            json.dump(tasks_as_dicts, file)

    def load(self):
        try:
            with open("tasks.json", "r") as file:
                loaded = json.load(file)
                for details in loaded:
                    new_task = Task(details['id'], details['name'],
                                    details['description'], details['status'])
                    self.tasks.append(new_task)
            if self.tasks:
                self.next_id = max(task.id for task in self.tasks)
        except FileNotFoundError:
            self.tasks = []
```

**Key concepts learned:**
- `self` refers to the specific object the method is called on — `task2.mark_done()` changes only `task2`
- Composition ("has-a"): `TodoList` contains `Task` objects
- `__str__` — Python calls it automatically when you `print(object)`
- `id` is a Python built-in — naming your variable `id` silently stores the function reference
- List of objects vs dict of dicts — O(n) loop vs O(1) dict lookup; tradeoff depends on use case

---

### Week 4: Error Handling, Virtual Environments, APIs
**What you built:** Script that fetches from a public API with graceful error handling

**Virtual environment setup (Windows PowerShell):**
```bash
& "C:\Users\...\Python313\python.exe" -m venv venv   # & prefix required
venv\Scripts\activate.bat                              # use .bat not .ps1
& .\venv\Scripts\pip.exe install requests             # always use venv's pip
```

**Two categories of failure:**
```python
def fetch_user(user_id: int) -> dict | None:
    try:
        response = requests.get(f"https://jsonplaceholder.typicode.com/users/{user_id}")
        if response.status_code == 200:          # HTTP-level failure — no exception raised
            return response.json()
        else:
            print(f"Request failed: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:  # Network-level failure — exception raised
        print("Could not connect")
        return None
```

**Key distinction:** HTTP failure (404, 500) doesn't crash Python — check `status_code`. Network failure crashes with `ConnectionError` — catch with `except`.

---

### Week 5: Type Hints + Pydantic
**What you built:** Type-hinted API client with Pydantic-validated response models

**Type hints — hints only, not enforcement:**
```python
def add_numbers(a: int, b: int) -> int:
    return a + b
add_numbers("hello", "world")  # runs, returns "helloworld" — no error
```

**Pydantic — actual validation:**
```python
from pydantic import BaseModel, ValidationError
from typing import Optional

class Address(BaseModel):
    street: str
    city: str
    zipcode: str

class User(BaseModel):
    id: int
    name: str
    username: str
    email: str
    address: Address
    phone: Optional[str] = None

# Behaviors:
User(id="not_a_number", ...)  # → ValidationError
User(id="42", ...)            # → SUCCESS: "42" coerced to 42
User(**response.json())       # ** unpacks dict into keyword args
```

**Key bug:** naming your file `pydantic.py` shadows the library — same pattern as naming a variable `id`.

---

### Week 6: Async/Await
**What you built:** Concurrent HTTP fetcher — 10 users in 0.51s vs 4.25s sequential

**Key syntax:**
```python
import asyncio
import httpx

async def fetch_user(client: httpx.AsyncClient, user_id: int) -> str:
    response = await client.get(f".../{user_id}")  # pause here, let others run
    return response.json()["name"]

async def main():
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[fetch_user(client, i) for i in range(1, 11)]
        )

asyncio.run(main())  # starts event loop
```

**Key rules:**
- `async def` creates a coroutine — calling it returns a coroutine object, doesn't execute
- `await` can only be used inside `async def`
- `asyncio.sleep(1)` pauses current coroutine; `time.sleep(1)` blocks everything
- `requests` is sync-only; use `httpx` for async HTTP

**Real measurement: 4.25s sequential → 0.51s async = 8× speedup**

---

## PHASE 1: LLM Fundamentals (Weeks 7–8)

### Week 7: Raw LLM API Calls
**What you built:** Raw Gemini API calls → chatbot with memory → streaming → tool calling

**Raw API call (httpx, no SDK):**
```python
response = httpx.post(
    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}",
    headers={"Content-Type": "application/json"},
    json={"contents": [{"role": "user", "parts": [{"text": "What is the capital of France?"}]}]},
    timeout=30.0
)
answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]
```

**Conversation memory — stateless LLM, manual history:**
```python
conversation_history = []

def chat(user_message: str) -> str:
    conversation_history.append({"role": "user", "parts": [{"text": user_message}]})
    response = httpx.post(..., json={"contents": conversation_history})
    reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    conversation_history.append({"role": "model", "parts": [{"text": reply}]})
    return reply
```

**Token count progression observed:** Turn 1: 13 tokens, Turn 2: 260 tokens, Turn 3: 275 tokens — each turn sends entire history, cost grows linearly.

**Tool calling — full loop:**
```python
# Step 1: Model responds with functionCall instead of text
{'functionCall': {'name': 'get_weather', 'args': {'city': 'Paris'}}}

# Step 2: Your code executes the actual function
tool_result = get_weather("Paris")  # → "22°C, sunny"

# Step 3: Send result back
messages.append({"role": "model", "parts": [{"functionCall": {...}}]})
messages.append({"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"result": tool_result}}}]})

# Step 4: Model gives final answer using the result
```

**Key concept:** tool calling is request-execute-return, not the model running your code.

---

### Week 8: Multi-Tool Agent + Structured Outputs
**What you built:** Agent with 3 tools (weather, calculator, fact lookup), `while True` loop, Pydantic response model

```python
TOOLS = {
    "get_weather": get_weather,
    "calculate": calculator,
    "get_fact": get_fact
}

def run_agent(user_message: str) -> AgentResponse:
    messages = [{"role": "user", "parts": [{"text": user_message}]}]
    tools_used = []
    
    try:
        while True:
            response = httpx.post(...)
            parts = response.json()["candidates"][0]["content"]["parts"]
            has_tool_call = False

            for part in parts:
                if "functionCall" in part:
                    has_tool_call = True
                    function_name = part["functionCall"]["name"]
                    function_args = part["functionCall"]["args"]
                    tool_result = TOOLS[function_name](**function_args)
                    tools_used.append(function_name)
                    # append model + user function response messages

            if not has_tool_call:
                text_part = next(p for p in parts if "text" in p)
                return AgentResponse(
                    answer=text_part["text"],
                    tools_used=tools_used,
                    tool_count=len(tools_used),
                    success=True
                )
    except Exception as e:
        return AgentResponse(answer="", tools_used=tools_used,
                             tool_count=len(tools_used), success=False, error=str(e))
```

**Parallel tool calling discovered:** model can return multiple `functionCall` parts in one response. Fixed by looping all parts before checking `if not has_tool_call`.

**Multi-step chaining tested:** "Weather in Paris and Tokyo, sum their temperatures?" → 3 tool calls chained.

**Key bug:** `tools_count` in Pydantic model vs `tool_count` in code — naming mismatch caused ValidationError.

---

## PHASE 2: RAG (Weeks 9–10)

### Week 9: Naive RAG — Build It Wrong First
**What you built:** End-to-end RAG pipeline on a company HR handbook

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def load_and_chunk(filepath: str, chunk_size: int = 3) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return [" ".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]

model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode(chunks)

def retrieve(query, chunks, chunk_embeddings, model, top_k=3):
    query_emb = model.encode([query])
    sims = np.dot(chunk_embeddings, query_emb.T).flatten()
    sims = sims / (np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_emb))
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_idx]
```

**Document:** 27 non-blank lines → 9 chunks (3 lines each)

**Failure modes discovered:**
- Vocabulary mismatch: "WFH" vs "remote work" — actually worked due to embedding training data
- Ambiguous query: "What are the limits?" → partial silent answer
- **Key insight: naive RAG fails silently — no crash, just a confident partial answer**

---

### Week 10: Fix Naive RAG Systematically
**What you built:** Eval framework with semantic similarity scoring, improved prompt

```python
def semantic_similarity(text1, text2, model):
    embeddings = model.encode([text1, text2])
    return float(np.dot(embeddings[0], embeddings[1]) /
                 (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))

def run_eval(chunks, chunk_embeddings, embedding_model, threshold=0.7):
    for item in EVAL_SET:
        retrieved = retrieve(item["question"], ...)
        actual = generate_answer(item["question"], retrieved)
        score = semantic_similarity(item["expected"], actual, embedding_model)
        passed = score >= threshold
```

**Lessons learned:**
- Initial eval: 3/8 (38%) — expected answers too terse ("20 days" vs "20 days of annual leave per year")
- After fixing expected answer phrasing: 7/8 (88%)
- Receipt question failure: document states requirement, not consequence — unanswerable as phrased
- Changed "what happens if I lose a receipt" → "do I need a receipt" → 8/8 (100%)
- `if __name__ == "__main__":` guard required so importing `naive_rag.py` doesn't run test calls

**Improved prompt:**
```python
prompt = f"""You are a helpful HR assistant. Answer using ONLY the context below.
- Answer in complete sentences
- If you can directly answer or reasonably infer from context, provide it
- If not in context, say exactly: "I don't have that information."

Context: {context}
Question: {query}
Answer:"""
```

**Key lesson:** eval set design is as hard as RAG design. A bad eval gives false confidence.

---

## PHASE 3: Agentic Systems (Weeks 11–12)

### Week 11: Agentic RAG
**What you built:** Agent that decides when to search, with conversation memory

**Key difference from naive RAG:** agent decides whether to call the RAG tool at all, based on tool description.

**Agent decides + rewrites queries:**
- "How many days of annual leave do I get?" → searched "annual leave days vacation policy"
- "What is 15 multiplied by 4?" → **no search, answered directly: 60**
- "What is the capital of France?" → **no search, answered directly: Paris**

**Multi-turn conversation with memory:**
```python
def run_agentic_rag(user_question: str, history: list = None) -> tuple[str, list]:
    if history is None:
        history = []
    
    messages = history.copy()
    messages.append({"role": "user", "parts": [{"text": user_question}]})
    final_answer = ""

    while True:
        response = client.models.generate_content(...)
        candidate = response.candidates[0]
        parts = candidate.content.parts
        has_tool_call = False

        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                has_tool_call = True
                tool_result = TOOLS[function_name](**function_args)
                messages.append(candidate.content)  # SDK native object — includes thought_signature
                messages.append({"role": "user", "parts": [{"function_response": {...}}]})

        if not has_tool_call:
            text_part = next((p for p in parts if hasattr(p, "text") and p.text), None)
            if text_part:
                final_answer = text_part.text
            break

    # Only persist user question + final answer (not tool calls)
    updated_history = history.copy()
    updated_history.append({"role": "user", "parts": [{"text": user_question}]})
    updated_history.append({"role": "model", "parts": [{"text": final_answer}]})
    return final_answer, updated_history
```

**Multi-turn test results:**
- Q1: "How many days of annual leave?" → searched, answered
- Q2: "Can I carry them over?" → **no search** — resolved "them" from Q1's history
- Q3: "What about sick leave?" → searched "sick leave carry over policy"
- Q4: "What is the capital of France?" → **no search** despite 3 turns of HR history

**Critical SDK fix:** `messages.append(candidate.content)` instead of manually reconstructing the dict — SDK's native content object includes `thought_signature` required by Gemini.

---

### Week 12: Agent Evaluation Framework
**What you built:** Three-dimensional Pydantic eval framework

```python
class ToolCall(BaseModel):
    tool_name: str
    query: str
    chars_retrieved: int

class AgentRun(BaseModel):
    question: str
    answer: str
    tool_calls: list[ToolCall]
    tool_call_count: int
    duration_seconds: float
    success: bool
    error: Optional[str] = None

class TestCase(BaseModel):
    question: str
    expected_answer: str
    should_use_tool: bool
    max_tool_calls: int = 2
    topic: str = "general"

class EvalResult(BaseModel):
    test_case: TestCase
    run: AgentRun
    answer_score: float        # semantic similarity ≥ 0.7
    tool_use_correct: bool     # used tool iff should_use_tool
    efficiency_ok: bool        # tool_call_count ≤ max_tool_calls
    passed: bool               # ALL THREE must pass
```

**Why all three required:** correct answer doesn't guarantee correct behavior. An agent that answers "100" for "25 × 4" after searching the HR knowledge base got the right answer for the wrong reason — at scale, unnecessary searches cost 2-3× more compute and add latency.

**Final result: 6/6 (100%)** across all dimensions.

**Timing observed:**
- HR questions (with retrieval): 6-10 seconds
- General knowledge (no retrieval): 2-4 seconds
- Cost of RAG: ~2-3× latency when retrieval is needed

---

## Recurring Bug Patterns — All Instances

| Bug | Weeks | Notes |
|---|---|---|
| `print()` instead of `return` | 1 (×3), 3 | Pattern recognized by W3 |
| Missing `return` at end of function | 1, 3, 8, 12 | Still appearing — check every function |
| File/variable named same as library | 1 (`id`), 5 (`pydantic.py`) | Both caught and fixed |
| Script inside `venv/` folder | 4, 7, 8, 9 | Always put scripts in project root |
| Wrong model name in API call | 11 (×2), 12 | Verify before every run |
| Test calls running on import | 10, 11 | Fixed with `if __name__ == "__main__":` |
| `in dict[key]` vs `in dict` | 1 | Fixed, not repeated |
| Overwrite whole dict vs update field | 2, 3 | Fixed, not repeated after W3 |

---

## Tools and Libraries Used

| Library | Purpose | First used |
|---|---|---|
| `requests` | Synchronous HTTP | Week 4 |
| `httpx` | Sync + async HTTP | Week 6 |
| `pydantic` | Data validation and typed models | Week 5 |
| `python-dotenv` | Load API keys from `.env` | Week 7 |
| `google-genai` | Gemini API SDK | Week 7 |
| `sentence-transformers` | Text embedding model | Week 9 |
| `numpy` | Vector math for cosine similarity | Week 9 |
| `asyncio` | Async event loop | Week 6 |

---

## Key Concepts — One-Line Summaries

**LLMs are stateless** — every API call starts fresh; "memory" is just resending history  
**Tool calling** — model requests a function call; your code executes it; result sent back  
**RAG** — retrieve relevant text chunks → stuff into prompt → LLM answers from context  
**Chunking tradeoff** — smaller chunks: more precise retrieval; larger: more context preserved  
**Semantic search** — meaning-based (vectors), not word-based; finds synonyms  
**Eval sets** — build before improving; expected answers must match LLM's natural output style  
**Agent eval** — answer quality + tool use correctness + efficiency; all three required  
**Async** — cooperative multitasking; total time ≈ slowest single task, not sum of all  
**Pydantic** — validates at the boundary; prevents silent type errors downstream  
**Virtual environments** — isolate packages per project; always use venv's pip  

---

## Environment Setup (Windows PowerShell)

```bash
mkdir "D:\python\WeekN-Name"
cd "D:\python\WeekN-Name"
& "C:\Users\80118091\AppData\Local\Programs\Python\Python313\python.exe" -m venv venv
& "D:\python\WeekN-Name\venv\Scripts\pip.exe" install [packages]
```

- Create `.env` and `.gitignore` in **project root**, never inside `venv/`
- Always prefix quoted executable paths with `&` in PowerShell
- Current working model: `gemini-2.5-flash-lite`
- API key in `.env` as `GEMINI_API_KEY`
- Load with: `load_dotenv(dotenv_path=Path(__file__).parent / ".env")`

---

## Path A vs Path B — Recommendation

### Path A: Continue roadmap
Cost tracking → HITL → fine-tuning → deployment. Deeper technical knowledge before shipping. Risk: by the time you ship, December is close with less time to iterate on real feedback.

### Path B: Ship it now (Recommended)
Build Streamlit UI → deploy publicly → fix what real users break. Risk: deployed system will have gaps (no cost tracking, HITL, or fine-tuning initially). But those gaps become obvious and motivating once real users hit them.

### Recommendation: Path B with one addition

**Week 13 plan:**
1. First half: add token counting and cost logging to `run_agent` (2-3 hours)
2. Second half: build Streamlit interface wrapping `agentic_rag.py`
3. Week 14: deploy to Render or Streamlit Cloud (free tier), share the URL
4. Week 15+: fix what real users break, add HITL/fine-tuning when motivated by real failures

**Why you're ready to ship:** fully functional agentic RAG with conversation memory, multi-tool support, and three-dimensional eval framework. That's a production-worthy architecture, not a tutorial exercise.

---

## Learner Profile

**Strengths:**
- Self-corrects on bug patterns over time — stops making the same category of mistake twice
- Asks "why does this scale?" not just "how do I make it work?"
- Reads error messages directly before guessing
- Insists on real restart tests (caught `uniquenumber` and `next_id` recovery bugs)

**Persistent gaps:**
- Conceptual answers still one level too vague — points at the right thing but doesn't name the mechanism
- Edge case testing still needs prompting (happy path tested well, failure modes need suggestion)
- Environment setup friction repeats weekly — needs a 5-step checklist as habit

**Honest pace:** ~1.5-2× slower than roadmap's nominal estimate due to feedback loop density. By December: strong hireable AI engineer with deployable RAG + agent system.
