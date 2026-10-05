import json
from pathlib import Path
from pydantic import BaseModel

#----- Pydantic models -------------------------------

class TrainingExample(BaseModel):
    messages:list[dict] ## [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

class DatasetReport(BaseModel):
    total_escalations: int
    resolved: int
    skipped_pending:int
    skipped_no_answer:int
    final_examples:int
    output_file: str

#-- Seed data----------------------------------------------------------------------
# High-quality HR Q&A pairs we know are correct.
# In production these come from your verified eval set + SME review.
# Fine-tuning needs ~100+ examples minimum; we're building the pipeline, not training yet.

SEED_EXAMPLES = [
    ("How many days of annual leave do I get?",
     "Employees are entitled to 20 days of annual leave per year."),
    ("How many days can I carry over?",
     "Unused leave can be carried over to the next year, up to a maximum of 10 days."),
    ("Can I work from home every day?",
     "No. Employees may work remotely up to 3 days per week."),
    ("What are the WFH rules?",
     "Remote work must be approved by your direct manager. You must be available during core hours: 10am to 3pm. Equipment must be approved by IT."),
    ("How often are performance reviews?",
     "Performance reviews are conducted twice a year, in June and December."),
    ("How often must I change my password?",
     "All passwords must be changed every 90 days."),
    ("Is two-factor authentication required?",
     "Yes. Two-factor authentication is mandatory for all company systems."),
    ("How much notice do I need to give for leave?",
     "Leave must be requested at least 2 weeks in advance."),
    ("How much sick leave do I get?",
     "Sick leave is separate from annual leave and is capped at 15 days per year."),
    ("What is the meal expense limit?",
     "Meal expenses are capped at $50 per person per day."),
    ("When must I submit expense claims?",
     "Business expenses must be submitted within 30 days of the expense date."),
    ("Do I need a receipt for expenses?",
     "Receipts are required for any expense over $25."),
    ("What travel expenses need pre-approval?",
     "Travel expenses require pre-approval for amounts over $500."),
    ("How long is maternity leave?",
     "Maternity leave is 6 months, paid."),
    ("How long is paternity leave?",
     "Paternity leave is 2 weeks, paid."),
    ("What happens if I get a rating of 2 or below twice in a row?",
     "Employees with a rating of 2 or below for two consecutive reviews may face a performance improvement plan."),
    ("What should I do if I discover a data breach?",
     "Data breaches must be reported to IT within 24 hours of discovery."),
    ("Can I install software on my work computer?",
     "No. Employees must not install unauthorized software on company devices."),
    ("What is a satisfactory performance rating?",
     "A rating of 3 or above is considered satisfactory performance."),
    ("What scale are performance ratings on?",
     "Employees receive a rating from 1 to 5 in performance reviews."),
]

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_escalations(filepath: str) -> list[dict]:
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  Warning: {filepath} not found — skipping escalation data")
        return []

def format_example(question: str, answer: str) -> TrainingExample:
    return TrainingExample(messages=[
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ])


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_dataset(escalations_file: str = "escalations.json",
                  output_file: str = "training_data.jsonl") -> DatasetReport:

    examples: list[TrainingExample] = []

    # Source 1: Seed examples (hand-verified)
    for question, answer in SEED_EXAMPLES:
        examples.append(format_example(question, answer))
    print(f"  Seed examples:        {len(SEED_EXAMPLES)}")

    # Source 2: Resolved HITL escalations (human-verified)
    escalations = load_escalations(escalations_file)
    total_esc = len(escalations)
    skipped_pending = 0
    skipped_no_answer = 0
    hitl_added = 0

    for record in escalations:
        if record["status"] != "resolved":
            skipped_pending += 1
            continue
        if not record.get("human_answer"):
            skipped_no_answer += 1
            continue

        # Deduplicate: skip if same question already in seed examples
        existing_questions = [ex.messages[0]["content"].lower() for ex in examples]
        if record["question"].lower() in existing_questions:
            continue

        examples.append(format_example(record["question"], record["human_answer"]))
        hitl_added += 1

    print(f"  HITL escalations:     {total_esc} total, {hitl_added} added, "
          f"{skipped_pending} pending, {skipped_no_answer} no answer")

    # Write JSONL — one JSON object per line (standard fine-tuning format)
    with open(output_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example.model_dump()) + "\n")

    report = DatasetReport(
        total_escalations=total_esc,
        resolved=hitl_added,
        skipped_pending=skipped_pending,
        skipped_no_answer=skipped_no_answer,
        final_examples=len(examples),
        output_file=output_file
    )
    return report

def preview_dataset(output_file: str, n: int = 3):
    print(f"\n  First {n} examples from {output_file}:")
    with open(output_file, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            example = json.loads(line)
            user_msg = example["messages"][0]["content"]
            assistant_msg = example["messages"][1]["content"]
            print(f"\n  [{i+1}]")
            print(f"    user:      {user_msg}")
            print(f"    assistant: {assistant_msg}")

def explain_finetune_submission(output_file: str, example_count: int):
    print(f"""
{'='*60}
WHAT TO DO WITH {output_file}
{'='*60}

You now have {example_count} training examples in JSONL format.
Each line is one (question, answer) pair the model will learn from.

To actually submit a fine-tuning job on Gemini:
  1. Upload training_data.jsonl to Google Cloud Storage
  2. Call the Gemini fine-tuning API:
       tuning_job = client.tunings.tune(
           base_model='models/gemini-3.5-flash-lite',
           training_dataset={{'gcs_uri': 'gs://your-bucket/training_data.jsonl'}},
           config=types.CreateTuningJobConfig(epoch_count=5, learning_rate_multiplier=1.0)
       )
  3. Wait ~1-4 hours for training
  4. Get back a new model ID like: tunedModels/hr-assistant-abc123
  5. Replace MODEL = "gemini-3.5-flash-lite" with your tuned model ID
  6. Re-run your eval framework (Week 12) to measure improvement

MINIMUM for real fine-tuning: ~100 examples
Current count: {example_count}
Gap: {max(0, 100 - example_count)} more examples needed before submitting

How to get more examples:
  - Run more questions through your HITL agent, resolve escalations
  - Ask HR team to write 50 Q&A pairs from common employee questions
  - Generate variations: "How much leave do I get?" / "What's my leave entitlement?"
{'='*60}
""")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building fine-tuning dataset...\n")
    report = build_dataset()

    print(f"\n{'='*60}")
    print(f"DATASET REPORT")
    print(f"{'='*60}")
    print(f"  Final training examples: {report.final_examples}")
    print(f"  Output file: {report.output_file}")

    preview_dataset(report.output_file, n=3)
    explain_finetune_submission(report.output_file, report.final_examples)