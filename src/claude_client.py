import os
import anthropic
from dotenv import load_dotenv

load_dotenv()
import pandas as pd
import asyncio
import re as _re
from pathlib import Path


client = anthropic.Anthropic()
async_client = anthropic.AsyncAnthropic()

MODEL = "claude-sonnet-4-6"

# How many emails we send to Claude for labeling
LABEL_SAMPLE_SIZE = 500

# Batch config: emails per prompt × parallel requests
EMAILS_PER_BATCH = 10
MAX_CONCURRENT = 15

BATCH_PROMPT_TEMPLATE = """You are analyzing internal corporate emails from Enron Corporation.

Rate EACH email below on two scales:

SELF-DISCLOSURE (does the sender share personal information, feelings, or experiences?):
1 = Formal business only (reports, contracts, scheduling)
2 = Mostly work with slight personal touch
3 = Mixed work and personal content
4 = Mostly personal (social plans, personal news, feelings)
5 = Deeply personal (intimate feelings, private matters, vulnerability)

RESPONSIVENESS (does the sender show understanding, validation, or caring toward the recipient?):
1 = Hostile, angry, or threatening
2 = Curt, cold, or distant
3 = Neutral, professional tone
4 = Friendly, warm, supportive
5 = Deeply caring, validating, emotionally attuned

{emails_block}

Reply with ONLY one line per email, in order: two numbers separated by a comma.
Example reply for 3 emails:
2,4
1,3
3,5"""


def _build_emails_block(texts: list) -> str:
    """Format multiple emails into a numbered block for the prompt."""
    parts = []
    for i, text in enumerate(texts, 1):
        snippet = (text[:600].strip() if isinstance(text, str) else "(empty)")
        parts.append(f"--- EMAIL {i} ---\n{snippet}")
    return "\n\n".join(parts)


def _parse_batch_response(text: str, n_expected: int) -> list:
    """Parse multi-line response like '2,4\\n1,3\\n3,5' into list of dicts."""
    results = []
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]

    for line in lines:
        # Extract first pair of numbers from the line
        match = _re.search(r"(\d)\s*[,]\s*(\d)", line)
        if match:
            intimacy = max(1, min(5, int(match.group(1))))
            warmth = max(1, min(5, int(match.group(2))))
            results.append({"intimacy": intimacy, "warmth": warmth})

    # Pad with defaults if Claude returned fewer lines
    while len(results) < n_expected:
        results.append({"intimacy": 1, "warmth": 3})

    return results[:n_expected]


async def _label_batch_async(texts: list, semaphore: asyncio.Semaphore) -> list:
    """Label a batch of emails with one async API call."""
    async with semaphore:
        emails_block = _build_emails_block(texts)
        prompt = BATCH_PROMPT_TEMPLATE.format(emails_block=emails_block)

        try:
            response = await async_client.messages.create(
                model=MODEL,
                max_tokens=5 * len(texts),
                messages=[{"role": "user", "content": prompt}]
            )
            return _parse_batch_response(response.content[0].text, len(texts))
        except Exception as e:
            print(f"    Batch failed: {e}")
            return [{"intimacy": 1, "warmth": 3}] * len(texts)


async def _label_all_async(texts: list) -> list:
    """Label all texts using batched async requests."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Split into batches
    batches = []
    for i in range(0, len(texts), EMAILS_PER_BATCH):
        batches.append(texts[i:i + EMAILS_PER_BATCH])

    total_batches = len(batches)
    print(f"  {len(texts)} emails → {total_batches} batches "
          f"({EMAILS_PER_BATCH}/batch, {MAX_CONCURRENT} parallel)")

    # Run all batches with progress
    all_results = []
    done = 0

    # Process in waves to show progress
    wave_size = MAX_CONCURRENT
    for wave_start in range(0, len(batches), wave_size):
        wave = batches[wave_start:wave_start + wave_size]
        tasks = [_label_batch_async(batch, semaphore) for batch in wave]
        wave_results = await asyncio.gather(*tasks)

        for batch_result in wave_results:
            all_results.extend(batch_result)
            done += 1

        labeled_so_far = len(all_results)
        print(f"    Labeled {labeled_so_far}/{len(texts)} emails "
              f"({done}/{total_batches} batches)", flush=True)

    return all_results



def label_email_batch(df: pd.DataFrame,
                      sample_size: int = LABEL_SAMPLE_SIZE) -> pd.DataFrame:
    """
    Take a random sample of emails, send them to Claude in batches
    with async parallelism for ~20x speedup over sequential calls.
    """
    print(f"Sending {sample_size} emails to Claude for labeling...")

    sample = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
    texts = sample["body"].tolist()

    # Run async batch labeling
    results = asyncio.run(_label_all_async(texts))

    sample["intimacy_label"] = [r["intimacy"] for r in results]
    sample["warmth_label"] = [r["warmth"] for r in results]

    print(f"Labeling complete.")
    print(f"  Self-disclosure distribution:\n{sample['intimacy_label'].value_counts().sort_index().to_string()}")
    print(f"  Responsiveness distribution:\n{sample['warmth_label'].value_counts().sort_index().to_string()}")

    return sample


def ask_question(question: str, context: str) -> str:
    """
    Answer a question about the Enron data in plain English.
    Used by the chat assistant in the Streamlit UI.
    """
    prompt = f"""You are an AI assistant helping explore the results of a
machine learning analysis of the Enron email dataset.

Here are the key findings from our analysis:
{context}

Answer this question based on the data above. Be specific and refer to
actual names and numbers from the findings. Keep your answer clear and
concise — this is being shown to a university professor.

Question: {question}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip()


def save_labeled_sample(labeled_df: pd.DataFrame, output_dir: str):
    """Save the Claude-labeled sample so we don't have to re-label next time."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(out / "claude_labeled_emails.csv", index=False)
    print(f"Labeled sample saved to {output_dir}/claude_labeled_emails.csv")


def load_labeled_sample(output_dir: str) -> pd.DataFrame:
    """Load a previously saved labeled sample. Returns None if not found or wrong format."""
    path = Path(output_dir) / "claude_labeled_emails.csv"
    if path.exists():
        df = pd.read_csv(path)
        # Check if it has the new two-scale labels
        if "intimacy_label" in df.columns and "warmth_label" in df.columns:
            print("Loading existing Claude labels (skipping API calls)...")
            return df
        else:
            print("Old label format found — will re-label with new two-scale format.")
            return None
    return None
