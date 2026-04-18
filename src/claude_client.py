import os
import anthropic
from dotenv import load_dotenv

load_dotenv()
import pandas as pd
from pathlib import Path


client = anthropic.Anthropic()

MODEL = "claude-sonnet-4-6"

# How many emails we send to Claude for labeling
LABEL_SAMPLE_SIZE = 500


def label_email(email_text: str) -> dict:
    """
    Ask Claude to rate a single email on two scales:
      - Intimacy (1-5): how personal/private is the content?
        1=formal business, 2=mostly work, 3=mixed, 4=personal, 5=deeply personal/romantic
      - Warmth (1-5): how warm/supportive is the tone?
        1=hostile/cold, 2=curt/distant, 3=neutral, 4=friendly/warm, 5=loving/intimate

    These two scales map to Gilbert & Karahalios (2009) dimensions:
    - Intimacy → "Intimacy" dimension (32.8% predictive contribution)
    - Warmth → "Emotional support" dimension

    We send the first 800 characters to give Claude enough context.
    Returns {"intimacy": int, "warmth": int}
    """
    if not isinstance(email_text, str) or not email_text.strip():
        return {"intimacy": 1, "warmth": 3}

    snippet = email_text[:800].strip()

    prompt = f"""You are analyzing internal corporate emails from Enron Corporation.

Rate this email on two scales:

INTIMACY (how personal/private is the content?):
1 = Formal business (reports, contracts, scheduling)
2 = Mostly work with slight personal touch
3 = Mixed work and personal content
4 = Mostly personal (social plans, personal news)
5 = Deeply personal or romantic (intimate feelings, love, private matters)

WARMTH (how warm/supportive is the tone?):
1 = Hostile, angry, or threatening
2 = Curt, cold, or distant
3 = Neutral, professional tone
4 = Friendly, warm, supportive
5 = Loving, intimate, deeply caring

Email text:
\"\"\"{snippet}\"\"\"

Reply with ONLY two numbers separated by a comma. Example: 2,4"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()

    # Parse the response
    try:
        parts = text.replace(" ", "").split(",")
        intimacy = int(parts[0])
        warmth = int(parts[1])
        # Clamp to valid range
        intimacy = max(1, min(5, intimacy))
        warmth = max(1, min(5, warmth))
        return {"intimacy": intimacy, "warmth": warmth}
    except (ValueError, IndexError):
        return {"intimacy": 1, "warmth": 3}


def label_email_batch(df: pd.DataFrame,
                      sample_size: int = LABEL_SAMPLE_SIZE) -> pd.DataFrame:
    """
    Take a random sample of emails, send each one to Claude to rate
    on intimacy and warmth scales, and return a DataFrame with labels added.
    """
    print(f"Sending {sample_size} emails to Claude for labeling...")

    sample = df.sample(n=min(sample_size, len(df)), random_state=42).copy()

    intimacy_labels = []
    warmth_labels = []

    for i, (_, row) in enumerate(sample.iterrows()):
        if i % 50 == 0:
            print(f"  Labeled {i}/{len(sample)} emails...")

        result = label_email(row["body"])
        intimacy_labels.append(result["intimacy"])
        warmth_labels.append(result["warmth"])

    sample["intimacy_label"] = intimacy_labels
    sample["warmth_label"] = warmth_labels

    print(f"Labeling complete.")
    print(f"  Intimacy distribution:\n{sample['intimacy_label'].value_counts().sort_index().to_string()}")
    print(f"  Warmth distribution:\n{sample['warmth_label'].value_counts().sort_index().to_string()}")

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
