import os
import anthropic
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from the .env file
import pandas as pd
from pathlib import Path


# We create one client that gets reused across all calls.
# It reads your API key from the environment variable ANTHROPIC_API_KEY.
client = anthropic.Anthropic()

# The Claude model we use — Sonnet is fast and smart, good balance for this project
MODEL = "claude-sonnet-4-6"

# How many emails we send to Claude for labeling
LABEL_SAMPLE_SIZE = 500


def label_email(email_text: str) -> str:
    """
    Ask Claude to classify a single email as:
      - "professional" : normal work email, business topics
      - "personal"     : friendly, casual, social, intimate, or romantic

    We use binary classification (2 classes instead of 3+) because:
    - Binary classifiers are more reliable with limited training data
    - We use the probability score as a spectrum rather than hard labels
    - The downstream clustering handles finer relationship types

    We send the first 800 characters to give Claude enough context
    without excessive cost.
    """
    if not isinstance(email_text, str) or not email_text.strip():
        return "professional"

    snippet = email_text[:800].strip()

    prompt = f"""You are analyzing internal corporate emails from Enron Corporation.

Classify this email into exactly ONE of these two categories:
- professional: normal work communication, business topics, formal requests, reports, scheduling
- personal: friendly, casual, social, emotional, intimate, romantic, or any non-work content

Email text:
\"\"\"{snippet}\"\"\"

Reply with only one word: professional or personal."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )

    label = response.content[0].text.strip().lower()

    if label not in ("professional", "personal"):
        return "professional"

    return label


def label_email_batch(df: pd.DataFrame,
                      sample_size: int = LABEL_SAMPLE_SIZE) -> pd.DataFrame:
    """
    Take a random sample of emails, send each one to Claude to label,
    and return a DataFrame with the labels added.

    This is Step 1 of Stage 1 — we label a sample to use as training data
    for the binary classifier.
    """
    print(f"Sending {sample_size} emails to Claude for labeling...")

    sample = df.sample(n=min(sample_size, len(df)), random_state=42).copy()

    labels = []
    for i, (_, row) in enumerate(sample.iterrows()):
        if i % 50 == 0:
            print(f"  Labeled {i}/{len(sample)} emails...")

        label = label_email(row["body"])
        labels.append(label)

    sample["claude_label"] = labels
    print(f"Labeling complete. Label distribution:")
    print(sample["claude_label"].value_counts().to_string())

    return sample


def ask_question(question: str, context: str) -> str:
    """
    Answer a question about the Enron data in plain English.
    Used by the chat assistant in the Streamlit UI.

    We pass in the analysis results as 'context' so Claude
    can answer based on what the data actually shows.
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
    """Load a previously saved labeled sample."""
    path = Path(output_dir) / "claude_labeled_emails.csv"
    if path.exists():
        print("Loading existing Claude labels (skipping API calls)...")
        return pd.read_csv(path)
    return None
