"""
The Social World of Enron — Main Pipeline Runner
RAF620M — University of Iceland

Run this file to execute the full pipeline:
  python main.py

Results are saved to data/results/ and can then be
explored in the Streamlit UI:
  streamlit run app.py
"""

import sys
from pathlib import Path
from src.loader import load_emails, load_processed, save_processed, filter_executives_only
from src.network import run_network_analysis
from src.stage1 import run_stage1
from src.stage2 import run_stage2
from src.stage3 import run_stage3

# Paths
RAW_DATA_PATH = "data/raw/maildir"
PROCESSED_PATH = "data/processed/emails.csv"
RESULTS_PATH = "data/results"


def main():
    print("=" * 60)
    print("  The Social World of Enron — Behavioral Analytics")
    print("=" * 60)

    # ----------------------------------------------------------------
    # STEP 1 — Load emails
    # ----------------------------------------------------------------
    print("\n[1/5] Loading emails...")

    if Path(PROCESSED_PATH).exists():
        df = load_processed(PROCESSED_PATH)
    else:
        print("Processed file not found — parsing raw emails (this takes ~15 mins)...")
        df = load_emails(RAW_DATA_PATH)
        save_processed(df, PROCESSED_PATH)

    # Filter to only emails between executives
    print("  Filtering to executive-only emails...")
    df = filter_executives_only(df, RAW_DATA_PATH)

    # TEST MODE: use a small sample for faster debugging
    # Change to False (or remove) for the full run
    TEST_MODE = True
    if TEST_MODE:
        df = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)
        print(f"  TEST MODE: sampled {len(df):,} emails")

    print(f"  {len(df):,} emails loaded")
    print(f"  {df['sender'].nunique():,} unique senders")
    print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    # ----------------------------------------------------------------
    # STEP 2 — Network Analysis
    # Builds the graph, classifies each person (Hub / Gatekeeper etc.)
    # and detects communities. Produces features used in Stage 2.
    # ----------------------------------------------------------------
    print("\n[2/5] Running network analysis...")
    graph, person_df = run_network_analysis(df, RESULTS_PATH)

    # ----------------------------------------------------------------
    # STEP 3 — Stage 1: Email-Level Scoring
    # Claude labels 500 emails -> train binary classifier (compare 3 models)
    # -> score all emails with personal_score + sentiment_score
    # ----------------------------------------------------------------
    print("\n[3/5] Running Stage 1 (email-level scoring)...")
    df = run_stage1(df, RESULTS_PATH)

    # ----------------------------------------------------------------
    # STEP 4 — Stage 2: Pair-Level Relationship Clustering
    # Build 17-feature vector per pair -> cluster with K-Means + DBSCAN
    # -> interpret clusters as relationship types
    # ----------------------------------------------------------------
    print("\n[4/5] Running Stage 2 (pair-level clustering)...")
    pair_features = run_stage2(df, person_df, RESULTS_PATH)

    # ----------------------------------------------------------------
    # STEP 5 — Stage 3: Cluster Interpretation
    # Compute z-scores per cluster, pull sample emails, ask Claude
    # to name each relationship type
    # ----------------------------------------------------------------
    print("\n[5/5] Running Stage 3 (cluster interpretation)...")
    pair_features = run_stage3(pair_features, df, RESULTS_PATH)

    # ----------------------------------------------------------------
    # DONE
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  All analyses complete!")
    print(f"  Results saved to: {RESULTS_PATH}/")
    print("  Launch the UI with:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
