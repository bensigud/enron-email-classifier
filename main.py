"""
The Social World of Enron — Main Pipeline Runner
RAF620M — University of Iceland

Run this file to execute the full pipeline:
  python main.py                    # uses default config
  python main.py --config path.json # uses custom config from UI

Results are saved to data/results/ and can then be
explored in the Streamlit UI:
  streamlit run app.py
"""

import json
import sys
import time
from pathlib import Path
from src.loader import load_emails, load_processed, save_processed, filter_executives_only
from src.network import run_network_analysis
from src.stage1 import run_stage1
from src.stage2 import run_stage2
from src.stage3 import run_stage3
from src.stage4 import run_stage4

# Paths
RAW_DATA_PATH = "data/raw/maildir"
PROCESSED_PATH = "data/processed/emails.parquet"
RESULTS_PATH = "data/results"
PROGRESS_PATH = "data/results/pipeline_progress.json"

# Default configuration
DEFAULT_CONFIG = {
    "batch_strategy": "sample",    # "sample", "custom", "full"
    "email_sample_size": 2000,
    "label_sample_size": 50,
    "min_emails_per_pair": 5,
    "k_min": 3,
    "k_max": 7,
    "dbscan_eps_values": [0.5, 0.75, 1.0, 1.5, 2.0],
    "dbscan_min_samples": 5,
}

FULL_CONFIG = {
    "batch_strategy": "full",
    "email_sample_size": 0,        # 0 = all
    "label_sample_size": 500,
    "min_emails_per_pair": 5,
    "k_min": 3,
    "k_max": 8,
    "dbscan_eps_values": [0.5, 0.75, 1.0, 1.5, 2.0],
    "dbscan_min_samples": 5,
}


def _update_progress(stage: str, status: str, detail: str = "",
                     stats: dict = None):
    """Write pipeline progress to a JSON file so the UI can read it."""
    progress_path = Path(PROGRESS_PATH)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
    else:
        progress = {"stages": {}, "started_at": time.time()}

    progress["stages"][stage] = {
        "status": status,
        "detail": detail,
        "timestamp": time.time(),
    }
    if stats:
        progress["stages"][stage]["stats"] = stats

    if status == "done" and stage == "stage4":
        progress["completed_at"] = time.time()

    progress_path.write_text(json.dumps(progress, indent=2))


def load_config(config_path: str = None) -> dict:
    """Load pipeline config from JSON file, or return defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_config = json.load(f)
        config = {**DEFAULT_CONFIG, **user_config}
        print(f"  Config loaded from {config_path}")
    else:
        config = DEFAULT_CONFIG.copy()
        print("  Using default config (sample mode)")
    return config


def main(config: dict = None):
    if config is None:
        config = DEFAULT_CONFIG.copy()

    strategy = config.get("batch_strategy", "sample")
    email_sample = config.get("email_sample_size", 2000)
    label_sample = config.get("label_sample_size", 50)

    # Retrain = full pipeline but no new Claude labels
    if strategy == "retrain":
        email_sample = 0  # all emails
        label_sample = 0  # use cached labels

    print("=" * 60)
    print("  The Social World of Enron — Behavioral Analytics")
    print(f"  Strategy: {strategy.upper()}")
    if email_sample > 0:
        print(f"  Email sample: {email_sample:,}")
    else:
        print(f"  Email sample: ALL")
    print(f"  Label sample: {label_sample:,}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # STEP 1 — Load emails (incremental batching)
    # ----------------------------------------------------------------
    _update_progress("load", "running", "Loading emails...")
    print("\n[1/6] Loading emails...")

    if Path(PROCESSED_PATH).exists():
        df_all = load_processed(PROCESSED_PATH)
    else:
        print("Processed file not found — parsing raw emails...")
        df_all = load_emails(RAW_DATA_PATH)
        save_processed(df_all, PROCESSED_PATH)

    print("  Filtering to executive-only emails...")
    df_all = filter_executives_only(df_all, RAW_DATA_PATH)

    # Sort deterministically so batches are consistent
    df_all = df_all.sort_values("date").reset_index(drop=True)

    if email_sample > 0 and len(df_all) > email_sample:
        # Incremental: read previous offset, take the NEXT batch
        offset_path = Path(RESULTS_PATH) / "batch_offset.json"
        prev_offset = 0
        if offset_path.exists():
            try:
                prev_offset = json.loads(offset_path.read_text()).get("offset", 0)
            except Exception:
                prev_offset = 0

        # Wrap around if we've reached the end
        if prev_offset >= len(df_all):
            prev_offset = 0
            print("  Reached end of dataset — wrapping to start")

        end = min(prev_offset + email_sample, len(df_all))
        df = df_all.iloc[prev_offset:end].reset_index(drop=True)

        # Save new offset
        new_offset = end
        offset_path.parent.mkdir(parents=True, exist_ok=True)
        offset_path.write_text(json.dumps({
            "offset": new_offset,
            "total": len(df_all),
            "batch_size": email_sample,
            "batches_completed": new_offset // email_sample,
        }, indent=2))

        print(f"  Batch: emails {prev_offset + 1:,} – {end:,} "
              f"of {len(df_all):,} ({end / len(df_all) * 100:.1f}%)")
    else:
        df = df_all
        print(f"  Processing ALL {len(df):,} emails")

    total_emails = len(df)
    unique_senders = df["sender"].nunique()
    date_min = str(df["date"].min().date()) if len(df) > 0 else "N/A"
    date_max = str(df["date"].max().date()) if len(df) > 0 else "N/A"

    print(f"  {total_emails:,} emails in this batch")
    print(f"  {unique_senders:,} unique senders")
    print(f"  Date range: {date_min} -> {date_max}")

    _update_progress("load", "done", f"{total_emails:,} emails loaded", {
        "total_emails": total_emails,
        "unique_senders": unique_senders,
        "date_range": f"{date_min} to {date_max}",
    })

    # ----------------------------------------------------------------
    # STEP 2 — Network Analysis
    # ----------------------------------------------------------------
    _update_progress("network", "running", "Building network graph...")
    print("\n[2/6] Running network analysis...")
    graph, person_df = run_network_analysis(df, RESULTS_PATH)

    n_people = len(person_df)
    n_hubs = int((person_df["person_class"] == "Hub").sum())
    _update_progress("network", "done", f"{n_people} people, {n_hubs} hubs", {
        "people": n_people,
        "hubs": n_hubs,
    })

    # ----------------------------------------------------------------
    # STEP 3 — Stage 1: Email-Level Scoring
    # ----------------------------------------------------------------
    # Determine mode: "train" for sample runs, "predict" for full runs
    stage1_mode = config.get("stage1_mode", "train")
    if strategy == "full":
        stage1_mode = "predict"

    if stage1_mode == "train":
        _update_progress("stage1", "running",
                         f"Training — labeling {label_sample} emails with Claude...")
        print("\n[3/6] Running Stage 1 (TRAIN mode)...")
        import src.claude_client as cc
        original_sample_size = cc.LABEL_SAMPLE_SIZE
        cc.LABEL_SAMPLE_SIZE = label_sample
    else:
        _update_progress("stage1", "running",
                         "Predict mode — loading saved models...")
        print("\n[3/6] Running Stage 1 (PREDICT mode — no Claude, no training)...")

    df = run_stage1(df, RESULTS_PATH, mode=stage1_mode)

    if stage1_mode == "train":
        cc.LABEL_SAMPLE_SIZE = original_sample_size

    _update_progress("stage1", "done", f"Scoring complete ({stage1_mode} mode)", {
        "mode": stage1_mode,
        "labels": label_sample if stage1_mode == "train" else "saved models",
        "mean_disclosure": round(float(df["intimacy_score"].mean()), 3),
        "mean_responsiveness": round(float(df["warmth_score"].mean()), 3),
        "mean_sentiment": round(float(df["sentiment_score"].mean()), 3),
    })

    # ----------------------------------------------------------------
    # STEP 4 — Stage 2: Pair-Level Feature Engineering
    # ----------------------------------------------------------------
    _update_progress("stage2", "running", "Building pair features...")
    print("\n[4/6] Running Stage 2 (pair-level features)...")

    # Override min emails per pair
    import src.stage2 as s2
    original_min = s2.MIN_EMAILS_PER_PAIR
    s2.MIN_EMAILS_PER_PAIR = config.get("min_emails_per_pair", 5)

    pair_features = run_stage2(df, person_df, RESULTS_PATH, graph=graph)

    s2.MIN_EMAILS_PER_PAIR = original_min

    n_pairs = len(pair_features)
    n_features = len(s2.FEATURE_COLS)
    _update_progress("stage2", "done", f"{n_pairs} pairs, {n_features} features", {
        "pairs": n_pairs,
        "features": n_features,
    })

    # ----------------------------------------------------------------
    # STEP 5 — Stage 3: Unsupervised Clustering
    # ----------------------------------------------------------------
    _update_progress("stage3", "running", "Clustering relationships...")
    print("\n[5/6] Running Stage 3 (clustering)...")
    pair_features = run_stage3(pair_features, RESULTS_PATH)

    n_clusters = pair_features["cluster"].nunique()
    _update_progress("stage3", "done", f"{n_clusters} clusters found", {
        "clusters": n_clusters,
    })

    # ----------------------------------------------------------------
    # STEP 6 — Stage 4: Cluster Interpretation
    # ----------------------------------------------------------------
    _update_progress("stage4", "running", "Claude naming clusters...")
    print("\n[6/6] Running Stage 4 (cluster interpretation)...")
    pair_features = run_stage4(pair_features, df, RESULTS_PATH)

    _update_progress("stage4", "done", "Pipeline complete")

    # ----------------------------------------------------------------
    # DONE
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  All analyses complete!")
    print(f"  Results saved to: {RESULTS_PATH}/")
    print("  Launch the UI with:  streamlit run app.py")
    print("=" * 60)

    # Save the config that was used
    config_out = Path(RESULTS_PATH) / "pipeline_config.json"
    with open(config_out, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    config_path = None
    if len(sys.argv) > 2 and sys.argv[1] == "--config":
        config_path = sys.argv[2]

    config = load_config(config_path)
    main(config)
