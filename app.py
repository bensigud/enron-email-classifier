"""
The Social World of Enron — Interactive UI
RAF620M — University of Iceland

Launch with:  streamlit run app.py
"""

import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from src.claude_client import ask_question

# ----------------------------------------------------------------
# Page config
# ----------------------------------------------------------------
st.set_page_config(
    page_title="The Social World of Enron",
    page_icon="📧",
    layout="wide",
)

RESULTS_PATH = Path("data/results")

# ----------------------------------------------------------------
# Colour maps
# ----------------------------------------------------------------
PERSON_COLORS = {
    "Hub":          "#e74c3c",
    "Gatekeeper":   "#e67e22",
    "Inner Circle": "#9b59b6",
    "Follower":     "#3498db",
    "Isolated":     "#95a5a6",
}

CLUSTER_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#e91e8c", "#1abc9c", "#95a5a6",
]


# ----------------------------------------------------------------
# Data loading — cached so it only loads once per session
# ----------------------------------------------------------------
@st.cache_data
def load_person_data():
    path = RESULTS_PATH / "person_classes.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_relationship_data():
    path = RESULTS_PATH / "relationship_pairs.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_cluster_profiles():
    path = RESULTS_PATH / "cluster_profiles.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_cluster_names():
    path = RESULTS_PATH / "cluster_names.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


@st.cache_data
def load_clustering_meta():
    path = RESULTS_PATH / "clustering_meta.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_model_comparison():
    path = RESULTS_PATH / "model_comparison.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_communities():
    path = RESULTS_PATH / "communities.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_emails():
    path = Path("data/processed/emails.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df


def show_email_viewer(emails_df, person_a, person_b):
    """Show emails between two people in an expandable viewer."""
    if emails_df is None:
        st.info("Email data not loaded.")
        return

    mask = (
        (emails_df["sender"].eq(person_a) &
         emails_df["recipients"].str.contains(person_b, na=False, regex=False)) |
        (emails_df["sender"].eq(person_b) &
         emails_df["recipients"].str.contains(person_a, na=False, regex=False))
    )
    matched = emails_df[mask].sort_values("date", ascending=False)

    if len(matched) == 0:
        st.info(f"No emails found between {person_a} and {person_b}.")
        return

    st.markdown(f"**Found {len(matched)} emails between these two**")

    for i, (_, row) in enumerate(matched.head(20).iterrows()):
        date_str = row["date"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["date"]) else "Unknown"
        subject = row["subject"] if isinstance(row["subject"], str) else "(no subject)"
        with st.expander(f"📧 {date_str} — {subject}"):
            st.markdown(f"**From:** {row['sender']}")
            st.markdown(f"**To:** {row['recipients']}")
            st.markdown(f"**Date:** {date_str}")
            st.markdown(f"**Subject:** {subject}")
            st.markdown("---")
            body = row["body"] if isinstance(row["body"], str) else "(empty)"
            st.text(body[:2000])

    if len(matched) > 20:
        st.caption(f"Showing 20 of {len(matched)} emails.")


def build_context_for_claude(person_df, pairs_df, cluster_profiles):
    """Build a text summary of findings to pass to Claude as context."""
    if person_df is None:
        return "No analysis results available yet. Run main.py first."

    top_hubs = person_df[person_df["person_class"] == "Hub"].nlargest(5, "total_degree")
    class_dist = person_df["person_class"].value_counts().to_dict()

    context = f"""
ENRON EMAIL ANALYSIS RESULTS

Total people analysed: {len(person_df)}
Person class distribution: {class_dist}

TOP HUBS (most connected people):
{top_hubs[['person','total_degree','betweenness']].to_string(index=False)}
"""

    if pairs_df is not None:
        context += f"""
RELATIONSHIP PAIRS ANALYSED: {len(pairs_df)}
Number of relationship clusters: {pairs_df['cluster'].nunique()}

CLUSTER DISTRIBUTION:
{pairs_df['cluster'].value_counts().to_string()}
"""

    if cluster_profiles is not None:
        context += f"""
CLUSTER PROFILES (average features per cluster):
{cluster_profiles.to_string(index=False)}
"""

    # Most personal pairs
    if pairs_df is not None:
        top_personal = pairs_df.nlargest(5, "avg_personal_score")
        context += f"""
MOST PERSONAL PAIRS:
{top_personal[['person_a','person_b','avg_personal_score','avg_sentiment']].to_string(index=False)}
"""

        # Most negative pairs
        most_negative = pairs_df.nsmallest(5, "avg_sentiment")
        context += f"""
MOST NEGATIVE SENTIMENT PAIRS:
{most_negative[['person_a','person_b','avg_sentiment','avg_personal_score']].to_string(index=False)}
"""

    return context


# ----------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------
st.sidebar.title("📧 Enron Social World")
st.sidebar.markdown("*RAF620M — University of Iceland*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview",
     "🕸️ Social Network",
     "👤 Person Profile",
     "🔗 Relationships",
     "🤖 Ask the Data"],
)

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------
person_df = load_person_data()
pairs_df = load_relationship_data()
cluster_profiles = load_cluster_profiles()
clustering_meta = load_clustering_meta()
model_comparison = load_model_comparison()
cluster_names = load_cluster_names()

data_ready = person_df is not None


if not data_ready:
    st.warning("⚠️ No results found. Please run `python main.py` first to generate the analysis.")


# ================================================================
# PAGE 1 — Overview
# ================================================================
if page == "🏠 Overview":
    st.title("The Social World of Enron")
    st.markdown("### A Behavioral Analytics Study of Corporate Fraud")
    st.markdown("""
    In 2001, Enron Corporation collapsed in one of the largest corporate fraud
    scandals in history. The investigation released **~500,000 internal emails**
    — giving us a rare window into the private world of a company falling apart.

    This app explores those emails through a **two-stage ML pipeline**:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.info("**🕸️ Network Analysis**\nWho emailed who? Who were the key players?")
        st.success("**📊 Stage 1: Email Scoring**\nBinary classifier + VADER sentiment on every email")
    with col2:
        st.warning("**🔗 Stage 2: Relationship Clustering**\nUnsupervised clustering discovers relationship types")
        st.error("**🤖 Ask the Data**\nChat with Claude about the findings")

    if data_ready:
        st.markdown("---")
        st.markdown("### Key Numbers")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("People", f"{len(person_df):,}")
        col2.metric("Hubs", f"{(person_df['person_class']=='Hub').sum()}")
        col3.metric("Pairs Analysed", f"{len(pairs_df):,}" if pairs_df is not None else "—")
        col4.metric("Clusters Found",
                     f"{pairs_df['cluster'].nunique()}" if pairs_df is not None else "—")

        # Show model comparison if available
        if model_comparison:
            st.markdown("---")
            st.markdown("### Stage 1: Model Comparison")
            comp_img = RESULTS_PATH / "model_comparison.png"
            if comp_img.exists():
                st.image(str(comp_img), use_container_width=True)

            cm_img = RESULTS_PATH / "confusion_matrix.png"
            if cm_img.exists():
                st.markdown("### Confusion Matrix (Best Model)")
                st.image(str(cm_img), use_container_width=True)


# ================================================================
# PAGE 2 — Social Network
# ================================================================
elif page == "🕸️ Social Network":
    st.title("🕸️ The Social Network")
    st.markdown("Each dot is a person. Each line is an email connection. Size = how connected they are.")

    plot_path = RESULTS_PATH / "network_plot.png"
    if plot_path.exists():
        st.image(str(plot_path), use_container_width=True)
    else:
        st.info("Network plot not generated yet. Run main.py first.")

    if data_ready:
        st.markdown("---")
        st.markdown("### Person Class Legend")
        cols = st.columns(5)
        for i, (cls, color) in enumerate(PERSON_COLORS.items()):
            cols[i].markdown(
                f"<div style='background:{color};padding:8px;border-radius:6px;"
                f"color:white;text-align:center;font-weight:bold'>{cls}</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### Class Distribution")
        dist = person_df["person_class"].value_counts().reset_index()
        dist.columns = ["Class", "Count"]
        st.bar_chart(dist.set_index("Class"))

        st.markdown("### Top 20 Most Connected People")
        top = person_df.nlargest(20, "total_degree")[
            ["person", "person_class", "total_degree", "betweenness", "pagerank"]
        ].reset_index(drop=True)
        top.index += 1
        st.dataframe(top, use_container_width=True)


# ================================================================
# PAGE 3 — Person Profile
# ================================================================
elif page == "👤 Person Profile":
    st.title("👤 Person Profile")
    st.markdown("Select any Enron employee to see their social profile.")

    if data_ready:
        people = sorted(person_df["person"].tolist())
        selected = st.selectbox("Choose a person", people)

        row = person_df[person_df["person"] == selected].iloc[0]
        cls = row["person_class"]
        color = PERSON_COLORS.get(cls, "#aaa")

        st.markdown(f"""
        <div style='background:{color};padding:16px;border-radius:10px;color:white;margin:12px 0'>
            <h2 style='margin:0'>{selected.split('@')[0].replace('.', ' ').title()}</h2>
            <p style='margin:4px 0;font-size:1.1em'>{selected}</p>
            <h3 style='margin:4px 0'>Class: {cls}</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Emails Sent To", int(row["out_degree"]))
        col2.metric("Emails Received From", int(row["in_degree"]))
        col3.metric("Betweenness", f"{row['betweenness']:.3f}")
        col4.metric("PageRank", f"{row['pagerank']:.4f}")

        if pairs_df is not None:
            st.markdown("---")
            st.markdown("### Relationships")
            person_pairs = pairs_df[
                (pairs_df["person_a"] == selected) | (pairs_df["person_b"] == selected)
            ].copy()

            if len(person_pairs) > 0:
                person_pairs["other_person"] = person_pairs.apply(
                    lambda r: r["person_b"] if r["person_a"] == selected else r["person_a"],
                    axis=1
                )
                # Use relationship_type if available, fall back to cluster number
                if "relationship_type" in person_pairs.columns and cluster_names:
                    type_col = "relationship_type"
                else:
                    person_pairs["relationship_type"] = person_pairs["cluster"].apply(
                        lambda c: cluster_names.get(int(c), {}).get("name", f"Cluster {c}")
                    )
                    type_col = "relationship_type"
                display = person_pairs[[
                    "other_person", type_col, "avg_personal_score",
                    "avg_sentiment", "email_count"
                ]].head(20)
                display.columns = ["Connected To", "Relationship", "Personal Score",
                                   "Sentiment", "Emails"]
                st.dataframe(display.reset_index(drop=True), use_container_width=True)
            else:
                st.info("No pair data found for this person.")
    else:
        st.info("Run main.py to generate results.")


# ================================================================
# PAGE 4 — Relationships
# ================================================================
elif page == "🔗 Relationships":
    st.title("🔗 Relationships")
    st.markdown("Relationship types discovered by unsupervised clustering of pair features.")

    if data_ready and pairs_df is not None:

        # Clustering results overview
        st.markdown("### Clustering Results")

        if clustering_meta:
            method = clustering_meta.get("best_method", "unknown")
            score = clustering_meta.get("best_score", 0)
            st.markdown(f"**Best method:** {method.upper()} — "
                        f"**Silhouette score:** {score:.3f}")

        sil_img = RESULTS_PATH / "silhouette_scores.png"
        if sil_img.exists():
            st.image(str(sil_img), use_container_width=True)

        # Cluster distribution
        st.markdown("---")
        st.markdown("### Cluster Distribution")
        dist_img = RESULTS_PATH / "cluster_distribution.png"
        if dist_img.exists():
            st.image(str(dist_img), use_container_width=True)

        # Cluster profiles (heatmap)
        st.markdown("---")
        st.markdown("### Cluster Feature Profiles")
        st.markdown("*What makes each cluster distinct — average feature values per cluster*")
        heatmap_img = RESULTS_PATH / "cluster_heatmap.png"
        if heatmap_img.exists():
            st.image(str(heatmap_img), use_container_width=True)

        if cluster_profiles is not None:
            st.dataframe(cluster_profiles, use_container_width=True)

        # Cluster names summary
        if cluster_names:
            st.markdown("---")
            st.markdown("### Relationship Types Discovered")
            for cid in sorted(cluster_names.keys()):
                info = cluster_names[cid]
                count = (pairs_df["cluster"] == cid).sum()
                color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
                st.markdown(
                    f"<div style='background:{color};padding:10px;border-radius:8px;"
                    f"color:white;margin:6px 0'>"
                    f"<b>{info['name']}</b> ({count} pairs) — "
                    f"{info.get('description', '')}</div>",
                    unsafe_allow_html=True
                )

        # Filter by cluster
        st.markdown("---")
        st.markdown("### Explore by Relationship Type")
        clusters = sorted(pairs_df["cluster"].unique())
        cluster_labels = {
            c: cluster_names.get(int(c), {}).get("name", f"Cluster {c}")
            for c in clusters
        }
        selected_label = st.selectbox(
            "Select a relationship type",
            clusters,
            format_func=lambda c: f"{cluster_labels[c]} ({(pairs_df['cluster']==c).sum()} pairs)"
        )

        cluster_pairs = pairs_df[pairs_df["cluster"] == selected_label].copy()
        cluster_pairs["Person A"] = cluster_pairs["person_a"].str.split("@").str[0]
        cluster_pairs["Person B"] = cluster_pairs["person_b"].str.split("@").str[0]

        display_cols = ["Person A", "Person B", "avg_personal_score",
                        "avg_sentiment", "email_count", "personal_imbalance"]
        st.dataframe(
            cluster_pairs[display_cols]
            .sort_values("avg_personal_score", ascending=False)
            .head(30)
            .reset_index(drop=True),
            use_container_width=True
        )

        # Look up a specific pair
        st.markdown("---")
        st.markdown("### Look Up a Specific Pair")
        all_people = sorted(set(
            pairs_df["person_a"].tolist() + pairs_df["person_b"].tolist()
        ))
        col1, col2 = st.columns(2)
        person_a = col1.selectbox("Person A", all_people, key="pa")
        person_b = col2.selectbox("Person B", all_people, key="pb", index=1)

        match = pairs_df[
            ((pairs_df["person_a"] == person_a) & (pairs_df["person_b"] == person_b)) |
            ((pairs_df["person_a"] == person_b) & (pairs_df["person_b"] == person_a))
        ]

        if len(match) > 0:
            row = match.iloc[0]
            cluster_id = int(row["cluster"])
            cluster_label = cluster_names.get(cluster_id, {}).get("name", f"Cluster {cluster_id}")
            color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
            st.markdown(
                f"<div style='background:{color};padding:12px;border-radius:8px;"
                f"color:white;text-align:center;font-size:1.3em'>"
                f"<b>{cluster_label}</b> — "
                f"Personal: {row['avg_personal_score']:.3f} | "
                f"Sentiment: {row['avg_sentiment']:.3f} | "
                f"Emails: {int(row['email_count'])}"
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown("---")
            st.markdown("### 📨 Their Emails")
            show_email_viewer(load_emails(), person_a, person_b)
        else:
            st.info("No direct relationship found between these two people.")

    else:
        st.info("Run main.py to generate results.")


# ================================================================
# PAGE 5 — Ask the Data (Claude Chat)
# ================================================================
elif page == "🤖 Ask the Data":
    st.title("🤖 Ask the Data")
    st.markdown(
        "Ask Claude anything about the Enron analysis results. "
        "Claude answers based on what our data actually shows."
    )

    context = build_context_for_claude(person_df, pairs_df, cluster_profiles)

    st.markdown("**Example questions:**")
    examples = [
        "Who was the most connected person at Enron?",
        "How many relationship types did the clustering find?",
        "Which pairs had the most personal communication?",
        "What does the data tell us about Ken Lay's social network?",
        "Which cluster seems to represent hostile relationships?",
    ]
    for ex in examples:
        if st.button(ex):
            st.session_state["question"] = ex

    st.markdown("---")
    question = st.text_input(
        "Your question:",
        value=st.session_state.get("question", ""),
        placeholder="Ask anything about the Enron data..."
    )

    if st.button("Ask Claude", type="primary") and question:
        with st.spinner("Claude is thinking..."):
            answer = ask_question(question, context)
        st.markdown("### Answer")
        st.markdown(answer)
