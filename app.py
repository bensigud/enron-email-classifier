"""
The Social World of Enron — Interactive UI
RAF620M — University of Iceland

Launch with:  streamlit run app.py
"""

import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from pathlib import Path
from src.claude_client import ask_question

# ----------------------------------------------------------------
# Page config — sets the browser tab title and layout
# ----------------------------------------------------------------
st.set_page_config(
    page_title="The Social World of Enron",
    page_icon="📧",
    layout="wide",
)

RESULTS_PATH = Path("data/results")

# ----------------------------------------------------------------
# Colour maps — consistent colours across the whole app
# ----------------------------------------------------------------
PERSON_COLORS = {
    "Hub":          "#e74c3c",   # red
    "Gatekeeper":   "#e67e22",   # orange
    "Inner Circle": "#9b59b6",   # purple
    "Follower":     "#3498db",   # blue
    "Isolated":     "#95a5a6",   # grey
}

RELATIONSHIP_COLORS = {
    "Professional": "#3498db",   # blue
    "Friendly":     "#2ecc71",   # green
    "Hostile":      "#e74c3c",   # red
    "Mentorship":   "#f39c12",   # yellow
    "Romantic":     "#e91e8c",   # pink
}


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
def load_pairs_data():
    path = RESULTS_PATH / "sentiment_pairs.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_romance_data():
    path = RESULTS_PATH / "romance_pairs.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_communities():
    path = RESULTS_PATH / "communities.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def build_context_for_claude(person_df, pairs_df, romance_df):
    """
    Build a text summary of our findings to pass to Claude
    as context when answering questions.
    """
    if person_df is None:
        return "No analysis results available yet. Run main.py first."

    top_hubs = person_df[person_df["person_class"] == "Hub"].nlargest(5, "total_degree")
    top_hostile = pairs_df[pairs_df["relationship_class"] == "Hostile"].nsmallest(3, "avg_sentiment") if pairs_df is not None else pd.DataFrame()
    top_romantic = romance_df.head(5) if romance_df is not None else pd.DataFrame()

    class_dist = person_df["person_class"].value_counts().to_dict()

    context = f"""
ENRON EMAIL ANALYSIS RESULTS

Total people analysed: {len(person_df)}
Person class distribution: {class_dist}

TOP HUBS (most connected people):
{top_hubs[['person','total_degree','betweenness']].to_string(index=False)}

RELATIONSHIP DISTRIBUTION:
{pairs_df['relationship_class'].value_counts().to_string() if pairs_df is not None else 'N/A'}

MOST HOSTILE PAIRS:
{top_hostile[['sender','recipient','avg_sentiment']].to_string(index=False) if not top_hostile.empty else 'None found'}

TOP ROMANTIC/PERSONAL PAIRS:
{top_romantic[['person_a','person_b','avg_romance_score']].to_string(index=False) if not top_romantic.empty else 'None found'}
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
     "❤️‍🔥 Friends & Enemies",
     "💌 Office Romance",
     "🤖 Ask the Data"],
)

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------
person_df  = load_person_data()
pairs_df   = load_pairs_data()
romance_df = load_romance_data()
communities = load_communities()

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

    This app explores those emails through **4 machine learning analyses**:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.info("**🕸️ Social Network**\nWho emailed who? Who were the key players?")
        st.success("**❤️‍🔥 Friends & Enemies**\nSentiment analysis between every pair of people")
    with col2:
        st.warning("**💌 Office Romance**\nPersonal and romantic emails hiding in the data")
        st.error("**🤖 Ask the Data**\nChat with Claude about the findings")

    if data_ready:
        st.markdown("---")
        st.markdown("### Key Numbers")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("People", f"{len(person_df):,}")
        col2.metric("Hubs", f"{(person_df['person_class']=='Hub').sum()}")
        col3.metric("Pairs Analysed", f"{len(pairs_df):,}" if pairs_df is not None else "—")
        col4.metric("Romantic Pairs", f"{len(romance_df):,}" if romance_df is not None else "—")


# ================================================================
# PAGE 2 — Social Network
# ================================================================
elif page == "🕸️ Social Network":
    st.title("🕸️ The Social Network")
    st.markdown("Each dot is a person. Each line is an email connection. Size = how connected they are.")

    # Show the pre-generated network plot
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
                (pairs_df["sender"] == selected) | (pairs_df["recipient"] == selected)
            ].copy()

            if len(person_pairs) > 0:
                person_pairs["other_person"] = person_pairs.apply(
                    lambda r: r["recipient"] if r["sender"] == selected else r["sender"],
                    axis=1
                )
                display = person_pairs[["other_person", "relationship_class",
                                        "avg_sentiment", "email_count"]].head(20)
                display.columns = ["Connected To", "Relationship", "Sentiment", "Emails"]
                st.dataframe(display.reset_index(drop=True), use_container_width=True)
            else:
                st.info("No pair data found for this person.")
    else:
        st.info("Run main.py to generate results.")


# ================================================================
# PAGE 4 — Friends & Enemies
# ================================================================
elif page == "❤️‍🔥 Friends & Enemies":
    st.title("❤️‍🔥 Friends & Enemies")
    st.markdown("Sentiment analysis of email tone between every pair of people.")

    if data_ready and pairs_df is not None:
        plot_path = RESULTS_PATH / "relationship_distribution.png"
        if plot_path.exists():
            st.image(str(plot_path), use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 😡 Most Hostile Pairs")
            hostile = (pairs_df[pairs_df["relationship_class"] == "Hostile"]
                       .nsmallest(10, "avg_sentiment")
                       [["sender", "recipient", "avg_sentiment"]])
            hostile.columns = ["Person A", "Person B", "Sentiment"]
            hostile["Person A"] = hostile["Person A"].str.split("@").str[0]
            hostile["Person B"] = hostile["Person B"].str.split("@").str[0]
            st.dataframe(hostile.reset_index(drop=True), use_container_width=True)

        with col2:
            st.markdown("### 😊 Most Friendly Pairs")
            friendly = (pairs_df[pairs_df["relationship_class"] == "Friendly"]
                        .nlargest(10, "avg_sentiment")
                        [["sender", "recipient", "avg_sentiment"]])
            friendly.columns = ["Person A", "Person B", "Sentiment"]
            friendly["Person A"] = friendly["Person A"].str.split("@").str[0]
            friendly["Person B"] = friendly["Person B"].str.split("@").str[0]
            st.dataframe(friendly.reset_index(drop=True), use_container_width=True)

        st.markdown("---")
        st.markdown("### Look up a specific pair")
        people = sorted(set(pairs_df["sender"].tolist() + pairs_df["recipient"].tolist()))
        col1, col2 = st.columns(2)
        person_a = col1.selectbox("Person A", people, key="pa")
        person_b = col2.selectbox("Person B", people, key="pb", index=1)

        match = pairs_df[
            ((pairs_df["sender"] == person_a) & (pairs_df["recipient"] == person_b)) |
            ((pairs_df["sender"] == person_b) & (pairs_df["recipient"] == person_a))
        ]

        if len(match) > 0:
            row = match.iloc[0]
            cls = row["relationship_class"]
            color = RELATIONSHIP_COLORS.get(cls, "#aaa")
            st.markdown(
                f"<div style='background:{color};padding:12px;border-radius:8px;"
                f"color:white;text-align:center;font-size:1.3em'>"
                f"<b>{cls}</b> relationship — sentiment score: {row['avg_sentiment']:.3f}"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.info("No direct email relationship found between these two people.")
    else:
        st.info("Run main.py to generate results.")


# ================================================================
# PAGE 5 — Office Romance
# ================================================================
elif page == "💌 Office Romance":
    st.title("💌 Office Romance")
    st.markdown("Personal and romantic emails detected by our machine learning classifier.")

    if data_ready and romance_df is not None:
        plot_path = RESULTS_PATH / "romance_pairs.png"
        if plot_path.exists():
            st.image(str(plot_path), use_container_width=True)

        st.markdown("---")
        st.markdown("### Flagged Pairs")
        st.markdown("*Pairs with consistently personal or romantic email patterns*")

        display = romance_df.copy()
        display["Person A"] = display["person_a"].str.split("@").str[0]
        display["Person B"] = display["person_b"].str.split("@").str[0]
        display["Romance Score"] = display["avg_romance_score"].round(3)
        display["Emails"] = display["email_count"]
        st.dataframe(
            display[["Person A", "Person B", "Romance Score", "Emails"]].reset_index(drop=True),
            use_container_width=True
        )
    else:
        st.info("Run main.py to generate results.")


# ================================================================
# PAGE 6 — Ask the Data (Claude Chat)
# ================================================================
elif page == "🤖 Ask the Data":
    st.title("🤖 Ask the Data")
    st.markdown(
        "Ask Claude anything about the Enron analysis results. "
        "Claude answers based on what our data actually shows."
    )

    # Build the context string from our results
    context = build_context_for_claude(person_df, pairs_df, romance_df)

    # Example questions to help the user get started
    st.markdown("**Example questions:**")
    examples = [
        "Who was the most connected person at Enron?",
        "Were there any romantic relationships involving senior executives?",
        "Which employees showed the most hostile communication?",
        "What does the data tell us about Ken Lay's social network?",
        "Which community had the most internal emails?",
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
