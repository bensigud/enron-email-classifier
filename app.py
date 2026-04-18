"""
The Social World of Enron — Interactive UI
RAF620M — University of Iceland

Launch with:  streamlit run app.py
"""

import json
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from pyvis.network import Network
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

RELATIONSHIP_COLORS = {
    "Professional":    "#3498db",
    "Professional Network": "#3498db",
    "Friendly":        "#2ecc71",
    "Friendly Colleagues": "#2ecc71",
    "Hostile":         "#e74c3c",
    "Mentorship":      "#f39c12",
    "Romantic":        "#e91e8c",
    "Executive Outreach": "#e67e22",
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


def _short_name(email_addr):
    """Convert email to display name: john.lavorato@enron.com -> John Lavorato"""
    return email_addr.split("@")[0].replace(".", " ").title() if email_addr else ""


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


def build_interactive_network(person_df, pairs_df, cluster_names_dict,
                              selected_person=None, max_nodes=100):
    """
    Build an interactive pyvis network graph.
    If a person is selected, highlight them and their connections.
    Returns HTML string.
    """
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=False,
    )
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.01,
    )

    # Get the people who appear in relationship pairs (they have interesting data)
    pair_people = set()
    if pairs_df is not None:
        pair_people = set(pairs_df["person_a"].tolist() + pairs_df["person_b"].tolist())

    # Pick which nodes to show
    if selected_person and selected_person in person_df["person"].values:
        # Show selected person + their connections + some top people
        connected = set()
        if pairs_df is not None:
            connected = set(
                pairs_df[pairs_df["person_a"] == selected_person]["person_b"].tolist() +
                pairs_df[pairs_df["person_b"] == selected_person]["person_a"].tolist()
            )
        # Also include top connected people for context
        top_people = set(person_df.nlargest(30, "total_degree")["person"].tolist())
        show_people = {selected_person} | connected | top_people
    else:
        # Show top connected people + all people in pairs
        top_people = set(person_df.nlargest(max_nodes, "total_degree")["person"].tolist())
        show_people = top_people | pair_people

    feat = person_df.set_index("person")

    # Add nodes
    for person in show_people:
        if person not in feat.index:
            continue
        row = feat.loc[person]
        cls = row["person_class"]
        color = PERSON_COLORS.get(cls, "#aaa")
        degree = int(row["total_degree"])
        size = max(10, min(50, degree * 0.5))
        name = _short_name(person)

        # Highlight selected person
        border_width = 1
        border_color = color
        if person == selected_person:
            border_width = 4
            border_color = "#ffffff"
            size = max(size, 30)

        title = (
            f"<b>{name}</b><br>"
            f"Email: {person}<br>"
            f"Class: {cls}<br>"
            f"Connections: {degree}<br>"
            f"Betweenness: {row['betweenness']:.3f}<br>"
            f"PageRank: {row['pagerank']:.4f}"
        )

        net.add_node(
            person,
            label=name,
            title=title,
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": "#ffffff", "border": color},
            },
            size=size,
            borderWidth=border_width,
            font={"size": 10, "color": "white"},
        )

    # Add edges from relationship pairs
    if pairs_df is not None:
        for _, row in pairs_df.iterrows():
            a, b = row["person_a"], row["person_b"]
            if a in show_people and b in show_people:
                if a in feat.index and b in feat.index:
                    cluster_id = int(row["cluster"])
                    rel_name = cluster_names_dict.get(cluster_id, {}).get("name", f"Cluster {cluster_id}")
                    edge_color = RELATIONSHIP_COLORS.get(rel_name, "#555555")

                    # Highlight edges connected to selected person
                    width = 1.5
                    if selected_person and (a == selected_person or b == selected_person):
                        width = 4

                    title = (
                        f"{_short_name(a)} ↔ {_short_name(b)}<br>"
                        f"Type: {rel_name}<br>"
                        f"Intimacy: {row['avg_intimacy']:.3f}<br>"
                        f"Warmth: {row['avg_warmth']:.3f}<br>"
                        f"Sentiment: {row['avg_sentiment']:.3f}<br>"
                        f"Emails: {int(row['email_count'])}"
                    )

                    net.add_edge(
                        a, b,
                        color=edge_color,
                        width=width,
                        title=title,
                    )

    # Generate HTML
    html = net.generate_html()

    # Inject custom JS: when a node is clicked, update the URL query param
    # so Streamlit can read it
    click_js = """
    <script>
    // After the network is drawn, add click handler
    var checkNetwork = setInterval(function() {
        if (typeof network !== 'undefined') {
            clearInterval(checkNetwork);
            network.on("doubleClick", function(params) {
                if (params.nodes.length > 0) {
                    var nodeId = params.nodes[0];
                    // Send to parent Streamlit frame via URL
                    window.parent.postMessage({
                        type: "streamlit:setComponentValue",
                        value: nodeId
                    }, "*");
                    // Also update URL for Streamlit to read
                    var url = new URL(window.parent.location);
                    url.searchParams.set("selected", nodeId);
                    window.parent.history.replaceState({}, "", url);
                    window.parent.location.reload();
                }
            });
        }
    }, 100);
    </script>
    """
    html = html.replace("</body>", click_js + "</body>")

    return html


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

    if pairs_df is not None:
        top_personal = pairs_df.nlargest(5, "avg_intimacy")
        context += f"""
MOST PERSONAL PAIRS:
{top_personal[['person_a','person_b','avg_intimacy','avg_sentiment']].to_string(index=False)}
"""
        most_negative = pairs_df.nsmallest(5, "avg_sentiment")
        context += f"""
MOST NEGATIVE SENTIMENT PAIRS:
{most_negative[['person_a','person_b','avg_sentiment','avg_intimacy']].to_string(index=False)}
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

# Check URL for selected person (from graph click)
query_params = st.query_params
if "selected" in query_params:
    st.session_state["selected_person"] = query_params["selected"]
    # Switch to person profile page
    if page != "👤 Person Profile":
        st.query_params["selected"] = query_params["selected"]

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

    This app explores those emails through a **four-stage ML pipeline**:
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("**📊 Stage 1: Email Scoring**\nIntimacy + warmth classifiers + VADER sentiment")
    with col2:
        st.warning("**🔧 Stage 2: Pair Features**\n20-feature vector per pair (Gilbert dimensions)")
    with col3:
        st.success("**🔗 Stage 3: Clustering**\nK-Means + DBSCAN discover relationship types")
    with col4:
        st.error("**🏷️ Stage 4: Naming**\nClaude interprets and names each cluster")

    if data_ready:
        st.markdown("---")
        st.markdown("### Key Numbers")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("People", f"{len(person_df):,}")
        col2.metric("Hubs", f"{(person_df['person_class']=='Hub').sum()}")
        col3.metric("Pairs Analysed", f"{len(pairs_df):,}" if pairs_df is not None else "—")
        col4.metric("Relationship Types",
                     f"{len(cluster_names)}" if cluster_names else "—")

        # Show model comparison if available
        comp_img = RESULTS_PATH / "model_comparison.png"
        if comp_img.exists():
            st.markdown("---")
            st.markdown("### Stage 1: Model Comparison (Intimacy + Warmth)")
            st.image(str(comp_img), use_container_width=True)

            st.markdown("### Confusion Matrices")
            col1, col2 = st.columns(2)
            cm_intimacy = RESULTS_PATH / "confusion_matrix_intimacy.png"
            cm_warmth = RESULTS_PATH / "confusion_matrix_warmth.png"
            if cm_intimacy.exists():
                col1.image(str(cm_intimacy), use_container_width=True)
            if cm_warmth.exists():
                col2.image(str(cm_warmth), use_container_width=True)


# ================================================================
# PAGE 2 — Social Network (Interactive)
# ================================================================
elif page == "🕸️ Social Network":
    st.title("🕸️ The Social Network")

    if data_ready:
        st.markdown(
            "Hover over a node to see details. **Double-click a person** to go to their profile. "
            "Drag nodes to rearrange. Scroll to zoom."
        )

        # Person class legend
        cols = st.columns(5)
        for i, (cls, color) in enumerate(PERSON_COLORS.items()):
            cols[i].markdown(
                f"<div style='background:{color};padding:6px;border-radius:6px;"
                f"color:white;text-align:center;font-weight:bold;font-size:0.85em'>{cls}</div>",
                unsafe_allow_html=True
            )

        # Optional: filter to show a specific person's network
        people_in_pairs = sorted(set(
            (pairs_df["person_a"].tolist() + pairs_df["person_b"].tolist())
            if pairs_df is not None else []
        ))
        focus_options = ["Everyone (top connected)"] + [
            f"{_short_name(p)} ({p})" for p in people_in_pairs
        ]
        focus = st.selectbox("Focus on a person", focus_options)

        selected = None
        if focus != "Everyone (top connected)":
            # Extract email from the label
            selected = focus.split("(")[1].rstrip(")")

        # Build and display the interactive graph
        html = build_interactive_network(
            person_df, pairs_df, cluster_names, selected_person=selected
        )
        components.html(html, height=620, scrolling=False)

        # Relationship type legend (edge colors)
        if cluster_names:
            st.markdown("---")
            st.markdown("### Edge Colors = Relationship Types")
            cols = st.columns(min(len(cluster_names), 4))
            for i, (cid, info) in enumerate(sorted(cluster_names.items())):
                color = RELATIONSHIP_COLORS.get(info["name"], CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])
                cols[i % len(cols)].markdown(
                    f"<div style='background:{color};padding:6px;border-radius:6px;"
                    f"color:white;text-align:center;font-size:0.85em'>"
                    f"{info['name']}</div>",
                    unsafe_allow_html=True
                )

        # Stats below the graph
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Class Distribution")
            dist = person_df["person_class"].value_counts().reset_index()
            dist.columns = ["Class", "Count"]
            st.bar_chart(dist.set_index("Class"))

        with col2:
            st.markdown("### Top 15 Most Connected")
            top = person_df.nlargest(15, "total_degree")[
                ["person", "person_class", "total_degree", "betweenness"]
            ].reset_index(drop=True)
            top["Name"] = top["person"].apply(_short_name)
            top.index += 1
            st.dataframe(
                top[["Name", "person_class", "total_degree", "betweenness"]],
                use_container_width=True
            )
    else:
        st.info("Run main.py to generate results.")


# ================================================================
# PAGE 3 — Person Profile
# ================================================================
elif page == "👤 Person Profile":
    st.title("👤 Person Profile")
    st.markdown("Select any Enron executive to see their social profile and relationships.")

    if data_ready:
        # Only show people who appear in pairs (executives with relationships)
        people_in_pairs = set()
        if pairs_df is not None:
            people_in_pairs = set(
                pairs_df["person_a"].tolist() + pairs_df["person_b"].tolist()
            )

        # All people sorted by degree, but prioritize those with relationships
        all_people = sorted(person_df["person"].tolist())
        pair_people_sorted = sorted(people_in_pairs)

        # Default selection from URL or session state
        default_person = pair_people_sorted[0] if pair_people_sorted else all_people[0]
        if "selected_person" in st.session_state:
            sp = st.session_state["selected_person"]
            if sp in all_people:
                default_person = sp

        default_idx = all_people.index(default_person) if default_person in all_people else 0

        selected = st.selectbox(
            "Choose a person",
            all_people,
            index=default_idx,
            format_func=lambda p: f"{_short_name(p)} ({p})"
        )

        row = person_df[person_df["person"] == selected].iloc[0]
        cls = row["person_class"]
        color = PERSON_COLORS.get(cls, "#aaa")

        st.markdown(f"""
        <div style='background:{color};padding:16px;border-radius:10px;color:white;margin:12px 0'>
            <h2 style='margin:0'>{_short_name(selected)}</h2>
            <p style='margin:4px 0;font-size:1.1em'>{selected}</p>
            <h3 style='margin:4px 0'>Class: {cls}</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Emails Sent To", int(row["out_degree"]))
        col2.metric("Emails Received From", int(row["in_degree"]))
        col3.metric("Betweenness", f"{row['betweenness']:.3f}")
        col4.metric("PageRank", f"{row['pagerank']:.4f}")

        # Show their mini network graph
        if pairs_df is not None:
            person_pairs = pairs_df[
                (pairs_df["person_a"] == selected) | (pairs_df["person_b"] == selected)
            ].copy()

            if len(person_pairs) > 0:
                st.markdown("---")
                st.markdown("### Their Network")
                mini_html = build_interactive_network(
                    person_df, pairs_df, cluster_names, selected_person=selected
                )
                components.html(mini_html, height=450, scrolling=False)

                st.markdown("### Relationships")
                person_pairs["other_person"] = person_pairs.apply(
                    lambda r: r["person_b"] if r["person_a"] == selected else r["person_a"],
                    axis=1
                )
                person_pairs["Name"] = person_pairs["other_person"].apply(_short_name)

                if "relationship_type" in person_pairs.columns and cluster_names:
                    type_col = "relationship_type"
                else:
                    person_pairs["relationship_type"] = person_pairs["cluster"].apply(
                        lambda c: cluster_names.get(int(c), {}).get("name", f"Cluster {c}")
                    )
                    type_col = "relationship_type"

                display = person_pairs[[
                    "Name", type_col, "avg_intimacy",
                    "avg_sentiment", "email_count"
                ]]
                display.columns = ["Person", "Relationship", "Intimacy",
                                   "Sentiment", "Emails"]
                st.dataframe(display.reset_index(drop=True), use_container_width=True)

                # Click on a relationship to see emails
                st.markdown("---")
                st.markdown("### 📨 Read Their Emails")
                pair_labels = [
                    f"{_short_name(r['other_person'])} — {r.get('relationship_type', 'Unknown')}"
                    for _, r in person_pairs.iterrows()
                ]
                if pair_labels:
                    selected_pair = st.selectbox("Select a relationship", pair_labels)
                    idx = pair_labels.index(selected_pair)
                    other = person_pairs.iloc[idx]["other_person"]
                    show_email_viewer(load_emails(), selected, other)
            else:
                st.info("No relationship data found for this person. "
                        "They may not have enough emails with other executives.")
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

        # Cluster profiles (heatmap)
        st.markdown("---")
        st.markdown("### Cluster Feature Profiles")
        st.markdown("*What makes each cluster distinct — average feature values per cluster*")
        heatmap_img = RESULTS_PATH / "cluster_heatmap.png"
        if heatmap_img.exists():
            st.image(str(heatmap_img), use_container_width=True)

        # Cluster names summary
        if cluster_names:
            st.markdown("---")
            st.markdown("### Relationship Types Discovered")
            for cid in sorted(cluster_names.keys()):
                info = cluster_names[cid]
                count = (pairs_df["cluster"] == cid).sum()
                color = RELATIONSHIP_COLORS.get(info["name"], CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])
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
        cluster_pairs["Person A"] = cluster_pairs["person_a"].apply(_short_name)
        cluster_pairs["Person B"] = cluster_pairs["person_b"].apply(_short_name)

        display_cols = ["Person A", "Person B", "avg_intimacy",
                        "avg_warmth", "avg_sentiment", "email_count"]
        st.dataframe(
            cluster_pairs[display_cols]
            .sort_values("avg_intimacy", ascending=False)
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
        person_a = col1.selectbox(
            "Person A", all_people, key="pa",
            format_func=lambda p: _short_name(p)
        )
        person_b = col2.selectbox(
            "Person B", all_people, key="pb", index=min(1, len(all_people)-1),
            format_func=lambda p: _short_name(p)
        )

        match = pairs_df[
            ((pairs_df["person_a"] == person_a) & (pairs_df["person_b"] == person_b)) |
            ((pairs_df["person_a"] == person_b) & (pairs_df["person_b"] == person_a))
        ]

        if len(match) > 0:
            row = match.iloc[0]
            cluster_id = int(row["cluster"])
            cluster_label = cluster_names.get(cluster_id, {}).get("name", f"Cluster {cluster_id}")
            color = RELATIONSHIP_COLORS.get(cluster_label, CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)])
            st.markdown(
                f"<div style='background:{color};padding:12px;border-radius:8px;"
                f"color:white;text-align:center;font-size:1.3em'>"
                f"<b>{cluster_label}</b> — "
                f"Intimacy: {row['avg_intimacy']:.3f} | "
                f"Warmth: {row.get('avg_warmth', 0):.3f} | "
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
