"""
The Social World of Enron — Interactive UI
RAF620M — University of Iceland

Launch with:  streamlit run app.py
"""

import json
import sys
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pyvis.network import Network
from src.claude_client import ask_question
from src.stage2 import FEATURE_COLS

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
    "Employee":     "#1abc9c",
    "External":     "#f1c40f",
}

PERSON_CLASS_DESCRIPTIONS = {
    "Hub":          "Top 10% by connections — central to the network",
    "Gatekeeper":   "Top 15% by betweenness — controls information flow",
    "Inner Circle": "High clustering + low reach — tight local group",
    "Follower":     "Average connectivity — regular employee",
    "Isolated":     "Bottom 10% by connections — peripheral",
    "Employee":     "Enron employee — no mailbox in dataset",
    "External":     "Non-executive — personal or business contact",
}

CLUSTER_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#e91e8c", "#1abc9c", "#95a5a6",
]

RELATIONSHIP_COLORS = {
    "Transactional":        "#aaaaaa",
    "Friendly Colleagues":  "#2ecc71",
    "Close": "#a855f7",
    "Boss-Employee":        "#3b82f6",
    "Mentor":               "#f59e0b",
    "Romance":              "#ec4899",
    "Tense / Conflict":     "#ef4444",
    "Fading":               "#6b7280",
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
def load_communities():
    path = RESULTS_PATH / "communities.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_external_contacts():
    path = RESULTS_PATH / "external_contacts.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_emails():
    parquet_path = Path("data/processed/emails.parquet")
    csv_path = Path("data/processed/emails.csv")
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df


@st.cache_data
def load_claude_labeled():
    path = RESULTS_PATH / "claude_labeled_emails.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_human_labels():
    """Load human labels (not cached — needs to reflect latest saves)."""
    path = RESULTS_PATH / "human_labels.csv"
    if not path.exists():
        return pd.DataFrame(columns=["message_id", "annotator", "intimacy_label", "warmth_label"])
    return pd.read_csv(path)


def save_human_label(message_id, annotator, intimacy, warmth):
    """Append one human label to the CSV file."""
    path = RESULTS_PATH / "human_labels.csv"
    new_row = pd.DataFrame([{
        "message_id": message_id,
        "annotator": annotator,
        "intimacy_label": intimacy,
        "warmth_label": warmth,
    }])
    if path.exists():
        existing = pd.read_csv(path)
        # Replace if same annotator already labeled this email
        existing = existing[
            ~((existing["message_id"] == message_id) & (existing["annotator"] == annotator))
        ]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(path, index=False)


def load_pair_labels():
    """Load human pair-level relationship labels."""
    path = RESULTS_PATH / "human_pair_labels.csv"
    if not path.exists():
        return pd.DataFrame(columns=[
            "person_a", "person_b", "annotator", "relationship_type",
            "confidence", "notes",
        ])
    return pd.read_csv(path)


def save_pair_label(person_a, person_b, annotator, relationship_type, confidence, notes):
    """Save one pair-level relationship label."""
    path = RESULTS_PATH / "human_pair_labels.csv"
    # Ensure consistent ordering
    a, b = min(person_a, person_b), max(person_a, person_b)
    new_row = pd.DataFrame([{
        "person_a": a,
        "person_b": b,
        "annotator": annotator,
        "relationship_type": relationship_type,
        "confidence": confidence,
        "notes": notes,
    }])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[
            ~((existing["person_a"] == a) & (existing["person_b"] == b)
              & (existing["annotator"] == annotator))
        ]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(path, index=False)


def load_cluster_labels():
    """Load human cluster-level validation labels."""
    path = RESULTS_PATH / "human_cluster_labels.csv"
    if not path.exists():
        return pd.DataFrame(columns=[
            "cluster", "annotator", "name_correct", "suggested_name", "notes",
        ])
    return pd.read_csv(path)


def save_cluster_label(cluster_id, annotator, name_correct, suggested_name, notes):
    """Save one cluster-level validation label."""
    path = RESULTS_PATH / "human_cluster_labels.csv"
    new_row = pd.DataFrame([{
        "cluster": cluster_id,
        "annotator": annotator,
        "name_correct": name_correct,
        "suggested_name": suggested_name,
        "notes": notes,
    }])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[
            ~((existing["cluster"] == cluster_id) & (existing["annotator"] == annotator))
        ]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(path, index=False)


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
                              selected_person=None, max_nodes=100,
                              show_types=None, min_emails_filter=1,
                              external_df=None):
    """
    Build an interactive pyvis network graph.
    If a person is selected, highlight them and their connections.
    Returns HTML string.
    """
    net = Network(
        height="650px",
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
    filtered_pairs = pairs_df
    if pairs_df is not None:
        # Apply filters
        if min_emails_filter > 1:
            filtered_pairs = filtered_pairs[filtered_pairs["email_count"] >= min_emails_filter]
        if show_types is not None:
            filtered_pairs = filtered_pairs[filtered_pairs["cluster"].apply(
                lambda c: cluster_names_dict.get(int(c), {}).get("name", "") in show_types
            )]
        pair_people = set(filtered_pairs["person_a"].tolist() + filtered_pairs["person_b"].tolist())

    # Pick which nodes to show
    if selected_person and selected_person in person_df["person"].values:
        connected = set()
        if filtered_pairs is not None:
            connected = set(
                filtered_pairs[filtered_pairs["person_a"] == selected_person]["person_b"].tolist() +
                filtered_pairs[filtered_pairs["person_b"] == selected_person]["person_a"].tolist()
            )
        top_people = set(person_df.nlargest(30, "total_degree")["person"].tolist())
        show_people = {selected_person} | connected | top_people
    else:
        # Default view: show executives + people in pairs (not all 7K employees)
        if "is_executive" in person_df.columns:
            exec_people = set(person_df[person_df["is_executive"]]["person"].tolist())
            show_people = exec_people | pair_people
        else:
            top_people = set(person_df.nlargest(max_nodes, "total_degree")["person"].tolist())
            show_people = top_people | pair_people

    feat = person_df.set_index("person")

    # Compute email count range for edge width scaling
    max_email_count = 1
    if filtered_pairs is not None and len(filtered_pairs) > 0:
        max_email_count = max(filtered_pairs["email_count"].max(), 1)

    # Determine which people are hubs or connected to selected
    hub_people = set(person_df[person_df["person_class"] == "Hub"]["person"].tolist())
    connected_to_selected = set()
    if selected_person and filtered_pairs is not None:
        connected_to_selected = set(
            filtered_pairs[filtered_pairs["person_a"] == selected_person]["person_b"].tolist() +
            filtered_pairs[filtered_pairs["person_b"] == selected_person]["person_a"].tolist()
        )

    # Compute size scaling based on the people we're showing (not all 7K)
    show_degrees = [int(feat.loc[p, "total_degree"]) for p in show_people if p in feat.index]
    max_degree = max(show_degrees) if show_degrees else 1
    min_degree = min(show_degrees) if show_degrees else 0

    # Add nodes
    for person in show_people:
        if person not in feat.index:
            continue
        row = feat.loc[person]
        cls = row["person_class"]
        color = PERSON_COLORS.get(cls, "#aaa")
        degree = int(row["total_degree"])
        # Scale size relative to the visible people (not absolute)
        if max_degree > min_degree:
            norm = (degree - min_degree) / (max_degree - min_degree)
        else:
            norm = 0.5
        size = 10 + norm * 40  # range: 10 to 50
        name = _short_name(person)

        # Highlight selected person
        border_width = 1
        border_color = color
        if person == selected_person:
            border_width = 4
            border_color = "#ffffff"
            size = max(size, 35)

        # Show name labels for all nodes (clean, readable names)
        show_label = True

        title = (
            f"{name}\n"
            f"Email: {person}\n"
            f"Class: {cls}\n"
            f"Connections: {degree}\n"
            f"Betweenness: {row['betweenness']:.3f}\n"
            f"PageRank: {row['pagerank']:.4f}\n"
            f"Community: {int(row.get('community', 0))}"
        )

        net.add_node(
            person,
            label=name if show_label else "",
            title=title,
            color=color,
            size=size,
            borderWidth=border_width,
            font={"size": 11, "color": "white", "strokeWidth": 2, "strokeColor": "#000000"},
        )

    # Add edges from relationship pairs
    if filtered_pairs is not None:
        for _, row in filtered_pairs.iterrows():
            a, b = row["person_a"], row["person_b"]
            if a in show_people and b in show_people:
                if a in feat.index and b in feat.index:
                    cluster_id = int(row["cluster"])
                    rel_name = cluster_names_dict.get(cluster_id, {}).get("name", f"Cluster {cluster_id}")
                    edge_color = RELATIONSHIP_COLORS.get(rel_name, "#555555")
                    email_count = int(row["email_count"])

                    # Base width scales with email count (1.5-6)
                    width = max(1.5, min(6, email_count / max_email_count * 6))

                    # Relationship-specific styling
                    if rel_name == "Close":
                        width = max(width, 3)       # always visible
                    elif rel_name == "Romance":
                        width = max(width, 3.5)     # prominent
                    elif rel_name == "Transactional":
                        width = min(width, 2)       # thin
                    elif rel_name == "Tense / Conflict":
                        width = max(width, 2.5)     # noticeable

                    # Highlight edges connected to selected person
                    if selected_person and (a == selected_person or b == selected_person):
                        width = max(width, 5)

                    # Dashed edges for Fading relationships
                    dashes = rel_name == "Fading"

                    # Build rich tooltip
                    tooltip_parts = [
                        f"{_short_name(a)} ↔ {_short_name(b)}\n",
                        f"Type: {rel_name}\n",
                        f"Emails: {email_count}\n",
                        f"Self-disclosure: {row['avg_intimacy']:.3f}\n",
                        f"Responsiveness: {row['avg_warmth']:.3f}\n",
                        f"Sentiment: {row.get('avg_sentiment', 0):.3f}\n",
                    ]
                    if "temporal_stability" in row.index:
                        tooltip_parts.append(f"Stability: {row['temporal_stability']:.2f}\n")
                    if "burstiness" in row.index:
                        tooltip_parts.append(f"Burstiness: {row['burstiness']:+.2f}\n")
                    if "dispersion" in row.index:
                        tooltip_parts.append(f"Dispersion: {row['dispersion']:.2f}")

                    title = "".join(tooltip_parts)

                    net.add_edge(
                        a, b,
                        color={"color": edge_color, "opacity": 0.8},
                        width=width,
                        title=title,
                        dashes=dashes,
                        label="",
                        font={"size": 0},
                    )

    # --- Add external contacts if provided ---
    if external_df is not None and len(external_df) > 0:
        ext_color = PERSON_COLORS.get("External", "#f1c40f")

        # Only show external contacts connected to visible people
        visible_ext = external_df[
            external_df["executive"].isin(show_people)
        ].copy()

        # If a person is selected, only show their external contacts
        if selected_person:
            visible_ext = visible_ext[visible_ext["executive"] == selected_person]

        # Limit to top contacts to avoid clutter
        visible_ext = visible_ext.nlargest(min(20, len(visible_ext)), "email_count")

        EXT_TYPE_COLORS = {
            "Romantic": "#e91e8c",
            "Close Personal": "#9b59b6",
            "Friendly": "#2ecc71",
            "Business": "#f1c40f",
            "Distant": "#95a5a6",
        }

        for _, erow in visible_ext.iterrows():
            ext_addr = erow["external_contact"]
            exec_addr = erow["executive"]
            domain = erow.get("domain", "")
            is_personal = erow.get("is_personal", False)
            count = int(erow["email_count"])
            ext_name = _short_name(ext_addr)
            rel_type = erow.get("relationship_type", "Unknown")
            intimacy = erow.get("avg_intimacy", 0)
            warmth = erow.get("avg_warmth", 0)

            node_color = EXT_TYPE_COLORS.get(rel_type, "#f1c40f")

            # Add external node
            net.add_node(
                ext_addr,
                label=f"{ext_name}\n({rel_type})",
                title=(
                    f"<b>{ext_name}</b><br>"
                    f"Email: {ext_addr}<br>"
                    f"Type: <b>{rel_type}</b><br>"
                    f"Domain: {domain}<br>"
                    f"Emails: {count}<br>"
                    f"Self-disclosure: {intimacy:.2f}<br>"
                    f"Responsiveness: {warmth:.2f}"
                ),
                color={
                    "background": node_color,
                    "border": "#ffffff" if rel_type in ("Romantic", "Close Personal") else "#888",
                    "highlight": {"background": "#ffffff", "border": node_color},
                },
                size=max(8, min(25, count * 0.5)),
                borderWidth=3 if rel_type in ("Romantic", "Close Personal") else 2,
                shape="diamond",
                font={"size": 9, "color": node_color},
            )

            # Add edge — color by relationship type
            net.add_edge(
                exec_addr, ext_addr,
                color=node_color,
                width=max(1, min(4, count * 0.1)),
                dashes=True,
                title=(
                    f"{_short_name(exec_addr)} → {ext_name}<br>"
                    f"Type: {rel_type} | {count} emails"
                ),
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
Framework: Reis & Shaver (1988) interpersonal process model + Ureña-Carrion et al. (2020)
temporal features + Backstrom & Kleinberg (2014) dispersion

Feature glossary:
  avg_intimacy = self-disclosure level (0=formal, 1=deeply personal)
  avg_warmth = responsiveness level (0=hostile, 1=deeply caring)
  burstiness = communication pattern (-1=regular, +1=bursty)
  temporal_stability = fraction of months with contact (0-1)
  dispersion = how spread out mutual contacts are (0=clustered, 1=dispersed)

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
HIGHEST SELF-DISCLOSURE PAIRS:
{top_personal[['person_a','person_b','avg_intimacy','avg_warmth']].to_string(index=False)}
"""
        lowest_warmth = pairs_df.nsmallest(5, "avg_warmth")
        context += f"""
LOWEST RESPONSIVENESS PAIRS:
{lowest_warmth[['person_a','person_b','avg_warmth','avg_intimacy']].to_string(index=False)}
"""

    return context


# ----------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------
st.sidebar.title("📧 Enron Social World")
st.sidebar.markdown("*RAF620M — University of Iceland*")
st.sidebar.markdown("---")

NAV_OPTIONS = [
    "🏋️ Training",
    "🏠 Overview",
    "🕸️ Social Network",
    "👤 Person Profile",
    "🔗 Relationships",
    "📈 Model Analysis",
    "🏷️ Human Validation",
    "🚀 Run",
    "🤖 Ask the Data",
]

# Check if we need to force-navigate (e.g. from graph double-click)
default_page_idx = 0
if "nav_override" in st.session_state:
    override = st.session_state.pop("nav_override")
    if override in NAV_OPTIONS:
        default_page_idx = NAV_OPTIONS.index(override)

page = st.sidebar.radio(
    "Navigate",
    NAV_OPTIONS,
    index=default_page_idx,
)

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------
person_df = load_person_data()
pairs_df = load_relationship_data()
cluster_profiles = load_cluster_profiles()
clustering_meta = load_clustering_meta()
cluster_names = load_cluster_names()

data_ready = person_df is not None

# Check URL for selected person (from graph double-click)
query_params = st.query_params
if "selected" in query_params:
    st.session_state["selected_person"] = query_params["selected"]
    # Clear the URL param so it doesn't keep triggering
    del st.query_params["selected"]
    # Navigate to Person Profile
    st.session_state["nav_override"] = "👤 Person Profile"
    st.rerun()

if not data_ready and page not in ["🏋️ Training", "🚀 Run"]:
    st.warning("⚠️ No results found. Please run `python main.py` first to generate the analysis.")


# ================================================================
# PAGE 0 — Training
# ================================================================
if page == "🏋️ Training":
    st.title("🏋️ Training")
    st.markdown("Label emails with Claude, train models, and build your dataset.")

    # --- Helper: check which output files exist ---
    def _file_info(path):
        """Return (exists, size_str, mod_time_str) for a file."""
        p = Path(path)
        if not p.exists():
            return False, "—", "—"
        size = p.stat().st_size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / (1024*1024):.1f} MB"
        import datetime
        mod = datetime.datetime.fromtimestamp(p.stat().st_mtime)
        return True, size_str, mod.strftime("%Y-%m-%d %H:%M")

    def _count_csv_rows(path):
        """Quick row count without loading full CSV."""
        p = Path(path)
        if not p.exists():
            return 0
        try:
            return max(sum(1 for _ in open(p)) - 1, 0)
        except Exception:
            return 0

    # --- Load pipeline progress if it exists ---
    progress_path = RESULTS_PATH / "pipeline_progress.json"
    progress = {}
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text())
        except Exception:
            progress = {}

    stages_progress = progress.get("stages", {})

    # --- VISUAL PIPELINE STEPPER ---
    st.markdown("### Pipeline Status")

    STAGE_DEFS = [
        ("load",    "Load",      "data/processed/emails.parquet",           "emails"),
        ("network", "Network",   "data/results/person_classes.csv",        "people"),
        ("stage1",  "Stage 1",   "data/results/claude_labeled_emails.csv", "labels"),
        ("stage2",  "Stage 2",   "data/results/relationship_pairs.csv",    "pairs"),
        ("stage3",  "Stage 3",   "data/results/clustering_meta.json",      "clusters"),
        ("stage4",  "Stage 4",   "data/results/cluster_names.json",        "types"),
    ]

    stage_cols = st.columns(len(STAGE_DEFS))

    def _count_rows(path):
        """Count rows in CSV or parquet file."""
        p = Path(path)
        if not p.exists():
            return 0
        try:
            if p.suffix == ".parquet":
                import pyarrow.parquet as pq
                return pq.read_metadata(p).num_rows
            return max(sum(1 for _ in open(p)) - 1, 0)
        except Exception:
            return 0

    for idx, (key, label, check_file, unit) in enumerate(STAGE_DEFS):
        with stage_cols[idx]:
            exists, size_str, mod_time = _file_info(check_file)
            stage_info = stages_progress.get(key, {})
            status = stage_info.get("status", "")

            # Determine visual state
            if status == "done" or exists:
                icon = "✅"
                bg = "#2ecc71"
            elif status == "running":
                icon = "⏳"
                bg = "#f39c12"
            else:
                icon = "⬜"
                bg = "#555555"

            # Get stat count
            stat_text = ""
            if exists:
                stats = stage_info.get("stats", {})
                # How many emails were processed in last run
                emails_processed = stages_progress.get("load", {}).get(
                    "stats", {}).get("total_emails", 0)

                if key == "load":
                    total = _count_rows(check_file)
                    # Read batch offset if available
                    batch_path = RESULTS_PATH / "batch_offset.json"
                    if batch_path.exists():
                        try:
                            batch_info = json.loads(batch_path.read_text())
                            offset = batch_info.get("offset", 0)
                            pct = offset / total * 100 if total > 0 else 0
                            stat_text = (f"Total: {total:,}<br>"
                                         f"Processed: {offset:,} ({pct:.0f}%)")
                        except Exception:
                            stat_text = f"Total: {total:,}<br>Processed: {emails_processed:,}" if emails_processed else f"Total: {total:,}"
                    else:
                        stat_text = f"Total: {total:,}<br>Processed: {emails_processed:,}" if emails_processed else f"Total: {total:,}"
                elif key == "network":
                    people = stats.get("people", _count_csv_rows(check_file))
                    hubs = stats.get("hubs", 0)
                    stat_text = f"{emails_processed:,} emails<br>{people:,} people, {hubs} hubs" if emails_processed else f"{people:,} {unit}"
                elif key == "stage1":
                    labels = stats.get("labels", _count_csv_rows(check_file))
                    labels_str = f"{labels:,}" if isinstance(labels, (int, float)) else str(labels)
                    # Load F1 / R² from saved model comparisons
                    _int_path = RESULTS_PATH / "model_comparison_intimacy.json"
                    _warm_path = RESULTS_PATH / "model_comparison_warmth.json"
                    _f1_str = ""
                    if _int_path.exists():
                        _int_data = json.loads(_int_path.read_text())
                        _best_int = max(
                            (v for k, v in _int_data.items() if not k.startswith("_")),
                            key=lambda v: v.get("f1", 0),
                        )
                        _f1_str += f"<br>Disclosure F1: {_best_int['f1']:.2f}"
                    if _warm_path.exists():
                        _warm_data = json.loads(_warm_path.read_text())
                        _best_warm = max(
                            (v for k, v in _warm_data.items() if not k.startswith("_")),
                            key=lambda v: v.get("r2", 0),
                        )
                        _f1_str += f"<br>Responsive R²: {_best_warm['r2']:.2f}"
                    stat_text = (f"{labels_str} labeled{_f1_str}" if _f1_str
                                 else f"{labels_str} {unit}")
                elif key == "stage2":
                    pairs = stats.get("pairs", _count_csv_rows(check_file))
                    stat_text = f"{emails_processed:,} emails<br>{pairs:,} pairs" if emails_processed else f"{pairs:,} {unit}"
                elif key == "stage3":
                    n = stats.get("clusters", "")
                    stat_text = f"{emails_processed:,} emails<br>{n} clusters" if emails_processed and n else (f"{n} {unit}" if n else "done")
                elif key == "stage4":
                    try:
                        names_path = Path(check_file)
                        if names_path.exists():
                            names_data = json.loads(names_path.read_text())
                            type_names = [v["name"] for v in names_data.values()]
                            stat_text = f"{len(type_names)} types<br>{'  |  '.join(type_names)}"
                        else:
                            stat_text = "done"
                    except Exception:
                        stat_text = "done"

            st.markdown(
                f"<div style='background:{bg};padding:14px 8px;border-radius:10px;"
                f"text-align:center;color:white;min-height:110px;"
                f"display:flex;flex-direction:column;justify-content:center'>"
                f"<div style='font-size:1.8em'>{icon}</div>"
                f"<div style='font-size:0.95em;font-weight:bold;margin:4px 0'>{label}</div>"
                f"<div style='font-size:0.75em;opacity:0.9'>{stat_text}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Arrow connectors
    st.markdown(
        "<div style='text-align:center;color:#888;font-size:0.8em;margin:-6px 0 12px 0'>"
        "Load → Network → Score → Features → Cluster → Name"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # --- TRAINING DATA PROGRESS ---
    LABEL_TARGET = 5000

    # Count accumulated labels
    labels_path = RESULTS_PATH / "claude_labeled_emails.csv"
    accumulated_labels = 0
    if labels_path.exists():
        accumulated_labels = max(_count_csv_rows(str(labels_path)), 0)

    # Count total emails available
    total_emails_available = 0
    load_file = Path("data/processed/emails.parquet")
    if load_file.exists():
        total_emails_available = _count_rows(str(load_file))

    models_exist = (RESULTS_PATH / "model_intimacy.pkl").exists()

    label_pct = min(accumulated_labels / LABEL_TARGET * 100, 100)
    ready_for_full = accumulated_labels >= LABEL_TARGET

    st.markdown("### Training Data Progress")

    # Progress bar toward target
    st.progress(
        min(label_pct / 100, 1.0),
        text=f"Labels: {accumulated_labels:,} / {LABEL_TARGET:,} ({label_pct:.0f}%)"
             f" — {'Ready for Full Run! Go to 🚀 Run page.' if ready_for_full else f'{LABEL_TARGET - accumulated_labels:,} more needed'}"
    )

    # Check if human labels were added since last training
    human_path = RESULTS_PATH / "human_labels.csv"
    model_path = RESULTS_PATH / "model_intimacy.pkl"
    human_labels_newer = False
    if human_path.exists() and model_path.exists():
        human_labels_newer = human_path.stat().st_mtime > model_path.stat().st_mtime

    st.markdown("---")

    # --- TRAINING MODE ---
    st.markdown("### Training Mode")

    if human_labels_newer:
        st.warning(
            "You have **new human labels** since the last training. "
            "Click **Retrain** below to incorporate them — no Claude calls needed."
        )

    strategy = st.radio(
        "Choose training mode",
        ["sample", "retrain", "custom"] if accumulated_labels > 0 else ["sample", "custom"],
        format_func=lambda s: {
            "sample":  "⚡ New batch — label next 2,000 emails + train (~5 min)",
            "retrain": "🔄 Retrain — rebuild models on existing labels, no Claude calls (~1 min)",
            "custom":  "🔧 Custom batch size and parameters",
        }[s],
        index=1 if human_labels_newer and accumulated_labels > 0 else 0,
        horizontal=True,
    )

    if strategy == "sample":
        batch_offset_path = RESULTS_PATH / "batch_offset.json"
        next_start = 0
        if batch_offset_path.exists():
            try:
                next_start = json.loads(batch_offset_path.read_text()).get("offset", 0)
            except Exception:
                pass
        next_end = min(next_start + 2000, total_emails_available)

        st.info(
            f"Next batch: emails {next_start + 1:,} – {next_end:,} of {total_emails_available:,}.\n\n"
            f"After this run: **{accumulated_labels + 2000:,}** labels, "
            f"trained models saved to disk."
        )
        email_sample = 2000
        label_sample = 2000
        min_pair = 5
        k_min = 3
        k_max = 7
        dbscan_min = 5

    elif strategy == "retrain":
        st.success(
            f"**No Claude calls.** Retrains classifiers on your existing "
            f"{accumulated_labels:,} labels (including human corrections), "
            f"scores all {total_emails_available:,} emails, rebuilds pairs, "
            f"clusters, and names — the full pipeline with updated models."
        )
        email_sample = 0  # all emails
        label_sample = 0  # cached labels only
        min_pair = 5
        k_min = 3
        k_max = 8
        dbscan_min = 5

    elif strategy == "custom":
        st.info("Configure training parameters manually.")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            email_sample = st.number_input(
                "Email batch size",
                min_value=500, max_value=50000, value=2000, step=500,
            )
            label_sample = st.number_input(
                "Claude label budget",
                min_value=0, max_value=10000, value=2000, step=500,
                help="0 = use existing labels only",
            )
        with p_col2:
            min_pair = st.number_input(
                "Min emails per pair",
                min_value=2, max_value=20, value=5,
            )
            k_min = st.number_input("K min (clusters)", min_value=2, max_value=10, value=3)
        with p_col3:
            k_max = st.number_input("K max (clusters)", min_value=3, max_value=15, value=7)
            dbscan_min = st.number_input("DBSCAN min samples", min_value=2, max_value=20, value=5)

    # --- PARAMETERS SUMMARY ---
    st.markdown("### Parameters")
    p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)
    if strategy == "retrain":
        p_col1.metric("Emails", "—")
        p_col2.metric("Labels", f"{accumulated_labels:,}")
        p_col3.metric("Mode", "Retrain only")
    else:
        display_emails = f"{email_sample:,}" if email_sample > 0 else f"{total_emails_available:,}"
        display_labels = f"{label_sample:,}" if label_sample > 0 else f"{accumulated_labels:,} (cached)"
        p_col1.metric("Emails", display_emails)
        p_col2.metric("Labels", display_labels)
        p_col3.metric("Mode", "Train")
    p_col4.metric("Min pair", f"{min_pair}")
    p_col5.metric("K range", f"{k_min}–{k_max}")
    p_col6.metric("DBSCAN min", f"{dbscan_min}")

    # --- COST & TIME ESTIMATE ---
    actual_labels = label_sample if label_sample > 0 else 0
    api_cost_low = actual_labels * 0.003
    api_cost_high = actual_labels * 0.006

    if actual_labels == 0:
        time_est = "2–3 min"  # retrain only, no Claude
    elif actual_labels <= 2000:
        time_est = "3–6 min"
    elif actual_labels <= 5000:
        time_est = "8–15 min"
    else:
        time_est = "15–40 min"

    est_col1, est_col2, est_col3 = st.columns(3)
    est_col1.metric("New labels this run", f"{actual_labels:,}")
    est_col2.metric("Claude API cost", f"${api_cost_low:.2f}–${api_cost_high:.2f}")
    est_col3.metric("Est. time", time_est)

    st.markdown("---")

    # --- RUN PIPELINE ---
    st.markdown("### Run Pipeline")

    # Build config dict
    run_config = {
        "batch_strategy": strategy,
        "email_sample_size": email_sample,
        "label_sample_size": label_sample,
        "min_emails_per_pair": min_pair,
        "k_min": k_min,
        "k_max": k_max,
        "dbscan_eps_values": [0.5, 0.75, 1.0, 1.5, 2.0],
        "dbscan_min_samples": dbscan_min,
    }

    # Check if raw data exists
    raw_exists = Path("data/raw/maildir").exists()
    processed_exists = Path("data/processed/emails.parquet").exists()

    if not raw_exists and not processed_exists:
        st.error(
            "**No email data found.** Place the Enron maildir dataset at "
            "`data/raw/maildir/` or a processed file at `data/processed/emails.parquet`."
        )
        can_run = False
    else:
        can_run = True

    col_run, col_status = st.columns([1, 2])

    with col_run:
        run_clicked = st.button(
            "▶ Run Pipeline",
            type="primary",
            disabled=not can_run,
            use_container_width=True,
        )

    with col_status:
        if "pipeline_running" in st.session_state and st.session_state["pipeline_running"]:
            st.warning("Pipeline is running... check the terminal for live output.")

    if run_clicked and can_run:
        # Save config
        config_path = RESULTS_PATH / "pipeline_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2)

        # Clear old progress
        prog_path = RESULTS_PATH / "pipeline_progress.json"
        if prog_path.exists():
            prog_path.unlink()

        st.session_state["pipeline_running"] = True

        st.markdown("#### Pipeline Output")
        output_container = st.empty()

        import subprocess, os

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            [sys.executable, "-u", "main.py", "--config", str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(Path(__file__).parent),
            env=env,
        )

        output_lines = []

        # Stage progress tracking
        STAGE_LABELS = {
            "[1/6]": ("Loading emails...", 0),
            "[2/6]": ("Network analysis...", 1),
            "[3/6]": ("Stage 1 — scoring emails...", 2),
            "[4/6]": ("Stage 2 — pair features...", 3),
            "[5/6]": ("Stage 3 — clustering...", 4),
            "[6/6]": ("Stage 4 — naming clusters...", 5),
        }

        with st.status("Starting pipeline...", expanded=True) as status:
            progress_bar = st.progress(0, text="Initialising...")
            stage_indicator = st.empty()
            log_placeholder = st.empty()

            current_stage = -1

            for line in iter(process.stdout.readline, ""):
                stripped = line.rstrip()
                if not stripped:
                    continue
                output_lines.append(stripped)

                # Detect stage transitions
                for marker, (label, stage_num) in STAGE_LABELS.items():
                    if marker in stripped:
                        current_stage = stage_num
                        pct = int((stage_num / 6) * 100)
                        progress_bar.progress(pct, text=f"Step {stage_num + 1}/6 — {label}")
                        status.update(label=label)
                        break

                # Detect sub-progress (Claude labeling, model training, etc.)
                if "Labeled" in stripped and "/" in stripped:
                    stage_indicator.caption(f"  ↳ {stripped.strip()}")
                elif "Training" in stripped or "Scoring" in stripped or "Computing" in stripped:
                    stage_indicator.caption(f"  ↳ {stripped.strip()}")
                elif "Best" in stripped or "Selected" in stripped:
                    stage_indicator.caption(f"  ↳ {stripped.strip()}")

                # Show last 20 lines
                display_text = "\n".join(output_lines[-20:])
                log_placeholder.code(display_text, language=None)

            process.wait()

            st.session_state["pipeline_running"] = False

            if process.returncode == 0:
                progress_bar.progress(100, text="All 6 stages complete!")
                status.update(label="Pipeline complete!", state="complete", expanded=False)
                st.success("Pipeline completed successfully! Refresh the page to see results.")
                st.balloons()
            else:
                progress_bar.progress(100, text="Failed")
                status.update(label="Pipeline failed", state="error")
                st.error(f"Pipeline failed with exit code {process.returncode}. Check the output above.")

        # Show full log in expander
        with st.expander("Full pipeline log", expanded=True):
            st.code("\n".join(output_lines), language=None)

    # --- OUTPUT FILES TABLE ---
    st.markdown("---")
    st.markdown("### Output Files")

    output_files = [
        ("data/processed/emails.parquet",             "Parsed emails",            "Load"),
        ("data/results/person_classes.csv",           "Person classifications",   "Network"),
        ("data/results/communities.json",             "Community assignments",    "Network"),
        ("data/results/claude_labeled_emails.csv",    "Claude-labeled samples",   "Stage 1"),
        ("data/results/email_scores.csv",             "Email scores",             "Stage 1"),
        ("data/results/model_comparison.json",        "Model comparison",         "Stage 1"),
        ("data/results/model_comparison_intimacy.json","Disclosure models",       "Stage 1"),
        ("data/results/model_comparison_warmth.json", "Responsiveness models",    "Stage 1"),
        ("data/results/relationship_pairs.csv",       "Pair feature vectors",     "Stage 2"),
        ("data/results/clustering_meta.json",         "Clustering metadata",      "Stage 3"),
        ("data/results/cluster_profiles.csv",         "Cluster profiles",         "Stage 3"),
        ("data/results/cluster_names.json",           "Cluster names",            "Stage 4"),
        ("data/results/cluster_zscores.csv",          "Cluster z-scores",         "Stage 4"),
        ("data/results/pipeline_config.json",         "Last run config",          "Config"),
    ]

    file_rows = []
    for fpath, desc, stage in output_files:
        exists, size_str, mod_time = _file_info(fpath)
        file_rows.append({
            "Stage": stage,
            "File": Path(fpath).name,
            "Description": desc,
            "Status": "✅" if exists else "—",
            "Size": size_str,
            "Modified": mod_time,
        })

    st.dataframe(
        pd.DataFrame(file_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn(width="small"),
            "Size": st.column_config.TextColumn(width="small"),
        },
    )

    # --- DANGER ZONE ---
    with st.expander("🗑️ Reset Pipeline (danger zone)"):

        ALWAYS_KEEP = {
            "human_labels.csv",
            "human_pair_labels.csv",
            "human_cluster_labels.csv",
            "claude_labeled_emails.csv",
            "model_intimacy.pkl",
            "model_warmth.pkl",
            "batch_offset.json",
        }

        st.markdown("**Reset pipeline outputs**")
        st.caption(
            "Deletes stage outputs (pairs, clusters, charts). "
            "Keeps: human labels, Claude labels, trained models, batch progress."
        )

        also_delete_labels = st.checkbox(
            "Also delete Claude labels (will re-call API on next run)",
            key="reset_labels",
        )
        also_delete_emails = st.checkbox(
            "Also delete parsed emails (will re-parse raw maildir on next run — slow)",
            key="reset_emails",
        )

        confirm = st.checkbox("I understand — reset now", key="reset_confirm")
        if st.button("Reset", type="secondary", disabled=not confirm):
            results_dir = Path("data/results")
            keep = set(ALWAYS_KEEP)
            if also_delete_labels:
                keep.discard("claude_labeled_emails.csv")
                keep.discard("batch_offset.json")

            if results_dir.exists():
                deleted = 0
                for f in results_dir.iterdir():
                    if f.is_file() and f.name not in keep:
                        f.unlink()
                        deleted += 1

            if also_delete_emails:
                p = Path("data/processed/emails.parquet")
                if p.exists():
                    p.unlink()
                    deleted += 1

            st.cache_data.clear()
            st.success(f"Reset complete — deleted {deleted} files.")
            st.rerun()


# ================================================================
# PAGE 1 — Overview
# ================================================================
elif page == "🏠 Overview":
    st.title("The Social World of Enron")
    st.markdown("### A Behavioral Analytics Study of Corporate Fraud")
    st.markdown("""
    In 2001, Enron Corporation collapsed in one of the largest corporate fraud
    scandals in history. The investigation released **~500,000 internal emails**
    — giving us a rare window into the private world of a company falling apart.

    This app explores those emails through a **four-stage ML pipeline** grounded
    in published research: **Reis & Shaver (1988)** for interpersonal closeness,
    **Ureña-Carrion et al. (2020)** for temporal communication patterns,
    **Backstrom & Kleinberg (2014)** for network dispersion, and
    **Ireland et al. (2011)** for linguistic style matching.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("**📊 Stage 1: Email Scoring**\nSelf-disclosure + responsiveness classifiers (Reis & Shaver 1988)")
    with col2:
        st.warning("**🔧 Stage 2: Pair Features**\n24-feature vector per pair (temporal + structural + linguistic)")
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

        # Show model performance
        _int_path = RESULTS_PATH / "model_comparison_intimacy.json"
        _warm_path = RESULTS_PATH / "model_comparison_warmth.json"

        if _int_path.exists() or _warm_path.exists():
            st.markdown("---")
            st.markdown("### Model Performance")
            st.markdown(
                "These models were trained on Claude-labeled emails (with human corrections) "
                "and then used to score **all** emails in the dataset."
            )

            mcol1, mcol2 = st.columns(2)

            # --- Self-disclosure classifier ---
            if _int_path.exists():
                int_data = json.loads(_int_path.read_text())
                model_names = [k for k in int_data if not k.startswith("_")]
                best_int_name = max(model_names, key=lambda k: int_data[k].get("f1", 0))
                best_int = int_data[best_int_name]
                sig_info = int_data.get("_significance", {})

                with mcol1:
                    st.markdown("#### Self-Disclosure (Classifier)")
                    st.markdown(f"**Best model: {best_int_name}**")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("F1", f"{best_int['f1']:.2f}")
                    m2.metric("Accuracy", f"{best_int['accuracy']:.2f}")
                    m3.metric("Precision", f"{best_int['precision']:.2f}")
                    m4.metric("Recall", f"{best_int['recall']:.2f}")

                    # All models comparison
                    rows = []
                    for name in model_names:
                        d = int_data[name]
                        rows.append({
                            "Model": name,
                            "F1": f"{d['f1']:.3f}",
                            "Accuracy": f"{d['accuracy']:.3f}",
                            "Precision": f"{d['precision']:.3f}",
                            "Recall": f"{d['recall']:.3f}",
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True)

                    if sig_info:
                        p = sig_info.get("p_value", 1)
                        sig_text = f"p={p:.3f} ({'significant' if p < 0.05 else 'not significant'})"
                        st.caption(
                            f"{sig_info.get('best', '')} vs {sig_info.get('second', '')}: {sig_text}"
                        )

            # --- Responsiveness regressor ---
            if _warm_path.exists():
                warm_data = json.loads(_warm_path.read_text())
                model_names_w = [k for k in warm_data if not k.startswith("_")]
                best_warm_name = max(model_names_w, key=lambda k: warm_data[k].get("r2", 0))
                best_warm = warm_data[best_warm_name]
                sig_info_w = warm_data.get("_significance", {})

                with mcol2:
                    st.markdown("#### Responsiveness (Regressor)")
                    st.markdown(f"**Best model: {best_warm_name}**")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("R²", f"{best_warm['r2']:.2f}")
                    m2.metric("MAE", f"{best_warm['mae']:.2f}")
                    m3.metric("RMSE", f"{best_warm['rmse']:.2f}")

                    rows_w = []
                    for name in model_names_w:
                        d = warm_data[name]
                        rows_w.append({
                            "Model": name,
                            "R²": f"{d['r2']:.3f}",
                            "MAE": f"{d['mae']:.3f}",
                            "RMSE": f"{d['rmse']:.3f}",
                        })
                    st.dataframe(pd.DataFrame(rows_w), hide_index=True)

                    if sig_info_w:
                        p = sig_info_w.get("p_value", 1)
                        sig_text = f"p={p:.3f} ({'significant' if p < 0.05 else 'not significant'})"
                        st.caption(
                            f"{sig_info_w.get('best', '')} vs {sig_info_w.get('second', '')}: {sig_text}"
                        )

        # Show charts
        comp_img = RESULTS_PATH / "model_comparison.png"
        if comp_img.exists():
            st.markdown("---")
            st.markdown("### Model Comparison Charts")
            st.image(str(comp_img), width='stretch')

            # Confusion matrices for all 3 classifier models
            st.markdown("#### Self-Disclosure — Confusion Matrices")
            cm_cols = st.columns(3)
            cm_models = [
                ("Logistic Regression", "logistic_regression"),
                ("SVM", "svm"),
                ("Random Forest", "random_forest"),
            ]
            for i, (label, safe_name) in enumerate(cm_models):
                cm_path = RESULTS_PATH / f"confusion_matrix_intimacy_{safe_name}.png"
                # Fallback to old single-model file
                if cm_path.exists():
                    cm_cols[i].markdown(f"**{label}**")
                    cm_cols[i].image(str(cm_path), width='stretch')

            # Responsiveness scatter plot
            scatter_warmth = RESULTS_PATH / "scatter_warmth.png"
            if scatter_warmth.exists():
                st.markdown("#### Responsiveness — Predicted vs Actual")
                st.image(str(scatter_warmth), width='stretch')


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

        # Node legend (person classes)
        st.markdown("**Nodes** = people (coloured by role in network)")
        cols = st.columns(7)
        for i, (cls, color) in enumerate(PERSON_COLORS.items()):
            desc = PERSON_CLASS_DESCRIPTIONS.get(cls, "")
            cols[i].markdown(
                f"<div style='background:{color};padding:8px 6px;border-radius:6px;"
                f"color:white;text-align:center;font-size:0.85em'>"
                f"<div style='font-weight:bold'>● {cls}</div>"
                f"<div style='font-size:0.75em;opacity:0.85;margin-top:2px'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # --- FILTER CONTROLS ---
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2, 2, 1, 1])

        with filter_col1:
            people_in_pairs = sorted(set(
                (pairs_df["person_a"].tolist() + pairs_df["person_b"].tolist())
                if pairs_df is not None else []
            ))
            focus_options = ["Everyone (top connected)"] + [
                f"{_short_name(p)} ({p})" for p in people_in_pairs
            ]
            focus = st.selectbox("Focus on a person", focus_options)

        with filter_col2:
            # Filter by relationship type
            available_types = []
            if cluster_names:
                available_types = [info["name"] for info in cluster_names.values()]
            show_types = st.multiselect(
                "Show relationship types",
                available_types,
                default=available_types,
            )

        with filter_col3:
            min_emails_filter = st.number_input(
                "Min emails",
                min_value=1, max_value=100, value=1,
                help="Only show edges with at least this many emails",
            )

        # Load external contacts (not cached — file may be new)
        _ext_path = RESULTS_PATH / "external_contacts.csv"
        ext_contacts = None
        if _ext_path.exists():
            ext_contacts = pd.read_csv(_ext_path)

        show_external = False
        with filter_col4:
            if ext_contacts is not None and len(ext_contacts) > 0:
                n_romantic = 0
                if "romantic_flag" in ext_contacts.columns:
                    n_romantic = int(ext_contacts["romantic_flag"].sum())
                show_external = st.checkbox(
                    f"External ({len(ext_contacts):,})",
                    value=False,
                    help=f"{len(ext_contacts):,} external contacts, {n_romantic} romantic-flagged. Shows as yellow diamonds.",
                )
            else:
                st.caption("No external contacts found")

        selected = None
        if focus != "Everyone (top connected)":
            selected = focus.split("(")[1].rstrip(")")

        # Build and display the interactive graph
        html = build_interactive_network(
            person_df, pairs_df, cluster_names, selected_person=selected,
            show_types=show_types if show_types != available_types else None,
            min_emails_filter=min_emails_filter,
            external_df=ext_contacts if show_external else None,
        )
        components.html(html, height=670, scrolling=False)

        # Relationship type legend (edge colors + styles)
        if cluster_names:
            st.markdown("---")
            st.markdown("**Edges** = relationships (coloured by type from clustering)")
            cols = st.columns(min(len(cluster_names), 4))
            for i, (cid, info) in enumerate(sorted(cluster_names.items())):
                name = info["name"]
                color = RELATIONSHIP_COLORS.get(name, CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])
                style_hint = ""
                if name == "Fading":
                    style_hint = " (dashed)"
                elif name == "Transactional":
                    style_hint = " (thin)"
                elif name in ("Close", "Romance"):
                    style_hint = " (thick)"

                # Draw a sample line + label
                border_style = "dashed" if name == "Fading" else "solid"
                line_height = "4px" if name in ("Close", "Romance") else "3px"
                if name == "Transactional":
                    line_height = "2px"

                cols[i % len(cols)].markdown(
                    f"<div style='padding:8px;border-radius:6px;text-align:center;"
                    f"margin:2px 0;border:1px solid #333'>"
                    f"<div style='background:{color};height:{line_height};"
                    f"border-style:{border_style};margin:4px auto;width:80%;"
                    f"border-radius:2px'></div>"
                    f"<div style='font-size:0.85em;font-weight:bold;color:{color}'>"
                    f"{name}{style_hint}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
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
                width='stretch'
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
                    "avg_warmth", "email_count"
                ]]
                display.columns = ["Person", "Relationship", "Disclosure",
                                   "Responsiveness", "Emails"]
                st.dataframe(display.reset_index(drop=True), width='stretch')

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
            st.image(str(sil_img), width='stretch')

        # Cluster profiles (heatmap)
        st.markdown("---")
        st.markdown("### Cluster Feature Profiles")
        st.markdown("*What makes each cluster distinct — average feature values per cluster*")
        heatmap_img = RESULTS_PATH / "cluster_heatmap.png"
        if heatmap_img.exists():
            st.image(str(heatmap_img), width='stretch')

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

        # --- Flagged outlier relationships ---
        if "flags" in pairs_df.columns:
            pairs_df["flags"] = pairs_df["flags"].fillna("")
            flagged = pairs_df[pairs_df["flags"] != ""].copy()
            if len(flagged) > 0:
                st.markdown("---")
                st.markdown("### Flagged Relationships")
                st.markdown(
                    "Rare relationship types detected by score thresholds — "
                    "these are outliers that clustering alone would miss."
                )

                # Count per flag type
                all_flags = []
                for f in flagged["flags"]:
                    all_flags.extend([x.strip() for x in f.split(",") if x.strip()])
                flag_counts = pd.Series(all_flags).value_counts()

                FLAG_COLORS = {
                    "Romantic": "#e91e8c",
                    "Hostile": "#e74c3c",
                    "After-hours": "#9b59b6",
                    "Hierarchical": "#f39c12",
                    "High-intensity": "#e67e22",
                }

                flag_cols = st.columns(min(len(flag_counts), 5))
                for i, (flag_name, count) in enumerate(flag_counts.items()):
                    color = FLAG_COLORS.get(flag_name, "#555")
                    with flag_cols[i % len(flag_cols)]:
                        st.markdown(
                            f"<div style='background:{color};padding:10px;border-radius:8px;"
                            f"color:white;text-align:center'>"
                            f"<div style='font-size:1.4em;font-weight:bold'>{count}</div>"
                            f"<div style='font-size:0.85em'>{flag_name}</div></div>",
                            unsafe_allow_html=True,
                        )

                # Filter by flag
                selected_flag = st.selectbox(
                    "Filter by flag",
                    ["All"] + list(flag_counts.index),
                    key="flag_filter",
                )

                if selected_flag == "All":
                    show_flagged = flagged
                else:
                    show_flagged = flagged[flagged["flags"].str.contains(selected_flag)]

                show_flagged = show_flagged.copy()
                show_flagged["Person A"] = show_flagged["person_a"].apply(_short_name)
                show_flagged["Person B"] = show_flagged["person_b"].apply(_short_name)

                flag_display = ["Person A", "Person B", "flags", "avg_intimacy",
                                "avg_warmth", "avg_sentiment", "email_count"]
                available = [c for c in flag_display if c in show_flagged.columns]
                st.dataframe(
                    show_flagged[available]
                    .sort_values("flags")
                    .reset_index(drop=True),
                    width="stretch",
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
                        "avg_warmth", "avg_sentiment", "email_count", "time_span_days"]
        col_rename = {"avg_intimacy": "Disclosure", "avg_warmth": "Responsiveness",
                      "avg_sentiment": "Sentiment", "email_count": "Emails",
                      "time_span_days": "Span (days)"}
        st.dataframe(
            cluster_pairs[display_cols]
            .rename(columns=col_rename)
            .sort_values("Disclosure", ascending=False)
            .head(30)
            .reset_index(drop=True),
            width='stretch'
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
                f"Disclosure: {row['avg_intimacy']:.3f} | "
                f"Responsiveness: {row.get('avg_warmth', 0):.3f} | "
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
# PAGE 5 — Model Analysis
# ================================================================
elif page == "📈 Model Analysis":
    st.title("📈 Model Analysis")
    st.markdown("Deep dive into the pipeline's features, clusters, and data quality.")

    if data_ready and pairs_df is not None:

        # Get feature columns that actually exist in the data
        available_features = [c for c in FEATURE_COLS if c in pairs_df.columns]

        # ---- TAB LAYOUT ----
        tab_pca, tab_corr, tab_imp, tab_sil, tab_base, tab_timeline, tab_dist = st.tabs([
            "Cluster Map (PCA)",
            "Feature Correlations",
            "Feature Importance",
            "Silhouette Analysis",
            "Baseline Comparison",
            "Email Timeline",
            "Score Distributions",
        ])

        # ==============================================================
        # TAB 1 — PCA scatter plot
        # ==============================================================
        with tab_pca:
            st.markdown("### 2D Cluster Visualisation (PCA)")
            st.markdown(
                "Each dot is a pair of people. Colours show cluster membership. "
                "If clusters are visually separated, the features discriminate well."
            )

            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            X = pairs_df[available_features].fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=2)
            coords = pca.fit_transform(X_scaled)

            pca_df = pd.DataFrame({
                "PC1": coords[:, 0],
                "PC2": coords[:, 1],
                "Cluster": pairs_df["cluster"].astype(str),
            })

            fig, ax = plt.subplots(figsize=(10, 7))
            for cluster_id in sorted(pca_df["Cluster"].unique()):
                mask = pca_df["Cluster"] == cluster_id
                label = cluster_names.get(int(cluster_id), {}).get("name", f"Cluster {cluster_id}")
                color = CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]
                ax.scatter(
                    pca_df.loc[mask, "PC1"], pca_df.loc[mask, "PC2"],
                    c=color, label=label, alpha=0.6, edgecolors="white", linewidths=0.3, s=40,
                )
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
            ax.legend(loc="best", fontsize=9)
            ax.set_title("Pairs projected onto first two principal components")
            st.pyplot(fig)
            plt.close()

            # Show PCA loadings
            with st.expander("PCA loadings — which features drive each axis"):
                loadings = pd.DataFrame(
                    pca.components_.T,
                    index=available_features,
                    columns=["PC1", "PC2"],
                ).round(3)
                loadings["abs_PC1"] = loadings["PC1"].abs()
                loadings = loadings.sort_values("abs_PC1", ascending=False).drop(columns="abs_PC1")
                st.dataframe(loadings, width="stretch")

        # ==============================================================
        # TAB 2 — Feature correlation heatmap
        # ==============================================================
        with tab_corr:
            st.markdown("### Feature Correlation Matrix")
            st.markdown(
                "Low correlations between features = each adds unique information. "
                "High correlation (>0.8) = redundant features."
            )

            corr_matrix = pairs_df[available_features].corr()

            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            sns.heatmap(
                corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 7},
            )
            ax.set_title("Pairwise feature correlations (lower triangle)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Flag high correlations
            high_corr = []
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    r = corr_matrix.iloc[i, j]
                    if abs(r) > 0.7:
                        high_corr.append({
                            "Feature A": corr_matrix.index[i],
                            "Feature B": corr_matrix.columns[j],
                            "Correlation": f"{r:.3f}",
                        })
            if high_corr:
                st.markdown("**Notable correlations (|r| > 0.7):**")
                st.dataframe(pd.DataFrame(high_corr), width="stretch")
            else:
                st.success("No high correlations found — features are well-separated.")

        # ==============================================================
        # TAB 3 — Feature importance
        # ==============================================================
        with tab_imp:
            st.markdown("### Feature Importance for Cluster Separation")
            st.markdown(
                "Measured by how much each feature's variance is *between* clusters "
                "vs *within* clusters (F-statistic from one-way ANOVA)."
            )

            from sklearn.preprocessing import StandardScaler
            X = pairs_df[available_features].fillna(0)
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=available_features)
            X_scaled["cluster"] = pairs_df["cluster"].values

            f_scores = {}
            for feat in available_features:
                groups = [g[feat].values for _, g in X_scaled.groupby("cluster")]
                groups = [g for g in groups if len(g) >= 2]
                if len(groups) >= 2:
                    from scipy.stats import f_oneway
                    f_stat, p_val = f_oneway(*groups)
                    f_scores[feat] = f_stat
                else:
                    f_scores[feat] = 0

            importance_df = (
                pd.DataFrame.from_dict(f_scores, orient="index", columns=["F-statistic"])
                .sort_values("F-statistic", ascending=True)
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            colors = []
            paper_map = {
                "burstiness": "Ureña-Carrion 2020", "inter_event_regularity": "Ureña-Carrion 2020",
                "temporal_stability": "Ureña-Carrion 2020",
                "dispersion": "Backstrom 2014",
                "style_similarity": "Ireland 2011", "formality_diff": "Ireland 2011",
                "pronoun_rate_diff": "Ireland 2011",
                "avg_intimacy": "Reis & Shaver 1988", "intimacy_imbalance": "Reis & Shaver 1988",
                "intimacy_std": "Reis & Shaver 1988",
                "avg_warmth": "Reis & Shaver 1988", "warmth_imbalance": "Reis & Shaver 1988",
                "warmth_std": "Reis & Shaver 1988",
            }
            paper_colors = {
                "Ureña-Carrion 2020": "#e74c3c",
                "Backstrom 2014": "#3498db",
                "Ireland 2011": "#2ecc71",
                "Reis & Shaver 1988": "#9b59b6",
            }
            for feat in importance_df.index:
                paper = paper_map.get(feat, "Other")
                colors.append(paper_colors.get(paper, "#95a5a6"))

            bars = ax.barh(importance_df.index, importance_df["F-statistic"], color=colors)
            ax.set_xlabel("F-statistic (higher = better cluster separation)")
            ax.set_title("Which features separate clusters most?")

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=c, label=p) for p, c in paper_colors.items()
            ]
            legend_elements.append(Patch(facecolor="#95a5a6", label="Other"))
            ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ==============================================================
        # TAB 4 — Silhouette analysis per cluster
        # ==============================================================
        with tab_sil:
            st.markdown("### Silhouette Analysis by Cluster")
            st.markdown(
                "Silhouette score measures how similar a pair is to its own cluster "
                "vs the nearest other cluster. Range: -1 (wrong cluster) to +1 (perfect fit)."
            )

            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_samples, silhouette_score

            X = pairs_df[available_features].fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            labels = pairs_df["cluster"].values

            overall_score = silhouette_score(X_scaled, labels)
            sample_scores = silhouette_samples(X_scaled, labels)

            fig, ax = plt.subplots(figsize=(10, 7))
            y_lower = 10
            cluster_ids = sorted(pairs_df["cluster"].unique())

            for cid in cluster_ids:
                cluster_mask = labels == cid
                cluster_scores = sample_scores[cluster_mask]
                cluster_scores.sort()
                n = len(cluster_scores)
                y_upper = y_lower + n

                color = CLUSTER_COLORS[int(cid) % len(CLUSTER_COLORS)]
                label = cluster_names.get(int(cid), {}).get("name", f"Cluster {cid}")
                ax.fill_betweenx(
                    np.arange(y_lower, y_upper), 0, cluster_scores,
                    facecolor=color, edgecolor=color, alpha=0.7,
                )
                ax.text(-0.05, y_lower + 0.5 * n, label, fontsize=9, va="center")
                y_lower = y_upper + 10

            ax.axvline(x=overall_score, color="red", linestyle="--",
                       label=f"Overall: {overall_score:.3f}")
            ax.set_xlabel("Silhouette coefficient")
            ax.set_ylabel("Pairs (sorted within each cluster)")
            ax.set_title("Silhouette plot — per-cluster cohesion")
            ax.legend(loc="upper right")
            ax.set_yticks([])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Per-cluster summary table
            summary_rows = []
            for cid in cluster_ids:
                cluster_mask = labels == cid
                scores = sample_scores[cluster_mask]
                label = cluster_names.get(int(cid), {}).get("name", f"Cluster {cid}")
                summary_rows.append({
                    "Cluster": label,
                    "Pairs": int(cluster_mask.sum()),
                    "Mean silhouette": f"{scores.mean():.3f}",
                    "Min": f"{scores.min():.3f}",
                    "% negative": f"{(scores < 0).mean():.0%}",
                })
            st.dataframe(pd.DataFrame(summary_rows), width="stretch")

        # ==============================================================
        # TAB 5 — Baseline comparison
        # ==============================================================
        with tab_base:
            st.markdown("### Baseline Comparison")
            st.markdown(
                "Does the full 24-feature model actually outperform simpler approaches? "
                "We compare against two baselines to prove the scientific features add value."
            )

            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.metrics import (
                silhouette_score, davies_bouldin_score, calinski_harabasz_score,
            )

            X_full = pairs_df[available_features].fillna(0)

            # Define feature groups
            baseline1_cols = [c for c in ["email_count"] if c in pairs_df.columns]
            baseline2_cols = [c for c in [
                "email_count", "direction_ratio", "after_hours_ratio", "direct_ratio",
            ] if c in pairs_df.columns]
            paper_cols = {
                "Reis & Shaver": [c for c in [
                    "avg_intimacy", "intimacy_imbalance", "intimacy_std",
                    "avg_warmth", "warmth_imbalance", "warmth_std",
                ] if c in pairs_df.columns],
                "Ureña-Carrion": [c for c in [
                    "burstiness", "inter_event_regularity", "temporal_stability",
                ] if c in pairs_df.columns],
                "Backstrom": [c for c in ["dispersion"] if c in pairs_df.columns],
                "Ireland": [c for c in [
                    "style_similarity", "formality_diff", "pronoun_rate_diff",
                ] if c in pairs_df.columns],
            }

            # Use the same K as the pipeline chose
            best_k = pairs_df["cluster"].nunique()

            models = {}

            # Baseline 1: email count only
            if len(baseline1_cols) >= 1:
                X1 = pairs_df[baseline1_cols].fillna(0).values.reshape(-1, 1)
                s1 = StandardScaler()
                X1s = s1.fit_transform(X1)
                km1 = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                l1 = km1.fit_predict(X1s)
                models["Email count only"] = (X1s, l1)

            # Baseline 2: intensity features only
            if len(baseline2_cols) >= 2:
                X2 = pairs_df[baseline2_cols].fillna(0)
                s2 = StandardScaler()
                X2s = s2.fit_transform(X2)
                km2 = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                l2 = km2.fit_predict(X2s)
                models[f"Intensity only ({len(baseline2_cols)} features)"] = (X2s, l2)

            # Full model (use existing cluster labels)
            sf = StandardScaler()
            Xfs = sf.fit_transform(X_full)
            models[f"Full model ({len(available_features)} features)"] = (Xfs, pairs_df["cluster"].values)

            # Compute metrics for all
            comparison_rows = []
            for name, (X_data, labels) in models.items():
                n_labels = len(set(labels))
                if n_labels < 2:
                    comparison_rows.append({
                        "Model": name, "Silhouette": "—",
                        "Davies-Bouldin": "—", "Calinski-Harabasz": "—",
                    })
                    continue
                sil = silhouette_score(X_data, labels)
                db = davies_bouldin_score(X_data, labels)
                ch = calinski_harabasz_score(X_data, labels)
                comparison_rows.append({
                    "Model": name,
                    "Silhouette": round(sil, 3),
                    "Davies-Bouldin": round(db, 3),
                    "Calinski-Harabasz": round(ch, 1),
                })

            comp_df = pd.DataFrame(comparison_rows)
            st.dataframe(comp_df, width="stretch", hide_index=True)

            # Explanation of metrics
            st.caption(
                "**Silhouette:** higher = better separated clusters (range -1 to 1). "
                "**Davies-Bouldin:** lower = better (tighter clusters, more separated). "
                "**Calinski-Harabasz:** higher = better (dense, well-separated clusters)."
            )

            # Visual comparison
            metric_cols = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]
            numeric_df = comp_df[comp_df["Silhouette"] != "—"].copy()

            if len(numeric_df) >= 2:
                fig, axes = plt.subplots(1, 3, figsize=(14, 5))

                for idx, metric in enumerate(metric_cols):
                    ax = axes[idx]
                    values = numeric_df[metric].astype(float)
                    bars = ax.bar(
                        range(len(numeric_df)),
                        values,
                        color=["#95a5a6", "#f39c12", "#2ecc71"][:len(numeric_df)],
                        edgecolor="white",
                    )
                    ax.set_xticks(range(len(numeric_df)))
                    ax.set_xticklabels(
                        [m.split("(")[0].strip() for m in numeric_df["Model"]],
                        fontsize=9,
                    )
                    ax.set_title(metric, fontsize=12)

                    # Add value labels on bars
                    for bar, val in zip(bars, values):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.3f}" if val < 100 else f"{val:.0f}",
                            ha="center", va="bottom", fontsize=9,
                        )

                    # Arrow showing which direction is better
                    better = "higher ↑" if metric != "Davies-Bouldin" else "lower ↓"
                    ax.set_xlabel(f"({better} is better)", fontsize=8, color="#888")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Per-paper ablation
            st.markdown("---")
            st.markdown("### Per-Paper Feature Contribution")
            st.markdown(
                "What happens when we remove one paper's features at a time? "
                "A big drop means that paper's features are essential."
            )

            ablation_rows = []
            full_sil = silhouette_score(Xfs, pairs_df["cluster"].values)

            for paper_name, paper_feats in paper_cols.items():
                if not paper_feats:
                    continue
                remaining = [c for c in available_features if c not in paper_feats]
                if len(remaining) < 2:
                    continue
                X_abl = pairs_df[remaining].fillna(0)
                s_abl = StandardScaler()
                X_abl_s = s_abl.fit_transform(X_abl)
                km_abl = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                l_abl = km_abl.fit_predict(X_abl_s)
                sil_abl = silhouette_score(X_abl_s, l_abl)
                drop = full_sil - sil_abl
                ablation_rows.append({
                    "Removed": paper_name,
                    "Features removed": len(paper_feats),
                    "Remaining": len(remaining),
                    "Silhouette": round(sil_abl, 3),
                    "Drop": round(drop, 3),
                })

            if ablation_rows:
                abl_df = pd.DataFrame(ablation_rows)

                fig, ax = plt.subplots(figsize=(10, 5))
                colors = {
                    "Reis & Shaver": "#9b59b6",
                    "Ureña-Carrion": "#e74c3c",
                    "Backstrom": "#3498db",
                    "Ireland": "#2ecc71",
                }
                bar_colors = [colors.get(r, "#95a5a6") for r in abl_df["Removed"]]

                bars = ax.bar(abl_df["Removed"], abl_df["Drop"], color=bar_colors, edgecolor="white")
                ax.axhline(y=0, color="#888", linewidth=0.5)
                ax.set_ylabel("Silhouette drop when removed")
                ax.set_title(
                    f"Ablation study — full model silhouette: {full_sil:.3f}"
                )

                for bar, val in zip(bars, abl_df["Drop"]):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{val:+.3f}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold",
                    )

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.caption(
                    "Positive drop = removing those features hurts the model. "
                    "Bigger bar = that paper's features are more important."
                )

                st.dataframe(abl_df, width="stretch", hide_index=True)

        # ==============================================================
        # TAB 6 — Email volume timeline (renumbered)
        # ==============================================================
        with tab_timeline:
            st.markdown("### Email Volume Over Time")
            st.markdown("Shows the rhythm of communication at Enron — volume spikes, quiet periods, and the lead-up to collapse.")

            emails_df = load_emails()
            if emails_df is not None and "date" in emails_df.columns:
                timeline = emails_df.copy()
                timeline["month"] = timeline["date"].dt.tz_localize(None).dt.to_period("M").astype(str)
                monthly = timeline.groupby("month").size().reset_index(name="emails")
                monthly = monthly.sort_values("month")

                fig, ax = plt.subplots(figsize=(12, 5))
                ax.fill_between(range(len(monthly)), monthly["emails"], alpha=0.3, color="#3498db")
                ax.plot(range(len(monthly)), monthly["emails"], color="#3498db", linewidth=1.5)
                ax.set_xticks(range(0, len(monthly), max(1, len(monthly) // 12)))
                ax.set_xticklabels(
                    monthly["month"].iloc[::max(1, len(monthly) // 12)],
                    rotation=45, ha="right", fontsize=8,
                )
                ax.set_ylabel("Emails per month")
                ax.set_title("Enron email volume over time")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                col1, col2, col3 = st.columns(3)
                col1.metric("Total emails", f"{len(emails_df):,}")
                col2.metric("Date range", f"{monthly['month'].iloc[0]} to {monthly['month'].iloc[-1]}")
                col3.metric("Peak month", monthly.loc[monthly["emails"].idxmax(), "month"])
            else:
                st.info("Email data not available. Run main.py first.")

        # ==============================================================
        # TAB 6 — Score distributions
        # ==============================================================
        with tab_dist:
            st.markdown("### Stage 1 Score Distributions")
            st.markdown(
                "Shows the output of our self-disclosure, responsiveness, and sentiment classifiers. "
                "Good classifiers produce spread-out distributions, not a single spike."
            )

            score_cols = {
                "avg_intimacy": ("Self-Disclosure (pair avg)", "#9b59b6"),
                "avg_warmth": ("Responsiveness (pair avg)", "#e67e22"),
                "avg_sentiment": ("Sentiment (pair avg)", "#3498db"),
            }

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for idx, (col, (label, color)) in enumerate(score_cols.items()):
                if col in pairs_df.columns:
                    ax = axes[idx]
                    data = pairs_df[col].dropna()
                    ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor="white")
                    ax.axvline(data.mean(), color="red", linestyle="--",
                               label=f"mean={data.mean():.3f}")
                    ax.set_title(label, fontsize=11)
                    ax.set_ylabel("Pairs")
                    ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # New features distributions
            st.markdown("### New Feature Distributions")
            st.markdown("Temporal and structural features added based on recent research.")

            new_cols = {
                "burstiness": ("Burstiness (Ureña-Carrion 2020)", "#e74c3c"),
                "temporal_stability": ("Temporal Stability", "#2ecc71"),
                "dispersion": ("Dispersion (Backstrom 2014)", "#3498db"),
            }

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for idx, (col, (label, color)) in enumerate(new_cols.items()):
                if col in pairs_df.columns:
                    ax = axes[idx]
                    data = pairs_df[col].dropna()
                    ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor="white")
                    ax.axvline(data.mean(), color="red", linestyle="--",
                               label=f"mean={data.mean():.3f}")
                    ax.set_title(label, fontsize=11)
                    ax.set_ylabel("Pairs")
                    ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Dispersion vs shared neighbours scatter
            if "dispersion" in pairs_df.columns and "shared_neighbors" in pairs_df.columns:
                st.markdown("### Dispersion vs Shared Neighbours")
                st.markdown(
                    "If these are uncorrelated, dispersion captures something different "
                    "from simple embeddedness — justifying its inclusion (Backstrom & Kleinberg 2014)."
                )
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(
                    pairs_df["shared_neighbors"], pairs_df["dispersion"],
                    alpha=0.4, c="#3498db", edgecolors="white", linewidths=0.3,
                )
                r = pairs_df["shared_neighbors"].corr(pairs_df["dispersion"])
                ax.set_xlabel("Shared Neighbours (embeddedness)")
                ax.set_ylabel("Dispersion")
                ax.set_title(f"Dispersion vs Embeddedness (r = {r:.3f})")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    else:
        st.info("Run main.py to generate results.")


# ================================================================
# PAGE 6 — Human Validation (3-level annotation tool)
# ================================================================
elif page == "🏷️ Human Validation":
    st.title("🏷️ Human Validation")
    st.markdown(
        "Validate the pipeline at every stage. Your labels measure how well "
        "the ML pipeline matches human judgment."
    )

    # --- Shared: annotator name in sidebar ---
    annotator = st.sidebar.text_input("Your name", value="", placeholder="e.g. Atli")
    if not annotator.strip():
        st.info("Enter your name in the sidebar to start labeling.")
    else:
        annotator = annotator.strip()

        tab1, tab2, tab3, tab4 = st.tabs([
            "Email Labels", "Relationship Review", "Cluster Review", "Results Dashboard"
        ])

        claude_labeled = load_claude_labeled()
        emails_df = load_emails()

        # ==============================================================
        # TAB 1 — Email-level labeling (active learning)
        # Shows the MOST UNCERTAIN emails first — where your label
        # matters most. Labels feed back into classifier training.
        # ==============================================================
        with tab1:
            if claude_labeled is None:
                st.warning("No Claude-labeled emails found. Run `python main.py` first.")
            else:
                human_labels = load_human_labels()

                # Active learning: sort by classifier uncertainty
                # Emails closest to the decision boundary (intimacy_label 2 or 3)
                # are the hardest for the classifier — label these first
                pool = claude_labeled.copy()

                # Compute uncertainty: distance from decision boundary (2.5)
                pool["_uncertainty"] = (pool["intimacy_label"] - 2.5).abs()
                # Most uncertain first (closest to boundary)
                pool = pool.sort_values("_uncertainty", ascending=True)
                # Take top 30 most uncertain (focused pool for 1 person)
                label_pool = pool.head(30).reset_index(drop=True)
                label_pool = label_pool.drop(columns=["_uncertainty"])

                done_ids = set()
                if len(human_labels) > 0:
                    done_ids = set(
                        human_labels[human_labels["annotator"] == annotator]["message_id"].tolist()
                    )

                remaining = label_pool[~label_pool["message_id"].isin(done_ids)]
                n_done = len(done_ids & set(label_pool["message_id"].tolist()))
                n_total = len(label_pool)

                st.caption(
                    "Active learning: showing the most uncertain emails first — "
                    "where your label has the biggest impact on model quality. "
                    "Your labels override Claude's and are used for retraining."
                )
                streak = st.session_state.get("label_streak", 0)

                # Init selection state
                if "sel_intimacy" not in st.session_state:
                    st.session_state["sel_intimacy"] = None

                sel_i = st.session_state["sel_intimacy"]

                # --- Header: progress + streak ---
                hcol1, hcol2 = st.columns([5, 1])
                with hcol1:
                    st.progress(n_done / n_total, text=f"{n_done}/{n_total} emails labeled")
                with hcol2:
                    st.metric("Streak", f"{streak}")

                # --- Reveal banner from last submission ---
                if "last_submitted_id" in st.session_state and st.session_state["last_submitted_id"]:
                    last_id = st.session_state["last_submitted_id"]
                    claude_row = claude_labeled[claude_labeled["message_id"] == last_id]
                    if len(claude_row) > 0:
                        cr = claude_row.iloc[0]
                        human_row = human_labels[
                            (human_labels["message_id"] == last_id) &
                            (human_labels["annotator"] == annotator)
                        ]
                        if len(human_row) > 0:
                            hr = human_row.iloc[0]
                            both_match = (hr["intimacy_label"] == cr["intimacy_label"]
                                          and hr["warmth_label"] == cr["warmth_label"])
                            close = (abs(hr["intimacy_label"] - cr["intimacy_label"]) <= 1
                                     and abs(hr["warmth_label"] - cr["warmth_label"]) <= 1)

                            if both_match:
                                color, icon, word = "#2ecc71", "✓", "Perfect match!"
                            elif close:
                                color, icon, word = "#f39c12", "~", "Close!"
                            else:
                                color, icon, word = "#e74c3c", "✗", "Different"

                            st.markdown(
                                f"<div style='background:{color};padding:10px 16px;border-radius:8px;"
                                f"color:white;margin-bottom:8px;font-size:0.95em'>"
                                f"<b>{icon} {word}</b> &nbsp; "
                                f"You: {int(hr['intimacy_label'])},{int(hr['warmth_label'])} "
                                f"| Claude: {int(cr['intimacy_label'])},{int(cr['warmth_label'])}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    st.session_state["last_submitted_id"] = None

                if len(remaining) == 0:
                    st.success(f"All {n_total} emails labeled! Check the Results Dashboard tab.")
                else:
                    row = remaining.iloc[0]
                    body_text = row["body"] if isinstance(row["body"], str) else "(empty email)"
                    subject = row["subject"] if ("subject" in row.index and isinstance(row["subject"], str)) else ""
                    sender_addr = row["sender"] if "sender" in row.index else ""

                    INTIMACY_COLORS = ["#3498db", "#5dade2", "#aab7b8", "#e67e22", "#e74c3c"]
                    INTIMACY_LABELS = ["None", "Low", "Mixed", "High", "Very high"]
                    INTIMACY_HINTS = [
                        "Reports, contracts, scheduling — zero personal content",
                        "Work email with a 'hope you're well' or small talk",
                        "Half work, half personal — social plans mixed with business",
                        "Mostly personal: feelings, family, social plans, venting",
                        "Deeply private: intimate feelings, secrets, vulnerability",
                    ]
                    WARMTH_COLORS = ["#2c3e50", "#7f8c8d", "#95a5a6", "#f39c12", "#e74c3c"]
                    WARMTH_LABELS = ["Hostile", "Cold", "Neutral", "Caring", "Deeply caring"]
                    WARMTH_HINTS = [
                        "Angry, threatening, blaming, confrontational",
                        "Curt, dismissive, no greeting, one-word replies",
                        "Professional tone — neither warm nor cold",
                        "Friendly, supportive, asks how they're doing, offers help",
                        "Deeply caring, validating feelings, emotional support",
                    ]

                    # --- Side-by-side: email left, labels right ---
                    left_col, right_col = st.columns([3, 2], gap="large")

                    with left_col:
                        st.markdown(f"**Email {n_done + 1} of {n_total}**")
                        st.markdown(f"**From:** {sender_addr}")
                        st.markdown(f"**Subject:** {subject}")
                        st.divider()
                        st.code(body_text[:2000], language=None)

                    with right_col:
                        # --- STEP 1: Pick intimacy (or show what was picked) ---
                        if sel_i is None:
                            st.markdown("**Step 1 — Self-Disclosure** (does the sender share personal info?)")
                            int_cols = st.columns(5)
                            for i in range(5):
                                val = i + 1
                                with int_cols[i]:
                                    st.markdown(
                                        f"<div style='background:{INTIMACY_COLORS[i]};padding:8px 2px;"
                                        f"border-radius:8px;text-align:center;color:white;"
                                        f"font-size:0.75em;line-height:1.3;margin-bottom:4px'>"
                                        f"<div style='font-size:1.6em;font-weight:bold'>{val}</div>"
                                        f"{INTIMACY_LABELS[i]}"
                                        f"<div style='font-size:0.7em;opacity:0.8;margin-top:3px'>"
                                        f"{INTIMACY_HINTS[i]}</div></div>",
                                        unsafe_allow_html=True,
                                    )
                                    if st.button(f"Pick {val}", key=f"int_{val}"):
                                        st.session_state["sel_intimacy"] = val
                                        st.rerun()
                        else:
                            # Intimacy already picked — show it and ask for warmth
                            st.markdown(
                                f"<div style='background:{INTIMACY_COLORS[sel_i-1]};padding:8px 12px;"
                                f"border-radius:8px;color:white;margin-bottom:8px'>"
                                f"Self-Disclosure: <b>{sel_i} — {INTIMACY_LABELS[sel_i-1]}</b></div>",
                                unsafe_allow_html=True,
                            )

                            # --- STEP 2: Pick warmth → auto-saves ---
                            st.markdown("**Step 2 — Responsiveness** (does the sender show care/support?)")
                            warm_cols = st.columns(5)
                            for i in range(5):
                                val = i + 1
                                with warm_cols[i]:
                                    st.markdown(
                                        f"<div style='background:{WARMTH_COLORS[i]};padding:8px 2px;"
                                        f"border-radius:8px;text-align:center;color:white;"
                                        f"font-size:0.75em;line-height:1.3;margin-bottom:4px'>"
                                        f"<div style='font-size:1.6em;font-weight:bold'>{val}</div>"
                                        f"{WARMTH_LABELS[i]}"
                                        f"<div style='font-size:0.7em;opacity:0.8;margin-top:3px'>"
                                        f"{WARMTH_HINTS[i]}</div></div>",
                                        unsafe_allow_html=True,
                                    )
                                    if st.button(f"Pick {val}", key=f"warm_{val}"):
                                        # Both selected — save and advance
                                        save_human_label(row["message_id"], annotator, sel_i, val)
                                        st.session_state["last_submitted_id"] = row["message_id"]
                                        cr2 = claude_labeled[
                                            claude_labeled["message_id"] == row["message_id"]
                                        ]
                                        if len(cr2) > 0:
                                            c = cr2.iloc[0]
                                            if sel_i == c["intimacy_label"] and val == c["warmth_label"]:
                                                st.session_state["label_streak"] = streak + 1
                                            else:
                                                st.session_state["label_streak"] = 0
                                        st.session_state["sel_intimacy"] = None
                                        st.rerun()

                        # --- Spam / skip button (always visible) ---
                        st.divider()
                        if st.button("Skip — junk / spam / irrelevant", key="t1_spam"):
                            save_human_label(row["message_id"], annotator, 0, 0)
                            st.session_state["sel_intimacy"] = None
                            st.session_state["last_submitted_id"] = None
                            st.rerun()

        # ==============================================================
        # TAB 2 — Pair-level relationship review (uncertainty-sorted)
        # Shows the MOST UNCERTAIN pairs first — where cluster
        # assignment is ambiguous. Your review validates edge cases.
        # ==============================================================
        with tab2:
            if pairs_df is None or emails_df is None:
                st.warning("Run `python main.py` first to generate results.")
            else:
                pair_labels = load_pair_labels()

                # Build pool: 15 pairs sorted by cluster uncertainty
                review_pairs = pairs_df.copy()
                if "primary_prob" in review_pairs.columns:
                    # GMM: sort by lowest primary probability (most uncertain)
                    review_pairs = review_pairs.sort_values(
                        "primary_prob", ascending=True
                    )
                    review_pool = review_pairs.head(15)
                    st.caption(
                        "Active learning: showing pairs with the most uncertain "
                        "cluster assignment first — where your judgment matters most."
                    )
                elif "cluster" in review_pairs.columns:
                    # Fallback: stratified sample from each cluster
                    sampled = review_pairs.groupby("cluster").apply(
                        lambda g: g.nlargest(min(5, len(g)), "email_count"),
                        include_groups=False,
                    ).reset_index(drop=True)
                    review_pool = sampled.head(15)
                else:
                    review_pool = review_pairs.nlargest(15, "email_count")

                done_pairs = set()
                if len(pair_labels) > 0:
                    ann_done = pair_labels[pair_labels["annotator"] == annotator]
                    for _, r in ann_done.iterrows():
                        done_pairs.add((r["person_a"], r["person_b"]))

                remaining_pairs = review_pool[
                    ~review_pool.apply(lambda r: (r["person_a"], r["person_b"]) in done_pairs, axis=1)
                ]

                n_done_p = len(done_pairs & set(
                    zip(review_pool["person_a"], review_pool["person_b"])
                ))
                n_total_p = len(review_pool)

                st.progress(
                    n_done_p / max(n_total_p, 1),
                    text=f"{n_done_p}/{n_total_p} pairs reviewed"
                )

                # --- Reveal banner from last pair ---
                if "last_pair_reveal" in st.session_state and st.session_state["last_pair_reveal"]:
                    rev = st.session_state["last_pair_reveal"]
                    match = rev["pipeline"] in rev["human"]
                    color = "#2ecc71" if match else "#e74c3c"
                    word = "Match!" if match else "Different"
                    st.markdown(
                        f"<div style='background:{color};padding:10px 16px;border-radius:8px;"
                        f"color:white;margin-bottom:8px;font-size:0.95em'>"
                        f"<b>{'✓' if match else '✗'} {word}</b> &nbsp; "
                        f"You: {rev['human']} | Pipeline: {rev['pipeline']}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.session_state["last_pair_reveal"] = None

                if len(remaining_pairs) == 0:
                    st.success(f"You have reviewed all {n_total_p} pairs!")
                else:
                    pair_row = remaining_pairs.iloc[0]
                    pa, pb = pair_row["person_a"], pair_row["person_b"]

                    # Pipeline's prediction
                    pipeline_type = "Unknown"
                    if cluster_names and "cluster" in pair_row.index:
                        cid = int(pair_row["cluster"])
                        pipeline_type = cluster_names.get(cid, {}).get("name", f"Cluster {cid}")

                    # --- Layout: emails left, label right ---
                    left_col, right_col = st.columns([3, 2], gap="large")

                    with left_col:
                        st.markdown(
                            f"**Pair {n_done_p + 1} of {n_total_p}:** "
                            f"**{_short_name(pa)}** and **{_short_name(pb)}**"
                        )

                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Emails", int(pair_row["email_count"]))
                        if "time_span_days" in pair_row.index:
                            col2.metric("Span", f"{int(pair_row['time_span_days'])}d")
                        if "temporal_stability" in pair_row.index:
                            col3.metric("Stability", f"{pair_row['temporal_stability']:.2f}")
                        if "burstiness" in pair_row.index:
                            col4.metric("Burstiness", f"{pair_row['burstiness']:+.2f}")
                        if "dispersion" in pair_row.index:
                            col5.metric("Dispersion", f"{pair_row['dispersion']:.2f}")

                        # Show emails
                        mask = (
                            (emails_df["sender"].eq(pa) &
                             emails_df["recipients"].str.contains(pb, na=False, regex=False)) |
                            (emails_df["sender"].eq(pb) &
                             emails_df["recipients"].str.contains(pa, na=False, regex=False))
                        )
                        matched = emails_df[mask].sort_values("date")

                        for i, (_, erow) in enumerate(matched.head(6).iterrows()):
                            date_str = erow["date"].strftime("%Y-%m-%d") if pd.notna(erow["date"]) else ""
                            subj = erow["subject"] if isinstance(erow["subject"], str) else ""
                            body = erow["body"] if isinstance(erow["body"], str) else ""
                            sender_name = _short_name(erow["sender"])
                            with st.expander(f"{sender_name} — {date_str} — {subj}", expanded=(i < 2)):
                                st.text(body[:600])

                    with right_col:
                        st.markdown("**Pick all that apply**, then submit:")

                        # Build options from what the pipeline actually discovered
                        rel_options = []
                        if cluster_names:
                            for cid_opt in sorted(cluster_names.keys()):
                                name = cluster_names[cid_opt].get("name", f"Cluster {cid_opt}")
                                if name not in rel_options:
                                    rel_options.append(name)

                        for fallback in ["Transactional", "Friendly Colleagues",
                                         "Close", "Boss-Employee",
                                         "Mentor", "Romance", "Tense / Conflict",
                                         "Fading"]:
                            if fallback not in rel_options:
                                rel_options.append(fallback)
                        rel_options.append("Spam/Junk")

                        REL_COLORS = {
                            "Transactional": "#95a5a6",
                            "Friendly Colleagues": "#2ecc71",
                            "Close": "#9b59b6",
                            "Boss-Employee": "#3498db",
                            "Mentor": "#f39c12",
                            "Romance": "#e91e8c",
                            "Tense / Conflict": "#e74c3c",
                            "Fading": "#7f8c8d",
                            "Spam/Junk": "#2c3e50",
                        }

                        with st.form("pair_label_form"):
                            selected = []
                            for rel in rel_options:
                                color = REL_COLORS.get(rel, "#555")
                                checked = st.checkbox(
                                    rel, key=f"t2_cb_{rel}",
                                )
                                if checked:
                                    selected.append(rel)

                            pair_submitted = st.form_submit_button(
                                "Submit & reveal", type="primary",
                            )

                        if pair_submitted and selected:
                            rel_str = " + ".join(selected)
                            save_pair_label(pa, pb, annotator, rel_str, "Medium", "")
                            # Match = pipeline type is one of the selected
                            st.session_state["last_pair_reveal"] = {
                                "human": rel_str,
                                "pipeline": pipeline_type,
                            }
                            st.rerun()
                        elif pair_submitted:
                            st.error("Pick at least one type.")

        # ==============================================================
        # TAB 3 — Cluster name editor (validates Stage 4)
        # Claude names first, humans edit. Changes saved to cluster_names.json.
        # ==============================================================
        with tab3:
            st.markdown("### Cluster Name Editor")
            st.markdown(
                "Claude suggested a name for each cluster. "
                "Review the profile and **rename** any cluster that doesn't fit. "
                "Changes are saved immediately."
            )

            if not cluster_names or pairs_df is None:
                st.warning("Run `python main.py` first to generate clusters.")
            else:
                # Load z-scores if available
                zscores_path = RESULTS_PATH / "cluster_zscores.csv"
                zscores_df = None
                if zscores_path.exists():
                    zscores_df = pd.read_csv(zscores_path, index_col="cluster")

                for cid in sorted(cluster_names.keys()):
                    info = cluster_names[cid]
                    cluster_pairs_df = pairs_df[pairs_df["cluster"] == cid]
                    n_pairs = len(cluster_pairs_df)

                    color = RELATIONSHIP_COLORS.get(
                        info["name"], CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])

                    # --- Cluster header with inline edit ---
                    head_col, edit_col = st.columns([3, 2])

                    with head_col:
                        st.markdown(
                            f"<div style='background:{color};padding:14px;border-radius:8px;"
                            f"color:white;margin:4px 0'>"
                            f"<div style='font-size:1.2em;font-weight:bold'>"
                            f"Cluster {cid}: {info['name']}</div>"
                            f"<div style='font-size:0.85em;opacity:0.9'>"
                            f"{n_pairs} pairs — {info.get('description', '')}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    with edit_col:
                        with st.form(f"rename_{cid}"):
                            new_name = st.text_input(
                                "Rename",
                                value=info["name"],
                                key=f"t3_name_{cid}",
                                label_visibility="collapsed",
                            )
                            new_desc = st.text_input(
                                "Description",
                                value=info.get("description", ""),
                                key=f"t3_desc_{cid}",
                                placeholder="Why this name?",
                                label_visibility="collapsed",
                            )
                            if st.form_submit_button("Save", type="primary"):
                                # Update in memory and on disk
                                cluster_names[cid]["name"] = new_name.strip()
                                cluster_names[cid]["description"] = new_desc.strip()
                                names_path = RESULTS_PATH / "cluster_names.json"
                                with open(names_path, "w") as f:
                                    json.dump(
                                        {str(k): v for k, v in cluster_names.items()},
                                        f, indent=2,
                                    )
                                # Also update relationship_pairs.csv
                                if "relationship_type" in pairs_df.columns:
                                    name_map = {
                                        c: cluster_names[c]["name"]
                                        for c in cluster_names
                                    }
                                    pairs_df["relationship_type"] = pairs_df["cluster"].map(name_map)
                                    pairs_df.to_csv(
                                        RESULTS_PATH / "relationship_pairs.csv",
                                        index=False,
                                    )
                                st.success(f"Cluster {cid} renamed to '{new_name.strip()}'")
                                st.cache_data.clear()
                                st.rerun()

                    # --- Cluster profile: key stats + top distinctive features ---
                    with st.expander(f"Profile & pairs for cluster {cid}"):
                        # Key stats
                        stat_cols = st.columns(4)
                        stat_map = [
                            ("Disclosure", "avg_intimacy"),
                            ("Responsiveness", "avg_warmth"),
                            ("Stability", "temporal_stability"),
                            ("Dispersion", "dispersion"),
                        ]
                        for idx, (label, col) in enumerate(stat_map):
                            if col in cluster_pairs_df.columns:
                                val = cluster_pairs_df[col].mean()
                                fmt = f"{val:.2f}" if val < 10 else f"{val:.0f}"
                                stat_cols[idx].metric(label, fmt)

                        # Top distinctive features from z-scores
                        if zscores_df is not None and cid in zscores_df.index:
                            zrow = zscores_df.loc[cid]
                            top_feats = zrow.abs().nlargest(5)
                            st.markdown("**Most distinctive features:**")
                            for feat in top_feats.index:
                                z = zrow[feat]
                                direction = "above" if z > 0 else "below"
                                bar_color = "#2ecc71" if z > 0 else "#e74c3c"
                                bar_width = min(abs(z) * 20, 100)
                                st.markdown(
                                    f"<div style='display:flex;align-items:center;gap:8px;"
                                    f"margin:2px 0;font-size:0.85em'>"
                                    f"<span style='width:160px'>{feat}</span>"
                                    f"<div style='background:{bar_color};height:14px;"
                                    f"width:{bar_width}%;border-radius:3px'></div>"
                                    f"<span style='color:#888'>z={z:+.2f}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                        # Top pairs table
                        st.markdown("**Top pairs:**")
                        top = cluster_pairs_df.nlargest(5, "email_count")
                        display_cols = ["person_a", "person_b", "avg_intimacy",
                                        "avg_warmth", "email_count"]
                        available_cols = [c for c in display_cols if c in top.columns]
                        display = top[available_cols].copy()
                        display["Person A"] = display["person_a"].apply(_short_name)
                        display["Person B"] = display["person_b"].apply(_short_name)
                        show_cols = ["Person A", "Person B"] + [
                            c for c in available_cols if c not in ("person_a", "person_b")
                        ]
                        st.dataframe(display[show_cols].reset_index(drop=True))

                    st.markdown("")

        # ==============================================================
        # TAB 4 — Results Dashboard
        # ==============================================================
        with tab4:
            st.markdown("### Validation Results Dashboard")

            from sklearn.metrics import cohen_kappa_score

            claude_labeled = load_claude_labeled()
            human_labels = load_human_labels()
            pair_labels = load_pair_labels()
            cluster_labels_df = load_cluster_labels()

            # --- Email-level agreement ---
            st.markdown("---")
            st.markdown("### Stage 1: Email Label Agreement")

            if claude_labeled is not None and len(human_labels) >= 10:
                label_pool = claude_labeled.sample(
                    n=min(50, len(claude_labeled)), random_state=123
                ).reset_index(drop=True)
                pool_ids = set(label_pool["message_id"].tolist())

                claude_subset = claude_labeled[claude_labeled["message_id"].isin(pool_ids)][
                    ["message_id", "intimacy_label", "warmth_label"]
                ].rename(columns={
                    "intimacy_label": "claude_intimacy",
                    "warmth_label": "claude_warmth",
                })

                pool_human = human_labels[human_labels["message_id"].isin(pool_ids)]
                annotators = pool_human["annotator"].unique()

                if len(annotators) == 0:
                    st.info("No email labels yet.")
                else:
                    rows = []
                    for ann in annotators:
                        ann_labels = pool_human[pool_human["annotator"] == ann][
                            ["message_id", "intimacy_label", "warmth_label"]
                        ].rename(columns={
                            "intimacy_label": "human_intimacy",
                            "warmth_label": "human_warmth",
                        })
                        merged = ann_labels.merge(claude_subset, on="message_id")
                        if len(merged) >= 5:
                            ki = cohen_kappa_score(merged["human_intimacy"], merged["claude_intimacy"])
                            kw = cohen_kappa_score(merged["human_warmth"], merged["claude_warmth"])
                            ai = (merged["human_intimacy"] == merged["claude_intimacy"]).mean()
                            aw = (merged["human_warmth"] == merged["claude_warmth"]).mean()
                            rows.append({
                                "Annotator": ann,
                                "Emails": len(merged),
                                "Disclosure Kappa": f"{ki:.3f}",
                                "Disclosure Agree": f"{ai:.0%}",
                                "Responsiveness Kappa": f"{kw:.3f}",
                                "Responsiveness Agree": f"{aw:.0%}",
                            })

                    if rows:
                        st.dataframe(pd.DataFrame(rows), width='stretch')

                    # Inter-human agreement (if 2+ annotators)
                    if len(annotators) >= 2:
                        st.markdown("#### Inter-human agreement")
                        for i, a1 in enumerate(annotators):
                            for a2 in annotators[i+1:]:
                                l1 = pool_human[pool_human["annotator"] == a1][
                                    ["message_id", "intimacy_label", "warmth_label"]
                                ].rename(columns={"intimacy_label": "int_1", "warmth_label": "warm_1"})
                                l2 = pool_human[pool_human["annotator"] == a2][
                                    ["message_id", "intimacy_label", "warmth_label"]
                                ].rename(columns={"intimacy_label": "int_2", "warmth_label": "warm_2"})
                                both = l1.merge(l2, on="message_id")
                                if len(both) >= 5:
                                    ki = cohen_kappa_score(both["int_1"], both["int_2"])
                                    kw = cohen_kappa_score(both["warm_1"], both["warm_2"])
                                    st.markdown(
                                        f"**{a1} vs {a2}** ({len(both)} shared): "
                                        f"disclosure kappa={ki:.3f}, responsiveness kappa={kw:.3f}"
                                    )
            else:
                st.info("Need Claude labels and at least 10 human labels to show agreement.")

            # --- Pair-level agreement ---
            st.markdown("---")
            st.markdown("### Stage 3+4: Relationship Label Accuracy")

            if pairs_df is not None and len(pair_labels) > 0:
                # Compare human labels to pipeline cluster names
                results = []
                for _, pl in pair_labels.iterrows():
                    match = pairs_df[
                        ((pairs_df["person_a"] == pl["person_a"]) &
                         (pairs_df["person_b"] == pl["person_b"]))
                    ]
                    if len(match) > 0:
                        cid = int(match.iloc[0]["cluster"])
                        pipeline_name = cluster_names.get(cid, {}).get("name", f"Cluster {cid}")
                        results.append({
                            "Annotator": pl["annotator"],
                            "Person A": _short_name(pl["person_a"]),
                            "Person B": _short_name(pl["person_b"]),
                            "Human Label": pl["relationship_type"],
                            "Pipeline Label": pipeline_name,
                            "Match": pipeline_name in str(pl["relationship_type"]),
                            "Confidence": pl["confidence"],
                        })

                if results:
                    results_df = pd.DataFrame(results)
                    accuracy = results_df["Match"].mean()

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Pairs reviewed", len(results_df))
                    col2.metric("Pipeline accuracy", f"{accuracy:.0%}")
                    col3.metric("Annotators", results_df["Annotator"].nunique())

                    # Color the match column
                    st.dataframe(results_df, width='stretch')

                    # Confusion summary
                    if len(results_df) >= 5:
                        st.markdown("#### Where the pipeline disagrees with humans")
                        disagreements = results_df[~results_df["Match"]]
                        if len(disagreements) > 0:
                            st.dataframe(
                                disagreements[["Person A", "Person B", "Human Label",
                                              "Pipeline Label", "Annotator"]],
                                width='stretch',
                            )
                        else:
                            st.success("Perfect agreement on all reviewed pairs!")
                else:
                    st.info("No pair labels match pipeline results yet.")
            else:
                st.info("No pair-level labels yet. Use the Relationship Review tab.")

            # --- Cluster-level agreement ---
            st.markdown("---")
            st.markdown("### Stage 4: Cluster Name Validation")

            if len(cluster_labels_df) > 0 and cluster_names:
                for cid in sorted(cluster_names.keys()):
                    reviews = cluster_labels_df[cluster_labels_df["cluster"] == cid]
                    if len(reviews) == 0:
                        continue
                    n_agree = reviews["name_correct"].sum()
                    n_total_r = len(reviews)
                    info = cluster_names[cid]
                    color = "#2ecc71" if n_agree == n_total_r else "#f39c12" if n_agree > 0 else "#e74c3c"
                    st.markdown(
                        f"<div style='background:{color};padding:8px;border-radius:6px;"
                        f"color:white;margin:4px 0'>"
                        f"<b>{info['name']}</b> — {n_agree}/{n_total_r} agree"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    suggestions = reviews[~reviews["name_correct"]]["suggested_name"].tolist()
                    suggestions = [s for s in suggestions if s]
                    if suggestions:
                        st.caption(f"Suggested alternatives: {', '.join(suggestions)}")
            else:
                st.info("No cluster reviews yet. Use the Cluster Review tab.")


# ================================================================
# PAGE 7 — Run (Predict on all data)
# ================================================================
elif page == "🚀 Run":
    st.title("🚀 Run — Predict All Emails")
    st.markdown(
        "Load your trained models and apply them to **all** emails. "
        "No Claude calls, no training — just prediction."
    )

    # Check readiness
    models_exist = (RESULTS_PATH / "model_intimacy.pkl").exists()
    labels_path = RESULTS_PATH / "claude_labeled_emails.csv"
    accumulated_labels = 0
    if labels_path.exists():
        accumulated_labels = max(sum(1 for _ in open(labels_path)) - 1, 0)

    total_emails_available = 0
    load_file = Path("data/processed/emails.parquet")
    if load_file.exists():
        import pyarrow.parquet as pq
        try:
            total_emails_available = pq.read_metadata(str(load_file)).num_rows
        except Exception:
            pass

    if not models_exist:
        st.error(
            "**No trained models found.** Go to the Training page first and run "
            "at least one training batch to create the models."
        )
        st.markdown("### What you need to do:")
        st.markdown(
            "1. Go to **🏋️ Training**\n"
            "2. Run at least one sample batch\n"
            "3. Optionally label emails in **🏷️ Human Validation**\n"
            "4. Come back here to run on all data"
        )
    else:
        # --- STATUS ---
        st.markdown("### Ready to run")

        col1, col2, col3 = st.columns(3)
        col1.metric("Trained on", f"{accumulated_labels:,} labels")
        col2.metric("Will predict", f"{total_emails_available:,} emails")
        col3.metric("Models", "Ready" if models_exist else "Missing")

        st.success(
            f"**Predict mode:** No Claude calls. No training. Your saved models "
            f"will score all {total_emails_available:,} emails, build pair features, "
            f"cluster relationships, and name them.\n\n"
            f"This produces your final results for the report."
        )

        # --- PARAMETERS (mostly fixed for production) ---
        st.markdown("### Parameters")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        min_pair = p_col1.number_input("Min emails per pair", min_value=2, max_value=20, value=5, key="run_min_pair")
        k_min = p_col2.number_input("K min", min_value=2, max_value=10, value=3, key="run_k_min")
        k_max = p_col3.number_input("K max", min_value=3, max_value=15, value=8, key="run_k_max")
        dbscan_min = p_col4.number_input("DBSCAN min", min_value=2, max_value=20, value=5, key="run_dbscan")

        st.markdown("---")

        # --- RUN BUTTON ---
        run_clicked = st.button(
            "🚀 Run on all emails",
            type="primary",
            use_container_width=True,
        )

        if run_clicked:
            # Save config
            run_config = {
                "batch_strategy": "full",
                "email_sample_size": 0,
                "label_sample_size": 0,
                "min_emails_per_pair": min_pair,
                "k_min": k_min,
                "k_max": k_max,
                "dbscan_eps_values": [0.5, 0.75, 1.0, 1.5, 2.0],
                "dbscan_min_samples": dbscan_min,
            }
            config_path = RESULTS_PATH / "pipeline_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(run_config, f, indent=2)

            # Clear old progress
            prog_path = RESULTS_PATH / "pipeline_progress.json"
            if prog_path.exists():
                prog_path.unlink()

            import subprocess, os
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            process = subprocess.Popen(
                [sys.executable, "-u", "main.py", "--config", str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).parent),
                env=env,
            )

            output_lines = []

            STAGE_LABELS = {
                "[1/6]": ("Loading emails...", 0),
                "[2/6]": ("Network analysis...", 1),
                "[3/6]": ("Scoring emails (predict mode)...", 2),
                "[4/6]": ("Building pair features...", 3),
                "[5/6]": ("Clustering...", 4),
                "[6/6]": ("Naming clusters...", 5),
            }

            with st.status("Running on all emails...", expanded=True) as status:
                progress_bar = st.progress(0, text="Starting...")
                stage_indicator = st.empty()
                log_placeholder = st.empty()

                for line in iter(process.stdout.readline, ""):
                    stripped = line.rstrip()
                    if not stripped:
                        continue
                    output_lines.append(stripped)

                    for marker, (label, stage_num) in STAGE_LABELS.items():
                        if marker in stripped:
                            pct = int((stage_num / 6) * 100)
                            progress_bar.progress(pct, text=f"Step {stage_num + 1}/6 — {label}")
                            status.update(label=label)
                            break

                    if "Scoring" in stripped or "Loading" in stripped or "Computing" in stripped:
                        stage_indicator.caption(f"  ↳ {stripped.strip()}")

                    display_text = "\n".join(output_lines[-20:])
                    log_placeholder.code(display_text, language=None)

                process.wait()

                if process.returncode == 0:
                    progress_bar.progress(100, text="All emails processed!")
                    status.update(label="Run complete!", state="complete", expanded=False)
                    st.success("Done! All emails scored and clustered. Check the other pages for results.")
                    st.balloons()
                else:
                    progress_bar.progress(100, text="Failed")
                    status.update(label="Run failed", state="error")
                    st.error(f"Failed with exit code {process.returncode}.")

            with st.expander("Full log", expanded=True):
                st.code("\n".join(output_lines), language=None)


# ================================================================
# PAGE 8 — Ask the Data (Claude Chat)
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
        "Which pairs show the highest self-disclosure?",
        "Which relationships had the most stable communication over time?",
        "What does the data tell us about Ken Lay's social network?",
        "Which pairs have high dispersion — relationships that bridge different social circles?",
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
