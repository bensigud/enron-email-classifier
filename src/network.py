import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# --- Person class thresholds (tuned after seeing data) ---
TOP_HUB_PERCENTILE = 0.90          # top 10% by unique contacts = Hub
HIGH_BETWEENNESS_PERCENTILE = 0.85  # top 15% by betweenness = Gatekeeper
HIGH_CLUSTERING_PERCENTILE = 0.75   # top 25% clustering + low contacts = Inner Circle
ISOLATED_PERCENTILE = 0.10          # bottom 10% by contacts = Isolated


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed graph from the emails DataFrame.
    Each node is a person (email address).
    Each edge A → B means A sent at least one email to B.
    Edge weight = number of emails sent from A to B.
    """
    graph = nx.DiGraph()

    for _, row in df.iterrows():
        sender = row["sender"]
        recipients = row["recipients"]

        if not sender or not recipients:
            continue

        for recipient in recipients:
            if recipient == sender:
                continue  # skip self-emails
            if graph.has_edge(sender, recipient):
                graph[sender][recipient]["weight"] += 1
            else:
                graph.add_edge(sender, recipient, weight=1)

    print(f"Network built: {graph.number_of_nodes()} people, {graph.number_of_edges()} connections")
    return graph


def compute_network_features(graph: nx.DiGraph) -> pd.DataFrame:
    """
    Compute behavioural features for each person in the network.
    These features are used to classify each person.

    Returns a DataFrame with one row per person.
    """
    undirected = graph.to_undirected()

    # Number of unique people this person emailed (out-degree)
    out_degree = dict(graph.out_degree())

    # Number of unique people who emailed this person (in-degree)
    in_degree = dict(graph.in_degree())

    # Betweenness centrality: how often does this person sit on the
    # shortest path between two others? High = Gatekeeper
    betweenness = nx.betweenness_centrality(undirected, normalized=True)

    # Clustering coefficient: how tightly connected are this person's
    # contacts to each other? High = tight inner circle
    clustering = nx.clustering(undirected)

    # PageRank: similar to Google's algorithm — who is emailed by
    # important people?
    pagerank = nx.pagerank(graph, alpha=0.85)

    people = list(graph.nodes())
    records = []
    for person in people:
        records.append({
            "person": person,
            "out_degree": out_degree.get(person, 0),
            "in_degree": in_degree.get(person, 0),
            "total_degree": out_degree.get(person, 0) + in_degree.get(person, 0),
            "betweenness": betweenness.get(person, 0),
            "clustering": clustering.get(person, 0),
            "pagerank": pagerank.get(person, 0),
        })

    return pd.DataFrame(records)


def classify_person(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a person class to each person based on their network features.

    Classes (in priority order):
    - Hub: top 10% by total unique contacts
    - Gatekeeper: top 15% by betweenness centrality
    - Inner Circle: high clustering + relatively low out-degree
    - Isolated: bottom 10% by total contacts
    - Follower: everyone else
    """
    df = features_df.copy()

    hub_threshold = df["total_degree"].quantile(TOP_HUB_PERCENTILE)
    betweenness_threshold = df["betweenness"].quantile(HIGH_BETWEENNESS_PERCENTILE)
    clustering_threshold = df["clustering"].quantile(HIGH_CLUSTERING_PERCENTILE)
    isolated_threshold = df["total_degree"].quantile(ISOLATED_PERCENTILE)

    def assign_class(row):
        if row["total_degree"] >= hub_threshold:
            return "Hub"
        elif row["betweenness"] >= betweenness_threshold:
            return "Gatekeeper"
        elif (row["clustering"] >= clustering_threshold and
              row["out_degree"] < df["out_degree"].median()):
            return "Inner Circle"
        elif row["total_degree"] <= isolated_threshold:
            return "Isolated"
        else:
            return "Follower"

    df["person_class"] = df.apply(assign_class, axis=1)
    return df


def detect_communities(graph: nx.DiGraph) -> dict:
    """
    Detect communities (clusters of people who email each other a lot).
    Returns a dict mapping person → community_id.
    """
    undirected = graph.to_undirected()
    communities = nx.community.greedy_modularity_communities(undirected)

    person_to_community = {}
    for i, community in enumerate(communities):
        for person in community:
            person_to_community[person] = i

    print(f"Detected {len(communities)} communities")
    return person_to_community


def plot_network(graph: nx.DiGraph, features_df: pd.DataFrame,
                 output_path: str, max_nodes: int = 150):
    """
    Draw the social network graph and save it as an image.
    Node size = total degree (more connected = bigger)
    Node colour = person class
    Edge thickness = email volume between pair
    """
    CLASS_COLORS = {
        "Hub": "#e74c3c",          # red
        "Gatekeeper": "#e67e22",   # orange
        "Inner Circle": "#9b59b6", # purple
        "Follower": "#3498db",     # blue
        "Isolated": "#95a5a6",     # grey
    }

    # Limit to most connected nodes for readability
    top_nodes = (features_df
                 .nlargest(max_nodes, "total_degree")["person"]
                 .tolist())
    subgraph = graph.subgraph(top_nodes)

    # Map features to nodes
    feat = features_df.set_index("person")
    node_colors = [CLASS_COLORS.get(feat.loc[n, "person_class"], "#aaa")
                   if n in feat.index else "#aaa"
                   for n in subgraph.nodes()]
    node_sizes = [max(50, feat.loc[n, "total_degree"] * 3)
                  if n in feat.index else 50
                  for n in subgraph.nodes()]

    fig, ax = plt.subplots(figsize=(20, 14))
    pos = nx.spring_layout(subgraph, seed=42, k=0.5)

    nx.draw_networkx_edges(subgraph, pos, alpha=0.15,
                           width=0.5, arrows=False, ax=ax)
    nx.draw_networkx_nodes(subgraph, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.85, ax=ax)

    # Only label the most important nodes
    top_labels = features_df.nlargest(20, "total_degree")["person"].tolist()
    labels = {n: n.split("@")[0] for n in subgraph.nodes() if n in top_labels}
    nx.draw_networkx_labels(subgraph, pos, labels,
                            font_size=7, font_color="black", ax=ax)

    # Legend
    legend_elements = [
        plt.scatter([], [], c=color, s=100, label=cls)
        for cls, color in CLASS_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
    ax.set_title("Enron Social Network — Who Emailed Who", fontsize=16)
    ax.axis("off")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Network plot saved to {output_path}")


def save_results(features_df: pd.DataFrame, communities: dict,
                 graph: nx.DiGraph, output_dir: str):
    """Save all network analysis results to data/results/."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    features_df.to_csv(out / "person_classes.csv", index=False)
    with open(out / "communities.json", "w") as f:
        json.dump(communities, f)

    print(f"Network results saved to {output_dir}")


def run_network_analysis(df: pd.DataFrame, output_dir: str = "data/results"):
    """
    Full pipeline: emails → graph → features → classes → plot → save.
    Call this from main.py.
    """
    print("\n=== Analysis 1 & 2: Social Network ===")
    graph = build_graph(df)
    features_df = compute_network_features(graph)
    features_df = classify_person(features_df)
    communities = detect_communities(graph)

    features_df["community"] = features_df["person"].map(communities)

    plot_network(graph, features_df,
                 output_path=f"{output_dir}/network_plot.png")
    save_results(features_df, communities, graph, output_dir)

    # Print summary
    print("\nPerson class distribution:")
    print(features_df["person_class"].value_counts().to_string())
    print(f"\nTop 10 most connected people:")
    print(features_df.nlargest(10, "total_degree")[
        ["person", "person_class", "total_degree", "betweenness"]
    ].to_string(index=False))

    return graph, features_df
