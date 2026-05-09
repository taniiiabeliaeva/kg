"""
Visualization — Research Bubble Detection
==========================================
1. Bar chart of bubble scores per institution
2. Temporal line chart of bubble score evolution
3. Interactive KG visualization (small subgraph) via pyvis

Requirements:
  pip install matplotlib pandas pyvis networkx
"""

import pandas as pd
import json
import os
import networkx as nx
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"


# ──────────────────────────────────────────────
# 1. BUBBLE SCORE BAR CHART
# ──────────────────────────────────────────────

def plot_bubble_scores(report_path=None, details=None):
    if details is None:
        with open(report_path or os.path.join(OUTPUT_DIR, "bubble_report.json")) as f:
            details = json.load(f)

    institutions = list(details.keys())
    scores = [details[i]["bubble_score"] for i in institutions]
    colors = ["#d62728" if s > 0.5 else "#1f77b4" for s in scores]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(institutions, scores, color=colors)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Bubble threshold (0.5)")
    ax.set_xlabel("Bubble Score  (internal citations / total citations)")
    ax.set_title("Research Bubble Score by Institution")
    ax.set_xlim(0, 1)
    ax.legend()

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bubble_scores.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()


# ──────────────────────────────────────────────
# 2. TEMPORAL LINE CHART
# ──────────────────────────────────────────────

def plot_temporal_scores(csv_path=None):
    path = csv_path or os.path.join(OUTPUT_DIR, "temporal_bubble_scores.csv")
    df = pd.read_csv(path)

    if df.empty:
        print("No temporal data found.")
        return

    pivot = df.pivot(index="year", columns="institution", values="bubble_score")

    fig, ax = plt.subplots(figsize=(9, 5))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", label=col)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Bubble threshold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Bubble Score")
    ax.set_title("Research Bubble Score Over Time")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "temporal_bubble_scores.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()


# ──────────────────────────────────────────────
# 3. INTERACTIVE KG (small subgraph)
# ──────────────────────────────────────────────

def visualize_subgraph(G, institution_id, max_papers=20):
    """
    Show a small subgraph around one institution:
    institution → papers → cited papers / authors
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("pyvis not installed. Run: pip install pyvis")
        return

    # Pick papers belonging to this institution
    papers = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == "paper" and G.has_edge(n, institution_id)
    ][:max_papers]

    sub_nodes = set(papers) | {institution_id}
    for p in papers:
        for neighbor in list(G.predecessors(p)) + list(G.successors(p)):
            sub_nodes.add(neighbor)

    subG = G.subgraph(sub_nodes)

    net = Network(height="700px", width="100%", directed=True, notebook=False)

    color_map = {
        "institution": "#e74c3c",
        "paper":       "#3498db",
        "author":      "#2ecc71",
        "concept":     "#f39c12",
    }

    for node, data in subG.nodes(data=True):
        ntype = data.get("type", "paper")
        label = data.get("label", str(node))[:30]
        net.add_node(str(node), label=label,
                     color=color_map.get(ntype, "#95a5a6"),
                     title=f"{ntype}: {label}")

    for src, dst, data in subG.edges(data=True):
        net.add_edge(str(src), str(dst), label=data.get("rel", ""))

    path = os.path.join(OUTPUT_DIR, f"kg_subgraph.html")
    net.save_graph(path)
    print(f"Saved interactive graph: {path}  (open in browser)")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    plot_bubble_scores()
    plot_temporal_scores()
    # For KG visualization, call visualize_subgraph(G, institution_id)
    # after loading G from pipeline.py
