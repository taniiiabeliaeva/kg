import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT_DIR = "output"

INST_COLORS = {
    "TU Wien":    "#4C72B0",
    "ETH Zurich": "#DD8452",
    "TU Berlin":  "#55A868",
}

CLUSTER_CENTERS = {
    "TU Wien":    np.array([-2.2,  0.0]),
    "ETH Zurich": np.array([ 2.2,  0.0]),
    "TU Berlin":  np.array([ 0.0,  2.5]),
}


def load_data(triples_path, max_internal_per_inst=35, max_external=20):
    df = pd.read_csv(triples_path, sep="\t")
    cites = df[df["relation"] == "CITES"]
    affil = df[df["relation"] == "AFFILIATED_WITH"][["head", "tail"]].copy()
    affil.columns = ["paper", "inst_id"]

    inst_id_to_name = {
        "I57206974":  "TU Wien",
        "I114027177": "ETH Zurich",
        "I63966007":  "TU Berlin",
    }
    affil["inst"] = affil["inst_id"].map(inst_id_to_name)
    paper_to_inst = dict(zip(affil["paper"], affil["inst"]))

    G = nx.DiGraph()
    internal_counts = {k: 0 for k in INST_COLORS}
    external_count = 0

    for _, row in cites.iterrows():
        h, t = row["head"], row["tail"]
        hi = paper_to_inst.get(h)
        ti = paper_to_inst.get(t)
        if not hi or not ti:
            continue
        if hi == ti and internal_counts[hi] < max_internal_per_inst:
            G.add_node(h, inst=hi)
            G.add_node(t, inst=ti)
            G.add_edge(h, t, etype="internal")
            internal_counts[hi] += 1
        elif hi != ti and external_count < max_external:
            G.add_node(h, inst=hi)
            G.add_node(t, inst=ti)
            G.add_edge(h, t, etype="external")
            external_count += 1

    return G


def make_pos(G):
    rng = np.random.default_rng(42)

    # start: scatter nodes randomly around their cluster center
    init_pos = {}
    for n, d in G.nodes(data=True):
        inst = d.get("inst", "TU Wien")
        center = CLUSTER_CENTERS.get(inst, np.array([0, 0]))
        init_pos[n] = center + rng.normal(0, 0.4, size=2)

    # run spring layout starting from cluster positions
    # low k = tighter clusters, fixed_nodes anchors nothing but seed helps
    pos = nx.spring_layout(
        G,
        pos=init_pos,
        k=0.3,
        iterations=80,
        seed=42,
        weight=None,
    )

    # after spring layout, gently pull each node back toward its cluster center
    # this keeps clusters from drifting into each other
    for n, d in G.nodes(data=True):
        inst = d.get("inst", "TU Wien")
        center = CLUSTER_CENTERS.get(inst, np.array([0, 0]))
        pos[n] = pos[n] * 0.5 + center * 0.5

    return pos


def draw(G):
    fig, ax = plt.subplots(figsize=(12, 9))

    pos = make_pos(G)

    node_colors = [INST_COLORS.get(G.nodes[n].get("inst"), "#aaa") for n in G.nodes]
    internal_edges = [(u, v) for u, v, d in G.edges(data=True) if d["etype"] == "internal"]
    external_edges = [(u, v) for u, v, d in G.edges(data=True) if d["etype"] == "external"]

    # soft cluster halos
    for inst, center in CLUSTER_CENTERS.items():
        nodes_in = [n for n, d in G.nodes(data=True) if d.get("inst") == inst]
        if not nodes_in:
            continue
        pts = np.array([pos[n] for n in nodes_in])
        cx, cy = pts.mean(axis=0)
        spread = max(np.std(pts, axis=0).max() * 1.8, 0.7)
        circle = plt.Circle((cx, cy), spread, color=INST_COLORS[inst],
                             alpha=0.10, zorder=0, linewidth=0)
        ax.add_patch(circle)
        ax.text(cx, cy - spread - 0.1, inst, ha="center", va="top",
                fontsize=11, fontweight="bold", color=INST_COLORS[inst])

    # edges first so nodes render on top
    nx.draw_networkx_edges(G, pos, edgelist=external_edges,
                           edge_color="#aaaaaa", alpha=0.3, arrows=True,
                           arrowsize=7, width=0.6, style="dashed",
                           connectionstyle="arc3,rad=0.15", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=internal_edges,
                           edge_color="#333333", alpha=0.5, arrows=True,
                           arrowsize=9, width=1.0,
                           connectionstyle="arc3,rad=0.1", ax=ax)

    nx.draw_networkx_nodes(G, pos, node_size=70, node_color=node_colors,
                           alpha=0.92, linewidths=0.5, edgecolors="white", ax=ax)

    handles = [mpatches.Patch(color=c, label=inst) for inst, c in INST_COLORS.items()]
    handles += [
        plt.Line2D([0], [0], color="#333333", lw=1.5, label="internal citation"),
        plt.Line2D([0], [0], color="#aaaaaa", lw=1.0, ls="--", label="cross-institution citation"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title("Citation network — papers grouped by institution\n"
                 "dense same-color edges within clusters indicate research bubbles",
                 fontsize=11, pad=12)
    ax.axis("off")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "citation_network.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")
    plt.show()


if __name__ == "__main__":
    G = load_data(f"{OUTPUT_DIR}/triples.tsv", max_internal_per_inst=35, max_external=20)
    print(f"nodes: {G.number_of_nodes()}  edges: {G.number_of_edges()}")
    draw(G)