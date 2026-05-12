import pandas as pd
import json
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"


def plot_scores():
    with open(f"{OUTPUT_DIR}/bubble_report.json") as f:
        details = json.load(f)

    institutions = list(details.keys())
    scores = [details[i]["bubble_score"] for i in institutions]
    colors = ["#d62728" if s > 0.5 else "#1f77b4" for s in scores]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(institutions, scores, color=colors)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="threshold (0.5)")
    ax.set_xlabel("bubble score  (internal / total citations)")
    ax.set_title("research bubble score by institution")
    ax.set_xlim(0, 1)
    ax.legend()

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/bubble_scores.png", dpi=150)
    plt.show()


def plot_temporal():
    df = pd.read_csv(f"{OUTPUT_DIR}/temporal_bubble_scores.csv")
    if df.empty:
        print("no temporal data found")
        return

    pivot = df.pivot(index="year", columns="institution", values="bubble_score")

    fig, ax = plt.subplots(figsize=(9, 5))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", label=col)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("year")
    ax.set_ylabel("bubble score")
    ax.set_title("bubble score over time")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/temporal_bubble_scores.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    plot_scores()
    plot_temporal()
