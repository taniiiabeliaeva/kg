# main.py — runs the full pipeline in order
# you can comment out steps you've already done

from src.fetch import fetch_all
from src.graph import build_graph
from src.bubbles import compute_bubble_scores, compute_temporal_scores
from src.export import export_triples

if __name__ == "__main__":

    # step 1 — fetch from openalex (cached after first run)
    all_works = fetch_all(use_cache=True)

    # step 2 — load into neo4j (set clear=True to wipe and reload)
    build_graph(all_works, clear=False)

    # step 3 — compute bubble scores using logic rules
    compute_bubble_scores()
    compute_temporal_scores()

    # step 4 — export triples for kge training
    export_triples()

    print("\ndone — next run: python models/kge_training.py")
