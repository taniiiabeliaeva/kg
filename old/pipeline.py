"""
Research Bubble Detection Pipeline — OpenAlex + Knowledge Graph
================================================================
Steps:
  1. Fetch papers, authors, institutions from OpenAlex API
  2. Build a Knowledge Graph (NetworkX) with typed edges
  3. Detect research bubbles using logic rules (recursive reachability)
  4. Export KG to PyKEEN-compatible triples for KGE training
  5. Basic bubble score visualization

"""

import requests
import networkx as nx
import pandas as pd
import json
import time
import os
from collections import defaultdict

# ──────────────────────────────────────────────
# CONFIG — edit these before running
# ──────────────────────────────────────────────

API_KEY = "YOUR_OPENALEX_API_KEY"   # get free key at openalex.org/settings/api

INSTITUTIONS = {
    "TU Wien":    "I57206974",       # OpenAlex institution IDs
    "ETH Zurich": "I114027177",
    "TU Berlin":  "I63966007",
}

# Filter by field (OpenAlex concept ID) — "Computer Science"
FIELD_CONCEPT_ID = "C41008148"

YEAR_START = 2018
YEAR_END   = 2023
MAX_WORKS_PER_INSTITUTION = 200   # keeping small to start; increase later

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# 1. FETCH FROM OPENALEX API
# ──────────────────────────────────────────────

def openalex_get(url, params=None):
    """Thin wrapper with polite rate limiting and auth."""
    if params is None:
        params = {}
    params["api_key"] = API_KEY
    params.setdefault("per_page", 100)
    time.sleep(0.12)   # ~8 req/s — well within free tier limits
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_works_for_institution(inst_id, concept_id, year_start, year_end, max_works):
    """Fetch papers from a single institution within a field and time range."""
    url = "https://api.openalex.org/works"
    filter_str = (
        f"authorships.institutions.id:{inst_id},"
        f"concepts.id:{concept_id},"
        f"publication_year:{year_start}-{year_end},"
        f"type:article"
    )
    params = {
        "filter": filter_str,
        "select": "id,title,publication_year,authorships,concepts,referenced_works",
        "sort": "cited_by_count:desc",
    }

    works = []
    page = 1
    while len(works) < max_works:
        params["page"] = page
        data = openalex_get(url, params)
        batch = data.get("results", [])
        if not batch:
            break
        works.extend(batch)
        page += 1
        if len(works) >= data["meta"]["count"]:
            break

    return works[:max_works]


def fetch_all_works(institutions, concept_id, year_start, year_end, max_per_inst):
    """Fetch works for all institutions and cache to disk."""
    cache_path = os.path.join(OUTPUT_DIR, "works_cache.json")
    if os.path.exists(cache_path):
        print("Loading works from cache...")
        with open(cache_path) as f:
            return json.load(f)

    all_works = {}
    for name, inst_id in institutions.items():
        print(f"Fetching works for {name}...")
        works = fetch_works_for_institution(inst_id, concept_id, year_start, year_end, max_per_inst)
        all_works[name] = works
        print(f"  → {len(works)} works fetched")

    with open(cache_path, "w") as f:
        json.dump(all_works, f)
    print(f"Cached to {cache_path}")
    return all_works


# ──────────────────────────────────────────────
# 2. BUILD KNOWLEDGE GRAPH
# ──────────────────────────────────────────────

def build_kg(all_works, institutions):
    """
    Build a directed multigraph with typed edges:
      - (author)       --authored_by-->  (paper)
      - (paper)        --affiliated_with--> (institution)
      - (paper)        --cites-->        (paper)
      - (paper)        --has_concept-->  (concept)

    Node types: paper, author, institution, concept
    """
    G = nx.DiGraph()
    work_to_inst = {}   # work_id -> institution name (for bubble scoring later)

    for inst_name, works in all_works.items():
        inst_id = institutions[inst_name]

        # Add institution node
        G.add_node(inst_id, type="institution", label=inst_name)

        for work in works:
            wid = work["id"].split("/")[-1]   # shorten e.g. W2345678
            work_to_inst[wid] = inst_name

            # Paper node
            G.add_node(wid, type="paper", label=work.get("title", "")[:60],
                       year=work.get("publication_year"))

            # Paper → Institution
            G.add_edge(wid, inst_id, rel="affiliated_with")

            # Author → Paper
            for authorship in work.get("authorships", []):
                author = authorship.get("author")
                if not author:
                    continue
                aid = author["id"].split("/")[-1]
                aname = author.get("display_name", "")
                G.add_node(aid, type="author", label=aname)
                G.add_edge(aid, wid, rel="authored")

            # Paper → Cited papers (citations)
            for ref in work.get("referenced_works", []):
                ref_id = ref.split("/")[-1]
                if not G.has_node(ref_id):
                    G.add_node(ref_id, type="paper", label="")
                G.add_edge(wid, ref_id, rel="cites")

            # Paper → Concepts
            for concept in work.get("concepts", [])[:3]:   # top 3 concepts only
                cid = concept["id"].split("/")[-1]
                cname = concept.get("display_name", "")
                G.add_node(cid, type="concept", label=cname)
                G.add_edge(wid, cid, rel="has_concept")

    print(f"\nKG built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, work_to_inst


# ──────────────────────────────────────────────
# 3. LOGIC-BASED BUBBLE DETECTION
# ──────────────────────────────────────────────

def get_institution_papers(all_works):
    """Map institution name → set of paper IDs."""
    inst_papers = {}
    for inst_name, works in all_works.items():
        inst_papers[inst_name] = {w["id"].split("/")[-1] for w in works}
    return inst_papers


def compute_bubble_scores(G, inst_papers):
    """
    Logic rules for bubble detection (mimicking Datalog-style reasoning):

    Rule 1 — internal_citation(P1, P2) :-
        cites(P1, P2), institution(P1, I), institution(P2, I).

    Rule 2 — external_citation(P1, P2) :-
        cites(P1, P2), institution(P1, I), institution(P2, J), I != J.

    Bubble Score = internal_citations / (internal + external_citations)
    A score close to 1.0 means a bubble; close to 0 means open community.
    """
    # Build a reverse index: paper_id → institution
    paper_to_inst = {}
    for inst_name, papers in inst_papers.items():
        for pid in papers:
            paper_to_inst[pid] = inst_name

    scores = {}
    details = {}

    for inst_name, papers in inst_papers.items():
        internal = 0
        external = 0
        internal_pairs = []
        external_pairs = []

        for pid in papers:
            if not G.has_node(pid):
                continue
            for _, cited_id, data in G.out_edges(pid, data=True):
                if data.get("rel") != "cites":
                    continue
                cited_inst = paper_to_inst.get(cited_id)
                if cited_inst == inst_name:
                    internal += 1
                    internal_pairs.append((pid, cited_id))
                elif cited_inst is not None:
                    external += 1
                    external_pairs.append((pid, cited_id, cited_inst))

        total = internal + external
        score = internal / total if total > 0 else 0.0
        scores[inst_name] = score
        details[inst_name] = {
            "internal_citations": internal,
            "external_citations": external,
            "total_citations": total,
            "bubble_score": round(score, 4),
            "sample_internal": internal_pairs[:5],
            "sample_external": external_pairs[:5],
        }

    return scores, details


def print_bubble_report(details):
    print("\n" + "="*55)
    print("BUBBLE SCORE REPORT")
    print("="*55)
    for inst, d in sorted(details.items(), key=lambda x: -x[1]["bubble_score"]):
        print(f"\n{inst}")
        print(f"  Bubble score:      {d['bubble_score']:.3f}  (1.0 = full bubble)")
        print(f"  Internal cites:    {d['internal_citations']}")
        print(f"  External cites:    {d['external_citations']}")
        print(f"  Total cites:       {d['total_citations']}")
    print("="*55)


# ──────────────────────────────────────────────
# 4. EXPORT TRIPLES FOR KGE TRAINING
# ──────────────────────────────────────────────

def export_triples(G, output_dir):
    """
    Export KG as (head, relation, tail) triples — standard input format
    for KGE libraries (PyKEEN, AmpliGraph, etc.)
    """
    triples = []
    for src, dst, data in G.edges(data=True):
        rel = data.get("rel", "related")
        triples.append((str(src), rel, str(dst)))

    df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    path = os.path.join(output_dir, "triples.tsv")
    df.to_csv(path, sep="\t", index=False)
    print(f"\nExported {len(triples)} triples to {path}")

    # Also save relation type distribution
    print("\nRelation distribution:")
    print(df["relation"].value_counts().to_string())
    return df


# ──────────────────────────────────────────────
# 5. TEMPORAL SLICE ANALYSIS (bonus)
# ──────────────────────────────────────────────

def temporal_bubble_scores(all_works, institutions, years):
    """
    Compute bubble score per institution per year.
    Useful for visualizing whether bubbles grow or dissolve over time.
    """
    results = []
    for year in years:
        year_works = {
            inst: [w for w in works if w.get("publication_year") == year]
            for inst, works in all_works.items()
        }
        if all(len(v) == 0 for v in year_works.values()):
            continue
        G_year, _ = build_kg(year_works, institutions)
        inst_papers_year = get_institution_papers(year_works)
        scores, _ = compute_bubble_scores(G_year, inst_papers_year)
        for inst, score in scores.items():
            results.append({"year": year, "institution": inst, "bubble_score": score})

    df = pd.DataFrame(results)
    path = os.path.join(OUTPUT_DIR, "temporal_bubble_scores.csv")
    df.to_csv(path, index=False)
    print(f"\nTemporal scores saved to {path}")
    return df


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1 — Fetch
    all_works = fetch_all_works(
        INSTITUTIONS, FIELD_CONCEPT_ID, YEAR_START, YEAR_END, MAX_WORKS_PER_INSTITUTION
    )

    # Step 2 — Build KG
    G, work_to_inst = build_kg(all_works, INSTITUTIONS)

    # Step 3 — Bubble detection (logic rules)
    inst_papers = get_institution_papers(all_works)
    scores, details = compute_bubble_scores(G, inst_papers)
    print_bubble_report(details)

    # Save detailed report
    with open(os.path.join(OUTPUT_DIR, "bubble_report.json"), "w") as f:
        json.dump(details, f, indent=2, default=str)

    # Step 4 — Export triples for KGE
    triples_df = export_triples(G, OUTPUT_DIR)

    # Step 5 — Temporal analysis
    years = list(range(YEAR_START, YEAR_END + 1))
    temporal_df = temporal_bubble_scores(all_works, INSTITUTIONS, years)
    print("\nTemporal bubble scores:")
    print(temporal_df.pivot(index="year", columns="institution", values="bubble_score").to_string())

    print("\nDone! Next step: load triples.tsv into PyKEEN or AmpliGraph for KGE training.")
    print("See kge_training.py for a starter script.")
