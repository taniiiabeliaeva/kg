# Research Bubble Detection — KG Project

## Structure

```
pipeline.py      # Step 1–5: fetch → KG → bubble logic → export triples
kge_training.py  # KGE training (TransE/ComplEx) + link prediction
visualize.py     # Bubble score charts + interactive KG viewer
output/
  works_cache.json              # cached API responses (delete to re-fetch)
  triples.tsv                   # KG as (head, relation, tail) — input for KGE
  bubble_report.json            # bubble scores per institution
  temporal_bubble_scores.csv    # bubble score per institution per year
  bubble_scores.png             # bar chart
  temporal_bubble_scores.png    # line chart over time
  kg_subgraph.html              # interactive graph (open in browser)
  kge/                          # trained KGE model files
```

## Quickstart

```bash
# 1. Install dependencies
pip install requests networkx pandas matplotlib pyvis pykeen torch

# 2. Get a free OpenAlex API key at https://openalex.org/settings/api
#    Set it in pipeline.py: API_KEY = "your_key_here"

# 3. Run the pipeline
python pipeline.py

# 4. Train KGE
python kge_training.py

# 5. Visualize
python visualize.py
```

## Key settings to tune (in pipeline.py)

| Setting | Default | Notes |
|---|---|---|
| `MAX_WORKS_PER_INSTITUTION` | 200 | Start small, increase later |
| `FIELD_CONCEPT_ID` | C41008148 (CS) | Find other concept IDs at openalex.org/concepts |
| `YEAR_START / YEAR_END` | 2018–2023 | Narrow range = faster |

## KGE models to try (in kge_training.py)

- `TransE` — simplest, good baseline
- `ComplEx` — handles asymmetric relations (good for citations)
- `RotatE` — generally strong, worth trying

## Bubble Score interpretation

```
score = internal_citations / (internal + external_citations)

~1.0  →  strong bubble (mostly citing own institution)
~0.5  →  mixed
~0.0  →  very open community
```

## Tip (from supervisor feedback)
Keep data ingestion minimal — use the cache after first run,
and focus time on the logic rules and KGE parts.
