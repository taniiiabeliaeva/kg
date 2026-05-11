# Research Bubble Detection in Academic Knowledge Graphs

This project detects research bubbles in academic knowledge graphs by combining logic-based citation analysis and knowledge graph embeddings on OpenAlex data.

## Project structure

```
research_bubbles/
├── .env                       ← your keys (never commit this)
├── .env.example               ← template — safe to commit
├── .gitignore
├── requirements.txt
├── main.py                    ← runs the full pipeline
├── explore.py                 ← start here to look at the api
├── data/
│   └── works_cache.json       ← auto-generated after first fetch
├── src/
│   ├── config.py              ← loads settings from .env
│   ├── fetch.py               ← openalex api calls
│   ├── graph.py               ← builds the kg in neo4j
│   ├── bubbles.py             ← bubble score logic (cypher queries)
│   └── export.py              ← exports triples for kge
├── models/
│   └── kge_training.py        ← trains TransE or ComplEx
├── output/                    ← auto-generated
│   ├── bubble_report.json
│   ├── temporal_bubble_scores.csv
│   ├── triples.tsv
│   └── kge/
│       ├── TransE/
│       └── ComplEx/
└── visualization/
    └── plot_bubbles.py        ← charts
```

---

## Setup

### Step 1 — Create virtual environment (Mac)

```bash
cd path/to/research_bubbles
python3 -m venv .venv
source .venv/bin/activate
```

`(.venv)` should appear in your terminal prompt. You need to run this activation command every time you open a new terminal.

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

`torch` and `sentence-transformers` are large — takes a few minutes, that's normal.

### Step 3 — Set up your .env file

```bash
cp .env.example .env
```

Open `.env` and fill in:
- `OPENALEX_API_KEY` — free key at https://openalex.org/settings/api
- `NEO4J_PASSWORD` — the password you set when creating the Neo4j database (step 4)

### Step 4 — Install and start Neo4j

1. Download **Neo4j Desktop** from https://neo4j.com/download/
2. Create a new project → Add → Local DBMS
3. Set a password (same one as in `.env`)
4. Click **Start**

The default connection `bolt://localhost:7687` works as-is.

---

## Running the project

### Full pipeline

```bash
python main.py
```

Runs in order:
1. Fetches papers from OpenAlex — cached to `data/works_cache.json` after first run
2. Loads the knowledge graph into Neo4j (nodes: Paper, Author, Institution, Concept)
3. Computes bubble scores using Cypher-based logic rules
4. Exports triples to `output/triples.tsv` for KGE training

To re-fetch fresh data, delete the cache first:
```bash
rm data/works_cache.json
python main.py
```

### Train KGE model

```bash
python models/kge_training.py --model TransE
python models/kge_training.py --model ComplEx
```

Models are saved separately to `output/kge/TransE/` and `output/kge/ComplEx/` so results don't overwrite each other. ComplEx handles asymmetric relations better than TransE, which matters for citation graphs (A cites B ≠ B cites A).

### Visualize results

```bash
python visualization/plot_bubbles.py
```

Generates:
- `output/bubble_scores.png` — bar chart comparing institutions
- `output/temporal_bubble_scores.png` — bubble score over time

---

## How bubble score is calculated

```
bubble_score = internal_citations / traced_citations

where:
  internal_citations = citations between papers from the same institution
  traced_citations   = citations where both papers are in the dataset
  (untraced citations to papers outside the dataset are excluded)
```

A score close to 1.0 means the institution mostly cites itself. A score close to 0.0 means it cites broadly across institutions.

---

## Configuration

All settings are in `src/config.py`:

| Setting | Default | Notes |
|---|---|---|
| `MAX_WORKS_PER_INSTITUTION` | 500 | increase for more data, slower fetch |
| `FIELD_CONCEPT_ID` | C41008148 (CS) | find other IDs at openalex.org/concepts |
| `YEAR_START / YEAR_END` | 2015–2025 | adjust time window |