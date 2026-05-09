
## Project structure

```
research_bubbles/
├── .env                  
├── .gitignore
├── requirements.txt
├── main.py               ← runs the full pipeline
├── explore.py            ← start here to look at the api
├── data/
│   └── works_cache.json  ← auto-generated after first fetch
├── src/
│   ├── config.py         ← loads settings from .env
│   ├── fetch.py          ← openalex api calls
│   ├── graph.py          ← builds the kg in neo4j
│   ├── bubbles.py        ← bubble score logic (cypher queries)
│   └── export.py         ← exports triples for kge
├── models/
│   └── kge_training.py   ← trains TransE / ComplEx / RotatE
├── output/               ← auto-generated
└── visualization/
    └── plot_bubbles.py   ← charts
```

---

## Step 1 — Create virtual environment (Mac)

```bash
cd path/to/research_bubbles
python3 -m venv .venv
source .venv/bin/activate
# (.venv) should appear in your terminal prompt
```

---

## Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

`torch` and `sentence-transformers` are large — this will take a few minutes, normal.

---

## Step 3 — Set up your .env file

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

Then open `.env` and set:
- `OPENALEX_API_KEY` — free at https://openalex.org/settings/api
- `NEO4J_PASSWORD` — you set this when installing neo4j (see step 4)

---

## Step 4 — Install and start Neo4j

1. Download **Neo4j Desktop** from https://neo4j.com/download/
2. Create a new project → Add → Local DBMS
3. Set a password (use the same one in your `.env`)
4. Click **Start**

The default connection `bolt://localhost:7687` should work as-is.

---


## Step 5 — Run the full pipeline

```bash
python main.py
```

This runs in order:
1. Fetches papers from OpenAlex (cached after first run)
2. Loads everything into Neo4j
3. Computes bubble scores using Cypher logic rules
4. Exports triples to `output/triples.tsv`


## Step 7 — Train KGE model

```bash
python models/kge_training.py
```

Change the model in the file:
```python
MODEL = "TransE"   # or "ComplEx", "RotatE"
```

---

## Step 8 — Visualize

```bash
python visualization/plot_bubbles.py
```
