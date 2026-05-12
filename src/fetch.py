# pulls papers, authors, concepts from openalex and caches locally

import requests
import json
import time
import os
from src.config import OPENALEX_API_KEY, INSTITUTIONS, FIELD_CONCEPT_ID, YEAR_START, YEAR_END, MAX_WORKS_PER_INSTITUTION

CACHE_PATH = "data/works_cache.json"


def _get(url, params=None):
    if params is None:
        params = {}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    time.sleep(0.12)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_works_for_institution(inst_id, concept_id, year_start, year_end, max_works):
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
        # sort by publication_year to get a representative spread, not just famous papers
        "sort": "publication_year:desc",
        "per_page": 100,
    }

    works, page = [], 1
    while len(works) < max_works:
        params["page"] = page
        data = _get(url, params)
        batch = data.get("results", [])
        if not batch:
            break
        works.extend(batch)
        page += 1
        if len(works) >= data["meta"]["count"]:
            break

    return works[:max_works]


def fetch_all(use_cache=True):
    if use_cache and os.path.exists(CACHE_PATH):
        print("loading from cache...")
        with open(CACHE_PATH) as f:
            return json.load(f)

    os.makedirs("data", exist_ok=True)
    all_works = {}

    for name, inst_id in INSTITUTIONS.items():
        print(f"fetching {name}...")
        works = fetch_works_for_institution(
            inst_id, FIELD_CONCEPT_ID, YEAR_START, YEAR_END, MAX_WORKS_PER_INSTITUTION
        )
        all_works[name] = works
        print(f"  {len(works)} works")

    with open(CACHE_PATH, "w") as f:
        json.dump(all_works, f)
    print(f"cached to {CACHE_PATH}")

    return all_works