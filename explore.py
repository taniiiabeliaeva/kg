# explore.py — just poke around the openalex api before committing to anything
# run sections one at a time by commenting out what you don't need

import requests
import time
from src.config import OPENALEX_API_KEY

TU_WIEN    = "I57206974"
ETH_ZURICH = "I114027177"
TU_BERLIN  = "I63966007"


def _get(url, params=None):
    if params is None:
        params = {}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    time.sleep(0.12)
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


# ── part 1: confirm institution ids ──────────────────────────────────────────

def check_institutions():
    for name in ["TU Wien", "ETH Zurich", "TU Berlin"]:
        data = _get("https://api.openalex.org/institutions", {
            "search": name, "per_page": 2,
            "select": "id,display_name,country_code,works_count",
        })
        print(f"\n{name}")
        for r in data["results"]:
            print(f"  {r['id']}  |  {r['display_name']}  |  {r['country_code']}  |  {r['works_count']} works")


# ── part 2: look at a few papers and see what fields come back ────────────────

def look_at_works(inst_id=TU_WIEN, year=2022, n=3):
    data = _get("https://api.openalex.org/works", {
        "filter": f"authorships.institutions.id:{inst_id},publication_year:{year},type:article",
        "per_page": n,
        "sort": "cited_by_count:desc",
    })
    print(f"\ntotal matching: {data['meta']['count']}")
    for work in data["results"]:
        print(f"\n  title:    {work['title']}")
        print(f"  id:       {work['id']}")
        print(f"  year:     {work['publication_year']}")
        print(f"  cited by: {work['cited_by_count']}")
        for a in work.get("authorships", [])[:2]:
            name = a.get("author", {}).get("display_name", "?")
            insts = [i.get("display_name", "?") for i in a.get("institutions", [])]
            print(f"    author: {name}  @  {insts}")
        for c in work.get("concepts", [])[:3]:
            print(f"    concept: {c['display_name']}  ({c['score']:.2f})")
        refs = work.get("referenced_works", [])
        print(f"  references: {len(refs)}  — first few: {refs[:3]}")


