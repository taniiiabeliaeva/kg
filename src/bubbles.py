# bubbles.py — logic-based bubble detection
#
# datalog-style rules:
#   inst_paper(P, I)            :- affiliated_with(P, I)
#   inst_paper(P, I)            :- authored(A, P), affiliated_with(A_paper, I), authored(A, A_paper)
#                                  -- fallback: infer institution via author
#
#   internal_citation(P1, P2, I) :- cites(P1, P2), inst_paper(P1, I), inst_paper(P2, I)
#   external_citation(P1, P2, I) :- cites(P1, P2), inst_paper(P1, I), NOT inst_paper(P2, I)
#   bubble_score(I)              :- count(internal) / (count(internal) + count(external))
#
# the key fix vs previous version: external now means "cited paper has no affiliation
# to this institution" — i.e. we count all outgoing citations, not just ones where
# the cited paper happens to also be in our dataset with an affiliation tag

import json
import os
import pandas as pd
from src.graph import get_driver
from src.config import INSTITUTIONS

OUTPUT_DIR = "output"


def compute_bubble_scores():
    driver = get_driver()
    results = {}

    with driver.session() as s:
        for inst_name, inst_id in INSTITUTIONS.items():

            # internal: p1 (from this inst) cites p2 (also from this inst)
            # p2's institution is confirmed via AFFILIATED_WITH edge
            r_internal = s.run(
                """
                MATCH (p1:Paper)-[:AFFILIATED_WITH]->(i:Institution {id: $iid}),
                      (p1)-[:CITES]->(p2:Paper),
                      (p2)-[:AFFILIATED_WITH]->(i)
                WHERE p1.id <> p2.id
                RETURN count(*) AS cnt
                """,
                iid=inst_id
            )
            internal = r_internal.single()["cnt"]

            # external: p1 (from this inst) cites p2 that has NO affiliation to this inst
            # this now includes all the stub papers that were only fetched as references
            r_external = s.run(
                """
                MATCH (p1:Paper)-[:AFFILIATED_WITH]->(i:Institution {id: $iid}),
                      (p1)-[:CITES]->(p2:Paper)
                WHERE NOT (p2)-[:AFFILIATED_WITH]->(i)
                RETURN count(*) AS cnt
                """,
                iid=inst_id
            )
            external = r_external.single()["cnt"]

            total = internal + external
            score = round(internal / total, 4) if total > 0 else 0.0

            results[inst_name] = {
                "internal_citations": internal,
                "external_citations": external,
                "total_citations": total,
                "bubble_score": score,
            }

    driver.close()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/bubble_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nbubble scores:")
    for inst, d in sorted(results.items(), key=lambda x: -x[1]["bubble_score"]):
        print(f"  {inst}: {d['bubble_score']} (internal={d['internal_citations']}, external={d['external_citations']}, total={d['total_citations']})")

    return results


def compute_temporal_scores():
    driver = get_driver()
    rows = []

    with driver.session() as s:
        for inst_name, inst_id in INSTITUTIONS.items():

            # internal per year
            r_int = s.run(
                """
                MATCH (p1:Paper)-[:AFFILIATED_WITH]->(i:Institution {id: $iid}),
                      (p1)-[:CITES]->(p2:Paper),
                      (p2)-[:AFFILIATED_WITH]->(i)
                WHERE p1.year IS NOT NULL AND p1.id <> p2.id
                RETURN p1.year AS year, count(*) AS cnt
                ORDER BY year
                """,
                iid=inst_id
            )
            year_data = {}
            for rec in r_int:
                year_data.setdefault(rec["year"], {"internal": 0, "external": 0})
                year_data[rec["year"]]["internal"] += rec["cnt"]

            # external per year
            r_ext = s.run(
                """
                MATCH (p1:Paper)-[:AFFILIATED_WITH]->(i:Institution {id: $iid}),
                      (p1)-[:CITES]->(p2:Paper)
                WHERE p1.year IS NOT NULL
                  AND NOT (p2)-[:AFFILIATED_WITH]->(i)
                RETURN p1.year AS year, count(*) AS cnt
                ORDER BY year
                """,
                iid=inst_id
            )
            for rec in r_ext:
                year_data.setdefault(rec["year"], {"internal": 0, "external": 0})
                year_data[rec["year"]]["external"] += rec["cnt"]

            for year, counts in sorted(year_data.items()):
                total = counts["internal"] + counts["external"]
                score = round(counts["internal"] / total, 4) if total > 0 else 0.0
                rows.append({"institution": inst_name, "year": year, "bubble_score": score})

    driver.close()

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/temporal_bubble_scores.csv", index=False)
    print(f"temporal scores saved to {OUTPUT_DIR}/temporal_bubble_scores.csv")
    return df