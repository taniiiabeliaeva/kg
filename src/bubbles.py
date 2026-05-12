# logic-based bubble detection
# datalog-style rules (revised):
#   known_paper(P)               :- affiliated_with(P, _)   -- paper is in our dataset
#   internal_citation(P1, P2, I) :- cites(P1, P2), affiliated_with(P1, I), affiliated_with(P2, I)
#   traced_citation(P1, P2, I)   :- cites(P1, P2), affiliated_with(P1, I), known_paper(P2)
#   bubble_score(I)              :- count(internal) / count(traced)
#
# key difference from naive approach:
#   - i only count citations where the cited paper is also in our dataset (known_paper)
#   - this avoids drowning the score with thousands of stub references to unknown papers
#   - result: "of citations we can trace, what fraction stay within the same institution?"

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

            # internal: p1 cites p2, both affiliated with this institution
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

            # traced external: p1 cites p2 (known paper), p2 is from a different institution
            r_external = s.run(
                """
                MATCH (p1:Paper)-[:AFFILIATED_WITH]->(i:Institution {id: $iid}),
                      (p1)-[:CITES]->(p2:Paper),
                      (p2)-[:AFFILIATED_WITH]->(j:Institution)
                WHERE i.id <> j.id
                RETURN count(*) AS cnt
                """,
                iid=inst_id
            )
            external = r_external.single()["cnt"]

            # also count total outgoing (including unknown stubs) for context
            r_total_out = s.run(
                """
                MATCH (p1:Paper)-[:AFFILIATED_WITH]->(i:Institution {id: $iid}),
                      (p1)-[:CITES]->(p2:Paper)
                RETURN count(*) AS cnt
                """,
                iid=inst_id
            )
            total_outgoing = r_total_out.single()["cnt"]

            traced = internal + external
            score = round(internal / traced, 4) if traced > 0 else 0.0

            results[inst_name] = {
                "internal_citations": internal,
                "external_citations": external,
                "traced_citations": traced,
                "total_outgoing": total_outgoing,
                "untraced_citations": total_outgoing - traced,
                "bubble_score": score,
            }

    driver.close()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/bubble_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nbubble scores (internal / traced citations):")
    for inst, d in sorted(results.items(), key=lambda x: -x[1]["bubble_score"]):
        print(f"  {inst}: {d['bubble_score']} "
              f"(internal={d['internal_citations']}, "
              f"external={d['external_citations']}, "
              f"traced={d['traced_citations']}, "
              f"untraced={d['untraced_citations']})")

    return results


def compute_temporal_scores():
    driver = get_driver()
    rows = []

    with driver.session() as s:
        for inst_name, inst_id in INSTITUTIONS.items():

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

            r_ext = s.run(
                """
                MATCH (p1:Paper)-[:AFFILIATED_WITH]->(i:Institution {id: $iid}),
                      (p1)-[:CITES]->(p2:Paper),
                      (p2)-[:AFFILIATED_WITH]->(j:Institution)
                WHERE p1.year IS NOT NULL AND i.id <> j.id
                RETURN p1.year AS year, count(*) AS cnt
                ORDER BY year
                """,
                iid=inst_id
            )
            for rec in r_ext:
                year_data.setdefault(rec["year"], {"internal": 0, "external": 0})
                year_data[rec["year"]]["external"] += rec["cnt"]

            for year, counts in sorted(year_data.items()):
                traced = counts["internal"] + counts["external"]
                score = round(counts["internal"] / traced, 4) if traced > 0 else 0.0
                rows.append({"institution": inst_name, "year": year, "bubble_score": score})

    driver.close()

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/temporal_bubble_scores.csv", index=False)
    print(f"temporal scores saved to {OUTPUT_DIR}/temporal_bubble_scores.csv")
    return df