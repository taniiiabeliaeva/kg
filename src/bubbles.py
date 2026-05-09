# bubbles.py — logic-based bubble detection
#
# the core idea is similar to datalog rules:
#   internal_citation(P1, P2) :- cites(P1, P2), affiliated(P1, I), affiliated(P2, I)
#   external_citation(P1, P2) :- cites(P1, P2), affiliated(P1, I), affiliated(P2, J), I != J
#   bubble_score(I) = count(internal) / (count(internal) + count(external))
#
# here we run this as cypher queries against neo4j instead of a datalog engine

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

            # count citations between papers of the same institution (internal)
            r_internal = s.run(
                """
                MATCH (p1:Paper)-[:AFFILIATED_WITH]->(i:Institution {id: $iid}),
                      (p2:Paper)-[:AFFILIATED_WITH]->(i),
                      (p1)-[:CITES]->(p2)
                WHERE p1.id <> p2.id
                RETURN count(*) AS cnt
                """,
                iid=inst_id
            )
            internal = r_internal.single()["cnt"]

            # count citations going out to a different institution (external)
            r_external = s.run(
                """
                MATCH (p1:Paper)-[:AFFILIATED_WITH]->(i:Institution {id: $iid}),
                      (p2:Paper)-[:AFFILIATED_WITH]->(j:Institution),
                      (p1)-[:CITES]->(p2)
                WHERE i.id <> j.id
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

    # save report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/bubble_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nbubble scores:")
    for inst, d in sorted(results.items(), key=lambda x: -x[1]["bubble_score"]):
        print(f"  {inst}: {d['bubble_score']} (internal={d['internal_citations']}, external={d['external_citations']})")

    return results


def compute_temporal_scores():
    # same logic but split by year — to see how bubbles evolve over time
    driver = get_driver()
    rows = []

    with driver.session() as s:
        for inst_name, inst_id in INSTITUTIONS.items():
            r = s.run(
                """
                MATCH (p1:Paper)-[:AFFILIATED_WITH]->(i:Institution {id: $iid}),
                      (p2:Paper)-[:AFFILIATED_WITH]->(j:Institution),
                      (p1)-[:CITES]->(p2)
                WHERE p1.year IS NOT NULL
                RETURN p1.year AS year,
                       i.id = j.id AS is_internal,
                       count(*) AS cnt
                ORDER BY year
                """,
                iid=inst_id
            )
            year_data = {}
            for record in r:
                year = record["year"]
                if year not in year_data:
                    year_data[year] = {"internal": 0, "external": 0}
                if record["is_internal"]:
                    year_data[year]["internal"] += record["cnt"]
                else:
                    year_data[year]["external"] += record["cnt"]

            for year, counts in sorted(year_data.items()):
                total = counts["internal"] + counts["external"]
                score = round(counts["internal"] / total, 4) if total > 0 else 0.0
                rows.append({"institution": inst_name, "year": year, "bubble_score": score})

    driver.close()

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/temporal_bubble_scores.csv", index=False)
    print(f"temporal scores saved to {OUTPUT_DIR}/temporal_bubble_scores.csv")
    return df
