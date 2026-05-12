# exports the kg as triples for kge training
# output format: (head, relation, tail) — works with pykeen, ampligraph, etc.

import pandas as pd
import os
from src.graph import get_driver

OUTPUT_DIR = "output"


def export_triples():
    driver = get_driver()
    triples = []

    with driver.session() as s:
        # get all edges with their types
        r = s.run(
            """
            MATCH (a)-[rel]->(b)
            WHERE a.id IS NOT NULL AND b.id IS NOT NULL
            RETURN a.id AS head, type(rel) AS relation, b.id AS tail
            """
        )
        for record in r:
            triples.append((record["head"], record["relation"], record["tail"]))

    driver.close()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    path = f"{OUTPUT_DIR}/triples.tsv"
    df.to_csv(path, sep="\t", index=False)

    print(f"exported {len(df)} triples to {path}")
    print(df["relation"].value_counts().to_string())

    return df
