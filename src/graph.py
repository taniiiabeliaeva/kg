# graph.py — builds the knowledge graph in neo4j
# nodes: Paper, Author, Institution, Concept
# edges: CITES, AUTHORED_BY, AFFILIATED_WITH, HAS_CONCEPT

from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, INSTITUTIONS


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def clear_graph(driver):
    # wipe everything — useful when re-running
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    print("graph cleared")


def create_constraints(driver):
    # uniqueness constraints — speeds up merges a lot
    with driver.session() as s:
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Institution) REQUIRE i.id IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE")


def _short_id(url):
    # openalex ids are full urls — just keep the last part e.g. W2345678
    return url.split("/")[-1]


def load_works(driver, all_works):
    with driver.session() as s:
        for inst_name, works in all_works.items():
            inst_id = INSTITUTIONS[inst_name]

            # institution node
            s.run(
                "MERGE (i:Institution {id: $id}) SET i.name = $name",
                id=inst_id, name=inst_name
            )

            for work in works:
                wid = _short_id(work["id"])

                # paper node
                s.run(
                    """
                    MERGE (p:Paper {id: $id})
                    SET p.title = $title, p.year = $year
                    """,
                    id=wid,
                    title=work.get("title", "")[:200],
                    year=work.get("publication_year"),
                )

                # paper → institution
                s.run(
                    """
                    MATCH (p:Paper {id: $pid}), (i:Institution {id: $iid})
                    MERGE (p)-[:AFFILIATED_WITH]->(i)
                    """,
                    pid=wid, iid=inst_id
                )

                # authors
                for authorship in work.get("authorships", []):
                    author = authorship.get("author")
                    if not author or not author.get("id"):
                        continue
                    aid = _short_id(author["id"])
                    s.run(
                        """
                        MERGE (a:Author {id: $id})
                        SET a.name = $name
                        WITH a
                        MATCH (p:Paper {id: $pid})
                        MERGE (a)-[:AUTHORED]->(p)
                        """,
                        id=aid,
                        name=author.get("display_name", ""),
                        pid=wid,
                    )

                # concepts (top 3 only)
                for concept in work.get("concepts", [])[:3]:
                    cid = _short_id(concept["id"])
                    s.run(
                        """
                        MERGE (c:Concept {id: $id})
                        SET c.name = $name
                        WITH c
                        MATCH (p:Paper {id: $pid})
                        MERGE (p)-[:HAS_CONCEPT]->(c)
                        """,
                        id=cid,
                        name=concept.get("display_name", ""),
                        pid=wid,
                    )

                # citations
                for ref_url in work.get("referenced_works", []):
                    ref_id = _short_id(ref_url)
                    s.run(
                        """
                        MERGE (r:Paper {id: $rid})
                        WITH r
                        MATCH (p:Paper {id: $pid})
                        MERGE (p)-[:CITES]->(r)
                        """,
                        rid=ref_id, pid=wid
                    )

    print("graph loaded into neo4j")


def build_graph(all_works, clear=False):
    driver = get_driver()
    if clear:
        clear_graph(driver)
    create_constraints(driver)
    load_works(driver, all_works)
    driver.close()