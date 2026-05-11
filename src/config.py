import os
from dotenv import load_dotenv

load_dotenv()

# openalex
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")

# neo4j
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# institutions to study — openalex ids
INSTITUTIONS = {
    "TU Wien":    "I57206974",
    "ETH Zurich": "I114027177",
    "TU Berlin":  "I63966007",
}

# filter by field — "computer science" concept id
FIELD_CONCEPT_ID = "C41008148"

YEAR_START = 2015
YEAR_END   = 2025

MAX_WORKS_PER_INSTITUTION = 1000
