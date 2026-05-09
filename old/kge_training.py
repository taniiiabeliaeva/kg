"""
KGE Training — Research Bubble Detection
=========================================
Loads the triples exported by pipeline.py and trains a KGE model
to do link prediction (e.g. will paper X cite paper Y in the future?).

This uses PyKEEN but the triples.tsv format works with any KGE library
(AmpliGraph, DGL-KE, etc.) — just swap out the training block.

Requirements:
  pip install pykeen torch
"""

import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

TRIPLES_PATH  = "output/triples.tsv"
OUTPUT_DIR    = "output/kge"
MODEL         = "TransE"    # try also: "ComplEx", "RotatE", "DistMult"
EPOCHS        = 100
EMBEDDING_DIM = 64
TRAIN_RATIO   = 0.8
VAL_RATIO     = 0.1
# TEST_RATIO  = 0.1  (remainder)


# ──────────────────────────────────────────────
# LOAD TRIPLES
# ──────────────────────────────────────────────

def load_triples(path):
    df = pd.read_csv(path, sep="\t")
    print(f"Loaded {len(df)} triples")
    print(df["relation"].value_counts())
    return df


# ──────────────────────────────────────────────
# FILTER — train only on citation edges
# (you can also train on all edges — experiment)
# ──────────────────────────────────────────────

def filter_citation_triples(df):
    cites_df = df[df["relation"] == "cites"].copy()
    print(f"\nCitation triples only: {len(cites_df)}")
    return cites_df


# ──────────────────────────────────────────────
# TRAIN WITH PYKEEN
# ──────────────────────────────────────────────

def train_kge(triples_df, model_name, epochs, embedding_dim, output_dir):
    try:
        from pykeen.triples import TriplesFactory
        from pykeen.pipeline import pipeline
    except ImportError:
        print("PyKEEN not installed. Run: pip install pykeen torch")
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build triples factory
    tf = TriplesFactory.from_labeled_triples(
        triples_df[["head", "relation", "tail"]].values
    )

    # Train/val/test split
    train, val, test = tf.split([TRAIN_RATIO, VAL_RATIO, 1 - TRAIN_RATIO - VAL_RATIO])

    print(f"\nTraining {model_name} for {epochs} epochs...")
    result = pipeline(
        training=train,
        validation=val,
        testing=test,
        model=model_name,
        model_kwargs={"embedding_dim": embedding_dim},
        training_kwargs={"num_epochs": epochs, "batch_size": 256},
        random_seed=42,
        device="cpu",   # change to "cuda" if GPU available
    )

    # Save
    result.save_to_directory(output_dir)
    print(f"\nModel saved to {output_dir}")

    # Print eval metrics
    metrics = result.metric_results.to_df()
    print("\nEvaluation metrics (test set):")
    print(metrics[metrics["Side"] == "both"][["Metric", "Value"]].to_string(index=False))

    return result


# ──────────────────────────────────────────────
# LINK PREDICTION — score future bubble citations
# ──────────────────────────────────────────────

def predict_links(result, head_entity, relation="cites", top_k=10):
    """
    Given a paper, predict which other papers it is most likely to cite.
    Useful for estimating whether a bubble will grow (mostly internal predictions)
    or open up (external predictions).
    """
    if result is None:
        return
    try:
        from pykeen.models.predict import get_tail_prediction_df
        df = get_tail_prediction_df(
            model=result.model,
            head_label=head_entity,
            relation_label=relation,
            triples_factory=result.training,
            add_novelties=True,
        )
        print(f"\nTop {top_k} predicted citations for {head_entity}:")
        print(df.head(top_k)[["tail_label", "score"]].to_string(index=False))
        return df
    except Exception as e:
        print(f"Link prediction error: {e}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    df = load_triples(TRIPLES_PATH)

    # Option A: train on citations only (more focused)
    cites_df = filter_citation_triples(df)

    # Option B: train on full KG (uncomment to use all edge types)
    # cites_df = df

    result = train_kge(cites_df, MODEL, EPOCHS, EMBEDDING_DIM, OUTPUT_DIR)

    # Example link prediction (replace with a real paper ID from your KG)
    # predict_links(result, head_entity="W2345678901")
