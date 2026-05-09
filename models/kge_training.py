# models/kge_training.py — trains a kge model on the exported triples

import pandas as pd
from pathlib import Path

TRIPLES_PATH  = "output/triples.tsv"
OUTPUT_DIR    = "output/kge"
MODEL         = "TransE"   # options: "TransE", "ComplEx", "RotatE"
EPOCHS        = 100
EMBEDDING_DIM = 64
TRAIN_RATIO   = 0.8
VAL_RATIO     = 0.1


def load_triples(path=TRIPLES_PATH):
    df = pd.read_csv(path, sep="\t")
    print(f"{len(df)} triples loaded")
    print(df["relation"].value_counts().to_string())
    return df


def train(triples_df, model_name=MODEL, epochs=EPOCHS, dim=EMBEDDING_DIM):
    try:
        from pykeen.triples import TriplesFactory
        from pykeen.pipeline import pipeline
    except ImportError:
        print("pykeen not installed — run: pip install pykeen torch")
        return None

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    tf = TriplesFactory.from_labeled_triples(
        triples_df[["head", "relation", "tail"]].values
    )
    train_tf, val_tf, test_tf = tf.split([TRAIN_RATIO, VAL_RATIO, 1 - TRAIN_RATIO - VAL_RATIO])

    print(f"\ntraining {model_name} for {epochs} epochs...")
    result = pipeline(
        training=train_tf,
        validation=val_tf,
        testing=test_tf,
        model=model_name,
        model_kwargs={"embedding_dim": dim},
        training_kwargs={"num_epochs": epochs, "batch_size": 256},
        random_seed=42,
        device="cpu",  # change to "cuda" if you have a gpu
    )

    result.save_to_directory(OUTPUT_DIR)
    print(f"model saved to {OUTPUT_DIR}")

    metrics = result.metric_results.to_df()
    print(metrics[metrics["Side"] == "both"][["Metric", "Value"]].to_string(index=False))

    return result


def predict_links(result, head_entity, relation="CITES", top_k=10):
    # given a paper id, predict which papers it's most likely to cite next
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
        print(f"\ntop {top_k} predicted links for {head_entity}:")
        print(df.head(top_k)[["tail_label", "score"]].to_string(index=False))
        return df
    except Exception as e:
        print(f"prediction failed: {e}")


if __name__ == "__main__":
    df = load_triples()

    # train on citation edges only — more focused for bubble analysis
    cites_df = df[df["relation"] == "CITES"].copy()

    # swap to df if you want to train on the full graph
    # cites_df = df

    result = train(cites_df)

    # uncomment and replace with a real paper id to test link prediction
    # predict_links(result, "W2345678901")
