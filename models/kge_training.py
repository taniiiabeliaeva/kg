# models/kge_training.py — TransE using plain torch, no pykeen

import pandas as pd
import torch
import torch.nn as nn
import json
from pathlib import Path

TRIPLES_PATH  = "output/triples.tsv"
OUTPUT_DIR    = "output/kge"
EPOCHS        = 200
EMBEDDING_DIM = 64
LEARNING_RATE = 0.005
BATCH_SIZE    = 256
MARGIN        = 1.0
TRAIN_RATIO   = 0.8
NEG_SAMPLES   = 3


def load_triples(path=TRIPLES_PATH):
    df = pd.read_csv(path, sep="\t")
    print(f"{len(df)} triples loaded")
    df = df[df["relation"] == "CITES"].copy()
    print(f"{len(df)} citation triples after filtering")
    return df


def build_vocab(df):
    entities    = pd.unique(df[["head", "tail"]].values.ravel())
    relations   = df["relation"].unique()
    entity2id   = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    return entity2id, relation2id


def encode(df, entity2id, relation2id):
    h = torch.tensor([entity2id[x] for x in df["head"]], dtype=torch.long)
    r = torch.tensor([relation2id[x] for x in df["relation"]], dtype=torch.long)
    t = torch.tensor([entity2id[x] for x in df["tail"]], dtype=torch.long)
    return h, r, t


class TransE(nn.Module):
    def __init__(self, n_entities, n_relations, dim):
        super().__init__()
        self.ent = nn.Embedding(n_entities, dim)
        self.rel = nn.Embedding(n_relations, dim)
        # initialise small — key for stable training
        nn.init.uniform_(self.ent.weight, -0.1, 0.1)
        nn.init.uniform_(self.rel.weight, -0.1, 0.1)

    def score(self, h, r, t):
        # normalise entity embeddings — standard TransE
        h_e = nn.functional.normalize(self.ent(h), p=2, dim=1)
        t_e = nn.functional.normalize(self.ent(t), p=2, dim=1)
        r_e = self.rel(r)
        return torch.norm(h_e + r_e - t_e, p=2, dim=1)


def train(df):
    entity2id, relation2id = build_vocab(df)
    n_ent = len(entity2id)
    n_rel = len(relation2id)

    # shuffle before split
    df_shuf  = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split    = int(len(df_shuf) * TRAIN_RATIO)
    train_df = df_shuf.iloc[:split]
    test_df  = df_shuf.iloc[split:]

    h_tr, r_tr, t_tr = encode(train_df, entity2id, relation2id)
    h_te, r_te, t_te = encode(test_df,  entity2id, relation2id)

    model     = TransE(n_ent, n_rel, EMBEDDING_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn   = nn.MarginRankingLoss(margin=MARGIN)

    print(f"\ntraining TransE — {n_ent} entities, {EPOCHS} epochs")
    print(f"train: {len(train_df)}  test: {len(test_df)}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(h_tr))
        h_tr, r_tr, t_tr = h_tr[perm], r_tr[perm], t_tr[perm]
        total_loss = 0

        for i in range(0, len(h_tr), BATCH_SIZE):
            bh = h_tr[i:i+BATCH_SIZE]
            br = r_tr[i:i+BATCH_SIZE]
            bt = t_tr[i:i+BATCH_SIZE]

            bh_r  = bh.repeat_interleave(NEG_SAMPLES)
            br_r  = br.repeat_interleave(NEG_SAMPLES)
            bt_r  = bt.repeat_interleave(NEG_SAMPLES)
            neg_t = torch.randint(0, n_ent, bt_r.shape)

            pos_s = model.score(bh_r, br_r, bt_r)
            neg_s = model.score(bh_r, br_r, neg_t)
            loss  = loss_fn(pos_s, neg_s, -torch.ones(len(bh_r)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 40 == 0:
            print(f"  epoch {epoch:3d} / {EPOCHS}  loss: {total_loss:.4f}")

    # eval on a sample — full eval over 30k entities is slow on cpu
    model.eval()
    sample = min(300, len(h_te))
    hits1, hits10 = 0, 0
    with torch.no_grad():
        for i in range(sample):
            h, r, t = h_te[i:i+1], r_te[i:i+1], t_te[i:i+1]
            scores = model.score(h.expand(n_ent), r.expand(n_ent), torch.arange(n_ent))
            rank = (scores < scores[t.item()]).sum().item() + 1
            if rank <= 1:  hits1  += 1
            if rank <= 10: hits10 += 1

    print(f"\nhits@1  (sample {sample}): {hits1/sample:.4f}")
    print(f"hits@10 (sample {sample}): {hits10/sample:.4f}")
    print(f"(random baseline with {n_ent} entities ≈ {10/n_ent:.5f})")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/transe_model.pt")
    with open(f"{OUTPUT_DIR}/entity2id.json", "w") as f:
        json.dump(entity2id, f)
    with open(f"{OUTPUT_DIR}/relation2id.json", "w") as f:
        json.dump(relation2id, f)
    print(f"\nmodel saved to {OUTPUT_DIR}/")
    return model, entity2id, relation2id


def predict_links(model, entity2id, relation2id, head_id, relation="CITES", top_k=10):
    id2entity = {v: k for k, v in entity2id.items()}
    if head_id not in entity2id:
        print(f"{head_id} not found in vocabulary")
        return
    h = torch.tensor([entity2id[head_id]])
    r = torch.tensor([relation2id[relation]])
    n = len(entity2id)
    model.eval()
    with torch.no_grad():
        scores = model.score(h.expand(n), r.expand(n), torch.arange(n))
    top_idx = scores.argsort()[:top_k]
    print(f"\ntop {top_k} predicted citations for {head_id}:")
    for idx in top_idx:
        print(f"  {id2entity[idx.item()]}  score: {scores[idx].item():.4f}")


if __name__ == "__main__":
    df = load_triples()
    model, entity2id, relation2id = train(df)

    # to test link prediction, grab a paper id from output/bubble_report.json
    # and uncomment:
    # predict_links(model, entity2id, relation2id, "W2345678901")