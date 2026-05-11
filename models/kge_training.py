# models/kge_training.py — TransE and ComplEx using plain torch
import argparse
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


# data 

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


# models 

class TransE(nn.Module):
    """
    TransE: score(h,r,t) = ||h + r - t||
    works well for simple hierarchical / directional relations like citations
    """
    def __init__(self, n_ent, n_rel, dim):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        nn.init.uniform_(self.ent.weight, -0.1, 0.1)
        nn.init.uniform_(self.rel.weight, -0.1, 0.1)

    def score(self, h, r, t):
        h_e = nn.functional.normalize(self.ent(h), p=2, dim=1)
        t_e = nn.functional.normalize(self.ent(t), p=2, dim=1)
        r_e = self.rel(r)
        return torch.norm(h_e + r_e - t_e, p=2, dim=1)


class ComplEx(nn.Module):
    """
    ComplEx: uses complex-valued embeddings (real + imaginary parts)
    better than TransE for asymmetric relations — useful here since
    citation is asymmetric (A cites B does not mean B cites A)
    score(h,r,t) = Re(<h, r, conj(t)>)
    """
    def __init__(self, n_ent, n_rel, dim):
        super().__init__()
        # each embedding split into real and imaginary halves
        self.ent_re = nn.Embedding(n_ent, dim)
        self.ent_im = nn.Embedding(n_ent, dim)
        self.rel_re = nn.Embedding(n_rel, dim)
        self.rel_im = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent_re.weight)
        nn.init.xavier_uniform_(self.ent_im.weight)
        nn.init.xavier_uniform_(self.rel_re.weight)
        nn.init.xavier_uniform_(self.rel_im.weight)

    def score(self, h, r, t):
        h_re, h_im = self.ent_re(h), self.ent_im(h)
        r_re, r_im = self.rel_re(r), self.rel_im(r)
        t_re, t_im = self.ent_re(t), self.ent_im(t)
        # Re(<h, r, conj(t)>) — lower magnitude = less plausible, so negate for consistency
        score = (
            (h_re * r_re * t_re).sum(dim=1)
            + (h_im * r_im * t_re).sum(dim=1)
            + (h_re * r_im * t_im).sum(dim=1)
            - (h_im * r_re * t_im).sum(dim=1)
        )
        return -score  # negate so lower = more plausible, consistent with TransE


MODELS = {"TransE": TransE, "ComplEx": ComplEx}


#training
def train(df, model_name):
    if model_name not in MODELS:
        raise ValueError(f"unknown model '{model_name}' — choose from {list(MODELS.keys())}")

    entity2id, relation2id = build_vocab(df)
    n_ent = len(entity2id)
    n_rel = len(relation2id)

    df_shuf  = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split    = int(len(df_shuf) * TRAIN_RATIO)
    train_df = df_shuf.iloc[:split]
    test_df  = df_shuf.iloc[split:]

    h_tr, r_tr, t_tr = encode(train_df, entity2id, relation2id)
    h_te, r_te, t_te = encode(test_df,  entity2id, relation2id)

    model     = MODELS[model_name](n_ent, n_rel, EMBEDDING_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn   = nn.MarginRankingLoss(margin=MARGIN)

    print(f"\ntraining {model_name} — {n_ent} entities, {EPOCHS} epochs")
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

    # eval
    model.eval()
    sample = min(300, len(h_te))
    hits1, hits10, mrr = 0, 0, 0.0
    with torch.no_grad():
        for i in range(sample):
            h, r, t = h_te[i:i+1], r_te[i:i+1], t_te[i:i+1]
            scores = model.score(h.expand(n_ent), r.expand(n_ent), torch.arange(n_ent))
            rank = (scores < scores[t.item()]).sum().item() + 1
            if rank <= 1:  hits1  += 1
            if rank <= 10: hits10 += 1
            mrr += 1.0 / rank

    print(f"\nresults on sample of {sample} test triples:")
    print(f"  hits@1:  {hits1/sample:.4f}")
    print(f"  hits@10: {hits10/sample:.4f}")
    print(f"  MRR:     {mrr/sample:.4f}")
    print(f"  (random baseline hits@10 ≈ {10/n_ent:.5f})")

    # save
    out = Path(OUTPUT_DIR) / model_name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    with open(out / "entity2id.json", "w") as f:
        json.dump(entity2id, f)
    with open(out / "relation2id.json", "w") as f:
        json.dump(relation2id, f)
    print(f"\nmodel saved to {out}/")

    return model, entity2id, relation2id


# link prediction 

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


# run 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TransE", choices=list(MODELS.keys()),
                        help="KGE model to train (default: TransE)")
    args = parser.parse_args()

    df = load_triples()
    model, entity2id, relation2id = train(df, args.model)

    # to test link prediction, uncomment and replace with a real paper id:
    predict_links(model, entity2id, relation2id, "W2949177718")