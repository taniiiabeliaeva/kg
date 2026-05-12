#  TransE and ComplEx, trained on affiliated papers only that gives a much smaller entity space (~4500 vs 183k) and meaningful hits@10

import argparse
import pandas as pd
import torch
import torch.nn as nn
import json
from pathlib import Path

TRIPLES_PATH  = "output/triples.tsv"
OUTPUT_DIR    = "output/kge"
EPOCHS        = 300
EMBEDDING_DIM = 64
LEARNING_RATE = 0.005
BATCH_SIZE    = 128
MARGIN        = 1.0
TRAIN_RATIO   = 0.8
NEG_SAMPLES   = 5


def load_triples(path=TRIPLES_PATH):
    full_df = pd.read_csv(path, sep="\t")
    print(f"{len(full_df)} total triples loaded")

    # only keep papers that have a known institution affiliation
    # these are the ~4500 papers we actually fetched, not reference stubs
    affiliated_papers = set(
        full_df[full_df["relation"] == "AFFILIATED_WITH"]["head"].unique()
    )
    print(f"{len(affiliated_papers)} affiliated papers in dataset")

    # filter citation triples to only include affiliated papers on both ends
    cites = full_df[full_df["relation"] == "CITES"].copy()
    cites_filtered = cites[
        cites["head"].isin(affiliated_papers) &
        cites["tail"].isin(affiliated_papers)
    ].copy()

    print(f"{len(cites_filtered)} citation triples between affiliated papers")
    print(f"(vs {len(cites)} total citation triples before filtering)")
    return cites_filtered, affiliated_papers


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
    """
    TransE: score(h,r,t) = ||h + r - t||
    simple, works well for directional relations
    struggles with asymmetric ones like citations
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
    ComplEx: complex-valued embeddings
    score = Re(<h, r, conj(t)>)
    handles asymmetric relations better — citation is asymmetric
    """
    def __init__(self, n_ent, n_rel, dim):
        super().__init__()
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
        score = (
            (h_re * r_re * t_re).sum(dim=1)
            + (h_im * r_im * t_re).sum(dim=1)
            + (h_re * r_im * t_im).sum(dim=1)
            - (h_im * r_re * t_im).sum(dim=1)
        )
        return -score  # negate: lower = more plausible, consistent with TransE


MODELS = {"TransE": TransE, "ComplEx": ComplEx}



def train(df, model_name):
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

        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d} / {EPOCHS}  loss: {total_loss:.4f}")

    # eval — full eval is now feasible since entity space is small
    model.eval()
    sample = min(500, len(h_te))
    hits1, hits10, mrr = 0, 0, 0.0
    with torch.no_grad():
        for i in range(sample):
            h, r, t = h_te[i:i+1], r_te[i:i+1], t_te[i:i+1]
            scores = model.score(h.expand(n_ent), r.expand(n_ent), torch.arange(n_ent))
            rank = (scores < scores[t.item()]).sum().item() + 1
            if rank <= 1:  hits1  += 1
            if rank <= 10: hits10 += 1
            mrr += 1.0 / rank

    print(f"\nresults on {sample} test triples:")
    print(f"  hits@1:  {hits1/sample:.4f}")
    print(f"  hits@10: {hits10/sample:.4f}")
    print(f"  MRR:     {mrr/sample:.4f}")
    print(f"  (random baseline hits@10 ≈ {10/n_ent:.5f}  —  {(hits10/sample)/(10/n_ent):.1f}x better)")

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
        print(f"{head_id} not in vocabulary — it may be a stub paper not in affiliated set")
        # suggest a valid id
        sample_ids = list(entity2id.keys())[:3]
        print(f"try one of these instead: {sample_ids}")
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
    parser.add_argument("--model", default="TransE", choices=list(MODELS.keys()))
    args = parser.parse_args()

    df, affiliated_papers = load_triples()
    model, entity2id, relation2id = train(df, args.model)

    # to test link prediction, uncomment and replace with a real paper id:
    predict_links(model, entity2id, relation2id, "W2949177718")