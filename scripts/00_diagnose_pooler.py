"""
Side quest: how much information does each encoder's pooler_output actually carry?

PLAPT feeds `pooler_output` from both encoders into its head. For a BERT-family
model that is the [CLS] token's final vector pushed through one extra dense layer
plus tanh. The dense layer exists to serve BERT's next-sentence-prediction
objective -- and if a model was pretrained without a working NSP loss, that layer
gets no useful gradient.

Inspecting the released weights showed two different pathologies:

    ChemBERTa pooler.dense  weight std 0.020011 (== config initializer_range),
                            bias EXACTLY all zeros
        -> never trained at all. RoBERTa has no NSP objective, so this layer is
           still at its random initialisation. That is survivable: a random
           linear map is roughly information-preserving (Johnson-Lindenstrauss),
           so the CLS vector's content mostly passes through.

    ProtBERT  pooler.dense  weight std 0.001429 (14x SMALLER than the 0.02 init),
                            bias nonzero (std 0.0878)
        -> trained, but driven toward zero. With W ~ 0 the output collapses to
           tanh(b) + a small input-dependent perturbation, i.e. nearly the same
           vector for every protein on earth.

This script measures the consequence directly: embed a random sample of real
proteins and ligands, then look at how distinguishable the resulting vectors are.
Pairwise cosine similarity near 1.0 for arbitrary pairs means the representation
has collapsed.

Run (CPU, a few minutes):
    .venv/Scripts/python.exe -I scripts/00_diagnose_pooler.py --n-prot 128 --n-mol 512
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"


def offdiag_cosines(X):
    """Cosine similarity between every distinct pair of rows."""
    X = X.astype(np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    C = X @ X.T
    iu = np.triu_indices(len(X), k=1)
    return C[iu]


def report(name, pooler, mean):
    print(f"\n--- {name} ---")
    for label, X in (("pooler_output", pooler), ("masked mean", mean)):
        c = offdiag_cosines(X)
        a = X.astype(np.float64)
        # "effective rank": how many directions the representation really uses.
        # A collapsed representation spends all its variance on one direction.
        s = np.linalg.svd(a - a.mean(0), compute_uv=False)
        p = s**2 / (s**2).sum()
        eff_rank = float(np.exp(-(p * np.log(p + 1e-300)).sum()))
        print(f"  {label:14s} cos(pairs): mean {c.mean():.4f}  "
              f"p5 {np.quantile(c,.05):.4f}  p95 {np.quantile(c,.95):.4f}")
        print(f"  {'':14s} |v| {np.abs(a).mean():.4f}   "
              f"participation-ratio effective rank {eff_rank:6.1f} / {X.shape[1]}")


@torch.no_grad()
def embed(texts, tok, model, max_len, preprocess=None):
    pool, mean = [], []
    for i, t in enumerate(texts):
        s = preprocess(t) if preprocess else t
        enc = tok([s], truncation=True, max_length=max_len, return_tensors="pt")
        out = model(**enc)
        pool.append(out.pooler_output[0].numpy())
        mean.append(out.last_hidden_state[0].mean(0).numpy())
        if (i + 1) % 32 == 0:
            print(f"    {i+1}/{len(texts)}", flush=True)
    return np.stack(pool), np.stack(mean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prot", type=int, default=128)
    ap.add_argument("--n-mol", type=int, default=512)
    ap.add_argument("--max-prot-len", type=int, default=1200,
                    help="cap sequence length so this stays a few minutes on CPU")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    proteins = pd.read_parquet(PROC / "proteins.parquet")
    ligands = pd.read_parquet(PROC / "ligands.parquet")

    # Random sample, not the first N -- the table is in order of first appearance
    # in the source data, so the head of it is full of related entries.
    pr = proteins[proteins.seq_len <= args.max_prot_len]
    pr = pr.iloc[rng.choice(len(pr), min(args.n_prot, len(pr)), replace=False)]
    lg = ligands.iloc[rng.choice(len(ligands), min(args.n_mol, len(ligands)), replace=False)]
    print(f"sampled {len(pr)} proteins (median {int(pr.seq_len.median())} aa) "
          f"and {len(lg)} ligands")

    import re
    print("\nembedding ligands with ChemBERTa...")
    mt = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    mm = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").eval()
    mp, mmean = embed(lg["smiles"].tolist(), mt, mm, 278)
    del mm

    print("\nembedding proteins with ProtBERT...")
    pt = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    pm = BertModel.from_pretrained("Rostlab/prot_bert").eval()
    pp, pmean = embed(pr["seq"].tolist(), pt, pm, 3200,
                      preprocess=lambda s: " ".join(re.sub(r"[UZOB]", "X", s)))

    report("ProtBERT on random real proteins", pp, pmean)
    report("ChemBERTa on random real ligands", mp, mmean)

    print("\nHow to read this: two unrelated proteins SHOULD give distinguishable")
    print("vectors. If mean cos(pairs) ~ 1.00 and effective rank ~ 1, every protein")
    print("looks the same to the model and PLAPT's protein branch is near-decorative.")
    np.savez(ROOT / "results" / "pooler_diagnostic.npz",
             prot_pooler=pp, prot_mean=pmean, mol_pooler=mp, mol_mean=mmean)


if __name__ == "__main__":
    main()
