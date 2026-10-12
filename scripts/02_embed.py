"""
Stage 2: run the two frozen encoders over the unique protein / ligand tables and
cache the resulting vectors to disk. This is the only GPU-hungry step.

Run on Colab (L4):
    !pip -q install transformers
    !python 02_embed.py --data . --out ./emb

Run locally on CPU to smoke-test the code path first (takes a minute):
    .venv/Scripts/python.exe -I scripts/02_embed.py --limit 32

What this step is actually doing
-------------------------------
PLAPT never fine-tunes ProtBERT or ChemBERTa. They are frozen feature
extractors: a sequence goes in, one fixed-length vector comes out, and that
vector never changes no matter how long you train the head. That has a useful
consequence -- the expensive part only has to happen ONCE, ever. After this
script runs, "training PLAPT" is just fitting a 1.5M-parameter MLP on a table of
numbers, which is minutes of CPU work and can be repeated as often as we like.

Caching activations like this is the standard move whenever the upstream model
is frozen. It is also what makes the ablation cheap: all three arms read the
same cached vectors, so the comparison is guaranteed to differ only in which
columns we feed the head.

Two pooling strategies, saved for the price of one
--------------------------------------------------
A transformer's last layer emits one vector PER TOKEN. To get a single vector
for the whole sequence you have to collapse that, and there are two common ways:

  pooler_output -- take the [CLS] token's vector, push it through one extra
                   dense layer + tanh. This is what PLAPT uses.
  mean pooling  -- average the last hidden state across all real (non-padding)
                   tokens.

Mean pooling is widely found to be the stronger summary, and there is a specific
reason to suspect pooler_output is weak for ProtBERT: that extra dense layer was
originally trained by BERT's next-sentence-prediction objective, and ProtBERT
was trained with masked-language-modelling only. If NSP was never run, those
pooler weights may be close to their random initialisation -- meaning PLAPT's
protein representation is a random projection of a token that was never trained
to summarise anything. We save both poolings so that is testable later for free;
the forward pass is the cost, and we are already paying it.

Why proteins are stored fp32 and ligands fp16
---------------------------------------------
Measured on 128 random real proteins (00_diagnose_pooler.py), the singular value
spectrum of ProtBERT's pooler_output, relative to its largest component, is:

    fp32    1.00e+00   9.98e-04   9.61e-06   3.74e-06
    fp16    1.00e+00   1.09e-03   1.05e-03   9.84e-04     <- all noise

float16 carries ~1e-3 of relative precision, and ProtBERT's second component sits
at 9.98e-04 -- exactly on that floor. Quantising to fp16 replaces components 2
onward with rounding noise, which would silently destroy the very signal this
experiment is trying to detect: we would conclude "the protein does not matter"
when the truth was "we threw the protein away in the cache". So proteins are
stored fp32 (8,638 x 1024 x 4 B = 35 MB, free).

ChemBERTa's components sit at 0.87, 0.78, 0.69 of the first -- a thousand times
above the fp16 floor, and its spectrum is unchanged to 3 significant figures by
quantisation. So the big table (182,402 x 768) stays fp16 and we halve 560 MB to
280 MB at no measurable cost.

Precision is an experimental-design decision, not just a storage one.
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

PROT_MODEL = "Rostlab/prot_bert"
MOL_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
PROT_MAX = 3200   # PLAPT's setting
MOL_MAX = 278     # PLAPT's setting


def prot_preprocess(seq: str) -> str:
    """Exactly what plapt.py does: ambiguous residues -> X, then space-separate.

    ProtBERT's vocabulary is one token per amino acid, so the tokenizer expects
    "M A V L K" rather than "MAVLK". Forgetting the spaces is the classic
    ProtBERT bug -- you get no error, just a sequence chopped into meaningless
    multi-character word-pieces and embeddings that quietly mean nothing.
    """
    return " ".join(re.sub(r"[UZOB]", "X", seq))


@torch.no_grad()
def embed_table(texts, tokenizer, model, max_length, device, token_budget,
                preprocess=None, desc="", ckpt_path=None, ckpt_every=50_000,
                dtype=np.float16):
    """Embed a list of strings, returning (pooler, meanpool) as float16 arrays.

    Length-sorted dynamic batching
    ------------------------------
    Transformers process a batch as a rectangle, so every sequence gets padded to
    the longest one in its batch. Mix a 50-residue peptide with a 3,000-residue
    monster and you spend 98% of that batch's compute on padding. Sorting by
    length first means each batch is internally uniform, which on this data is
    roughly a 5-10x speedup for free. We then scatter results back to the
    original row order, so the caller never sees the reordering.

    Attention is quadratic in sequence length, so we budget TOKENS per batch
    rather than fixing a row count: short sequences get big batches, long ones
    get small batches, and GPU memory use stays roughly flat.
    """
    n = len(texts)
    hidden = model.config.hidden_size
    pooler = np.zeros((n, hidden), dtype=dtype)
    meanp = np.zeros((n, hidden), dtype=dtype)

    raw_len = np.fromiter((len(t) for t in texts), dtype=np.int64, count=n)
    order = np.argsort(raw_len, kind="stable")

    done = 0
    t0 = time.time()
    i = 0
    while i < n:
        # Grow the batch while (rows x longest_row) stays under the token budget.
        longest = min(int(raw_len[order[i]]), max_length)
        j = i
        while j < n:
            cand = max(longest, min(int(raw_len[order[j]]), max_length))
            if (j - i + 1) * cand > token_budget and j > i:
                break
            longest = cand
            j += 1
        idx = order[i:j]
        batch = [texts[k] for k in idx]
        if preprocess is not None:
            batch = [preprocess(b) for b in batch]

        tok = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        # autocast runs the matmuls in fp16 for speed while keeping the master
        # weights in fp32 -- faster than full .half() inference is risky-free.
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            out = model(**tok)

        pooler[idx] = out.pooler_output.float().cpu().numpy().astype(dtype)

        # Mean pooling MUST be mask-aware. Averaging over the padded positions
        # too would make a sequence's embedding depend on what else happened to
        # land in its batch -- a nasty, silent, batch-size-dependent bug.
        mask = tok["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state.float() * mask).sum(dim=1)
        meanp[idx] = (summed / mask.sum(dim=1)).cpu().numpy().astype(dtype)

        done += len(idx)
        i = j
        if done % 5000 < len(idx) or i >= n:
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            print(f"  [{desc}] {done:,}/{n:,}  {rate:7.1f}/s  "
                  f"elapsed {el/60:5.1f}m  eta {(n-done)/max(rate,1e-9)/60:5.1f}m",
                  flush=True)
        if ckpt_path is not None and done % ckpt_every < len(idx):
            np.save(ckpt_path, pooler)  # cheap insurance against a disconnect

    return pooler, meanp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dir with proteins/ligands parquet")
    ap.add_argument("--out", default=None, help="dir to write .npy files")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only N rows of each")
    ap.add_argument("--prot-budget", type=int, default=16_000, help="tokens per protein batch")
    ap.add_argument("--mol-budget", type=int, default=65_536, help="tokens per ligand batch")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    data = Path(args.data) if args.data else root / "data" / "processed"
    out = Path(args.out) if args.out else root / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cpu":
        # Keep a laptop smoke test from silently turning into an overnight job.
        args.prot_budget = min(args.prot_budget, 2048)
        args.mol_budget = min(args.mol_budget, 4096)
        if not args.limit:
            print("WARNING: no GPU found. Pass --limit 32 to smoke-test, or run on Colab.")

    proteins = pd.read_parquet(data / "proteins.parquet")
    ligands = pd.read_parquet(data / "ligands.parquet")
    if args.limit:
        proteins, ligands = proteins.head(args.limit), ligands.head(args.limit)
    print(f"proteins: {len(proteins):,}   ligands: {len(ligands):,}")

    # ------------------------------------------------------------- ligands
    # Done first: it is the big table but the small model, so it is the fast
    # confidence-builder. If something is wrong with the pipeline you find out
    # in two minutes instead of twenty.
    print(f"\nloading {MOL_MODEL}")
    mol_tok = RobertaTokenizer.from_pretrained(MOL_MODEL)
    mol_model = RobertaModel.from_pretrained(MOL_MODEL).to(device).eval()
    mp, mm = embed_table(ligands["smiles"].tolist(), mol_tok, mol_model, MOL_MAX,
                         device, args.mol_budget, desc="ligand",
                         ckpt_path=out / "mol_pooler.partial.npy",
                         dtype=np.float16)   # safe: see module docstring
    np.save(out / "mol_pooler.npy", mp)
    np.save(out / "mol_mean.npy", mm)
    print(f"  wrote mol_pooler.npy {mp.shape} + mol_mean.npy {mm.shape}")
    del mol_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------- proteins
    print(f"\nloading {PROT_MODEL}")
    prot_tok = BertTokenizer.from_pretrained(PROT_MODEL, do_lower_case=False)
    prot_model = BertModel.from_pretrained(PROT_MODEL).to(device).eval()
    pp, pm = embed_table(proteins["seq"].tolist(), prot_tok, prot_model, PROT_MAX,
                         device, args.prot_budget, preprocess=prot_preprocess,
                         desc="protein", ckpt_path=out / "prot_pooler.partial.npy",
                         dtype=np.float32)   # fp16 would destroy components 2+
    np.save(out / "prot_pooler.npy", pp)
    np.save(out / "prot_mean.npy", pm)
    print(f"  wrote prot_pooler.npy {pp.shape} + prot_mean.npy {pm.shape}")

    # Record what produced these vectors. Future-you will want to know whether a
    # cache on disk came from pooler or mean, fp16 or fp32, which max_length.
    (out / "embed_meta.json").write_text(json.dumps({
        "prot_model": PROT_MODEL, "mol_model": MOL_MODEL,
        "prot_max_length": PROT_MAX, "mol_max_length": MOL_MAX,
        "n_proteins": int(len(proteins)), "n_ligands": int(len(ligands)),
        "prot_dtype": "float32", "mol_dtype": "float16",
        "poolings": ["pooler_output", "masked_mean"],
        "device": str(device), "limit": args.limit,
    }, indent=2))

    # A frozen encoder that emits near-identical vectors for every input would
    # make the whole experiment vacuous, so measure the spread before we leave.
    #
    # The number to watch is the effective rank: how many independent directions
    # the representation actually uses, out of the dimensions it nominally has.
    # Computed as the participation ratio exp(-sum p log p) over the normalised
    # eigenvalues of the centred covariance. A value near 1 means every input is
    # mapped onto a single line -- the representation has collapsed.
    print("\n--- representation health check ---")
    for name, arr in [("prot_pooler", pp), ("prot_mean", pm),
                      ("mol_pooler", mp), ("mol_mean", mm)]:
        a = arr.astype(np.float64)
        sub = a if len(a) <= 4000 else a[np.random.default_rng(0).choice(len(a), 4000, False)]
        X = sub - sub.mean(0)
        s = np.linalg.svd(X, compute_uv=False)
        p = s**2 / max((s**2).sum(), 1e-300)
        eff = float(np.exp(-(p * np.log(p + 1e-300)).sum()))
        n90 = int(np.searchsorted(np.cumsum(p), 0.90) + 1)
        print(f"  {name:12s} mean|v| {np.abs(a).mean():7.4f}  "
              f"per-dim std {a.std(axis=0).mean():7.4f}  "
              f"eff.rank {eff:7.1f}/{arr.shape[1]}  "
              f"dims for 90% var {n90:5d}")
    print("\n  If prot_pooler shows eff.rank ~1.0, every protein maps to the same")
    print("  line and PLAPT's 524,288-parameter protein branch sees one number.")


if __name__ == "__main__":
    main()
