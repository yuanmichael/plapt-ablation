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

Per-residue mode (--save-residues)
----------------------------------
Everything above collapses each protein to one vector. A cross-attention head
needs the uncollapsed grid: one row per residue, so a ligand atom can attend to
residue 213 specifically. --save-residues keeps it.

Three decisions, because the obvious implementation is 15x too big:

1. RAGGED, NOT PADDED. Our proteins run from 9 to 7,182 residues (median 375).
   A padded (8638, 7182, 640) array is 79.4 GB of which 93% is padding. We store
   one concatenated (total_residues, hidden) array plus an int64 offsets table:
   protein i occupies rows offsets[i] : offsets[i+1]. That is 5.4 GB at fp16.
   Padding then happens per BATCH at training time, to the longest member of
   that batch, not to the global maximum.

2. REAL RESIDUES ONLY. [CLS]/[SEP] (ProtBERT) and <cls>/<eos> (ESM-2) are
   dropped, so row r of a protein's slice is residue r of its sequence and the
   consumer needs no knowledge of tokenizer conventions. This makes the row
   count predictable -- min(seq_len, max_length - n_special) -- which is what
   lets us preallocate the memmap and write into it as we go rather than
   holding 5.4 GB in RAM.

   We exclude only the STRUCTURAL specials (cls/bos/sep/eos/pad), never
   unk/mask. Dropping an [UNK] would silently shift every subsequent residue by
   one and misalign the whole protein, so the code asserts the actual row count
   against the predicted one and fails loudly rather than writing corrupt data.

3. fp16 IS SAFE HERE, unlike for pooler_output. The fp32 requirement above
   exists because ProtBERT's pooler collapses: its 2nd component sits at 1e-3
   of the 1st, exactly on the fp16 floor. last_hidden_state has no such problem
   -- it is LayerNorm'd to roughly unit scale per dimension and has an effective
   rank in the hundreds. Rather than assume that, the run ends by recomputing
   the mean pool FROM the stored fp16 residues and correlating it against the
   fp32 mean computed in the forward pass. That single check validates the
   ragged layout, the offsets, the special-token masking and the precision all
   at once; anything below r = 0.999 means something is wrong.
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModel, AutoTokenizer, BertModel,
                          BertTokenizerFast, RobertaModel, RobertaTokenizerFast)

PROT_MODEL = "Rostlab/prot_bert"
MOL_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
PROT_MAX = 3200   # PLAPT's setting
MOL_MAX = 278     # PLAPT's setting


def load_tokenizer(name: str):
    """AutoTokenizer cannot resolve every older repo under transformers v5.

    Rostlab/prot_bert ships a plain `vocab.txt` and no `tokenizer.json`, and
    v5 removed the slow tokenizer classes, so AutoTokenizer fails with

        ValueError: Couldn't instantiate the backend tokenizer from one of: ...
        You need to have sentencepiece or tiktoken installed

    Installing sentencepiece does not help (verified) -- the conversion path for
    this repo is simply broken. Naming the concrete class works, because
    BertTokenizerFast can build a WordPiece tokenizer from vocab.txt directly.
    The original code called BertTokenizer explicitly, which is why switching to
    AutoTokenizer introduced this.

    Token ids are identical either way; only the loading path differs.
    """
    attempts = [("AutoTokenizer", lambda: AutoTokenizer.from_pretrained(name, do_lower_case=False)),
                ("BertTokenizerFast", lambda: BertTokenizerFast.from_pretrained(name, do_lower_case=False)),
                ("RobertaTokenizerFast", lambda: RobertaTokenizerFast.from_pretrained(name))]
    errors = []
    for label, fn in attempts:
        try:
            tok = fn()
            if label != "AutoTokenizer":
                print(f"  (AutoTokenizer failed for {name}; loaded with {label})")
            return tok
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}")
    raise RuntimeError(f"could not load a tokenizer for {name} -- tried {', '.join(errors)}")


def load_model(name: str):
    """Same problem as load_tokenizer, one layer down.

    Rostlab/prot_bert's config.json has no `model_type` key, so AutoModel cannot
    work out which class to build and raises

        ValueError: Unrecognized model in Rostlab/prot_bert.
        Should have a `model_type` key in its config.json.

    Naming BertModel explicitly works -- which is what the original code did.
    ESM-2 and ChemBERTa declare model_type properly and go through AutoModel.
    """
    attempts = [("AutoModel", lambda: AutoModel.from_pretrained(name)),
                ("BertModel", lambda: BertModel.from_pretrained(name)),
                ("RobertaModel", lambda: RobertaModel.from_pretrained(name))]
    errors = []
    for label, fn in attempts:
        try:
            m = fn()
            if label != "AutoModel":
                print(f"  (AutoModel failed for {name}; loaded with {label})")
            return m
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}")
    raise RuntimeError(f"could not load a model for {name} -- tried {', '.join(errors)}")


def prot_preprocess(seq: str, model_name: str) -> str:
    """Tokenizer conventions differ between protein encoders, and getting this
    wrong fails silently rather than loudly.

    ProtBERT's vocabulary is one token per amino acid and its tokenizer splits on
    whitespace, so it expects "M A V L K" rather than "MAVLK". Feed it an
    unspaced sequence and you get no error -- just a chain chopped into
    meaningless multi-character word-pieces and embeddings that quietly mean
    nothing. plapt.py also maps the ambiguous residues U/Z/O/B to X.

    ESM-2 tokenizes raw sequence directly and has its own vocabulary entries for
    the ambiguous residues, so it wants the string untouched.

    .upper() is NOT cosmetic. 29 of our 8,638 sequences arrive lowercase, and
    both vocabularies are uppercase-only:

      ESM-2:    the tokenizer splits on whitespace, and a lowercase sequence has
                none, so the ENTIRE protein becomes a single <unk>. A 181-residue
                protein encoded as one meaningless vector.
      ProtBERT: we space out every character first, so each lowercase residue
                becomes its own [UNK] -- 181 tokens, 181 of them garbage. The
                token COUNT is right, which is why this went unnoticed: there was
                no shape mismatch to trip over, only silently wrong content.

    Uppercasing takes the total <unk> count across all 3.8M residues from
    ~10,000 to 1 (a literal digit in one sequence labelled "biotin-C6linker-...",
    which is a conjugate rather than a plain peptide).
    """
    seq = seq.upper()
    if "esm" in model_name.lower():
        return seq
    return " ".join(re.sub(r"[UZOB]", "X", seq))


def count_residue_tokens(texts, tokenizer, max_length, struct_ids, preprocess=None,
                         unk_id=None, batch=256):
    """Exact non-special token count per sequence, by actually tokenising.

    The earlier version predicted this as min(seq_len, max_length - n_special),
    which assumes one token per character. That assumption is true for ProtBERT
    and ESM-2 on clean uppercase input and false the moment it is not -- see
    prot_preprocess. Predicting it meant the preallocated offsets could disagree
    with reality, and the only thing standing between that and a silently
    misaligned 9.75 GB cache was an assertion.

    Tokenising twice costs seconds and removes the assumption. It also gives us
    somewhere to count <unk>, which is the signal that a sequence was accepted
    but not understood.
    """
    counts = np.zeros(len(texts), dtype=np.int64)
    n_unk = 0
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        if preprocess is not None:
            chunk = [preprocess(t) for t in chunk]
        enc = tokenizer(chunk, truncation=True, max_length=max_length)
        for j, ids in enumerate(enc["input_ids"]):
            counts[i + j] = sum(1 for t in ids if t not in struct_ids)
            if unk_id is not None:
                n_unk += sum(1 for t in ids if t == unk_id)
    return counts, n_unk


@torch.no_grad()
def embed_table(texts, tokenizer, model, max_length, device, token_budget,
                preprocess=None, desc="", ckpt_path=None, ckpt_every=50_000,
                dtype=np.float16, exclude_special=False, special_ids=(),
                res_arr=None, res_off=None, struct_ids=()):
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

        if getattr(out, "pooler_output", None) is not None:
            pooler[idx] = out.pooler_output.float().cpu().numpy().astype(dtype)

        # Mean pooling MUST be mask-aware. Averaging over the padded positions
        # too would make a sequence's embedding depend on what else happened to
        # land in its batch -- a nasty, silent, batch-size-dependent bug.
        #
        # With exclude_special we also drop [CLS]/[SEP] (ProtBERT) and <cls>/<eos>
        # (ESM-2), so the average is over real residues only. Different encoders
        # add different numbers of special tokens, so leaving them in would make
        # a cross-encoder comparison depend on tokenizer bookkeeping rather than
        # on the models.
        m = tok["attention_mask"].clone()
        if exclude_special and special_ids:
            for sid in special_ids:
                m &= (tok["input_ids"] != sid)
        mask = m.unsqueeze(-1).float()
        summed = (out.last_hidden_state.float() * mask).sum(dim=1)
        meanp[idx] = (summed / mask.sum(dim=1).clamp(min=1)).cpu().numpy().astype(dtype)

        # --------------------------------------------- per-residue grid
        # Write each protein's real residue rows into its preallocated slice of
        # the ragged array. `idx` is in length-sorted order but the offsets were
        # computed in original row order, so the scatter lands correctly.
        if res_arr is not None:
            hs = out.last_hidden_state.float()
            keep = tok["attention_mask"].bool()
            for sid in struct_ids:
                keep &= (tok["input_ids"] != sid)
            for b, k in enumerate(idx):
                rows = hs[b][keep[b]].cpu().numpy().astype(res_arr.dtype)
                lo, hi = int(res_off[k]), int(res_off[k + 1])
                if rows.shape[0] != hi - lo:
                    # Never write misaligned data. If this fires, the predicted
                    # token count disagrees with the tokenizer's actual output
                    # and every residue after this protein would be shifted.
                    raise RuntimeError(
                        f"row {k}: tokenizer produced {rows.shape[0]} residue rows "
                        f"but {hi - lo} were reserved. Predicted count is "
                        f"min(seq_len, max_length - n_special); check whether this "
                        f"tokenizer adds a different number of special tokens, or "
                        f"emits more than one token per residue.")
                res_arr[lo:hi] = rows

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
    ap.add_argument("--prot-model", default=PROT_MODEL,
                    help="e.g. Rostlab/prot_bert, facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--prot-max", type=int, default=PROT_MAX,
                    help="max protein tokens. PLAPT uses 3200; ESM-2 was trained at 1024")
    ap.add_argument("--tag", default="",
                    help="suffix for the protein output files, so several encoders "
                         "can be cached side by side (e.g. _esm650)")
    ap.add_argument("--skip-ligands", action="store_true",
                    help="ligand embeddings do not change when only the protein "
                         "encoder is swapped; skip to save ~3 min per run")
    ap.add_argument("--save-residues", action="store_true",
                    help="also save the uncollapsed per-residue grid, ragged, as "
                         "prot_residues{tag}.npy + prot_offsets{tag}.npy. Needed "
                         "for a cross-attention head. ~5.4 GB at prot_max=3200, "
                         "~4.9 GB at 1024 (see module docstring)")
    ap.add_argument("--exclude-special", action="store_true",
                    help="exclude [CLS]/[SEP]/<cls>/<eos> from the mean pool. Required "
                         "for a fair comparison ACROSS encoders, since they add "
                         "different special tokens")
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
    mp = mm = None
    if not args.skip_ligands:
        print(f"\nloading {MOL_MODEL}")
        mol_tok = load_tokenizer(MOL_MODEL)
        mol_model = load_model(MOL_MODEL).to(device).eval()
        mp, mm = embed_table(ligands["smiles"].tolist(), mol_tok, mol_model, MOL_MAX,
                             device, args.mol_budget, desc="ligand",
                             ckpt_path=out / "mol_pooler.partial.npy",
                             dtype=np.float16,   # safe: see module docstring
                             exclude_special=args.exclude_special,
                             special_ids=set(mol_tok.all_special_ids))
        np.save(out / "mol_pooler.npy", mp)
        np.save(out / "mol_mean.npy", mm)
        print(f"  wrote mol_pooler.npy {mp.shape} + mol_mean.npy {mm.shape}")
        del mol_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("\nskipping ligands (--skip-ligands); they do not change when only "
              "the protein encoder is swapped")

    # ------------------------------------------------------------- proteins
    print(f"\nloading {args.prot_model}  (max_length={args.prot_max})")
    prot_tok = load_tokenizer(args.prot_model)
    prot_model = load_model(args.prot_model).to(device).eval()
    tag = args.tag

    # -------------------------------------------- per-residue preallocation
    res_arr = res_off = struct_ids = None
    if args.save_residues:
        # Only the structural specials. unk/mask are NOT excluded -- see docstring.
        struct_ids = {getattr(prot_tok, a + "_token_id", None)
                      for a in ("cls", "bos", "sep", "eos", "pad")}
        struct_ids = {s for s in struct_ids if s is not None}
        print("  counting residue tokens...", flush=True)
        rows_per, n_unk = count_residue_tokens(
            proteins["seq"].tolist(), prot_tok, args.prot_max, struct_ids,
            preprocess=lambda s: prot_preprocess(s, args.prot_model),
            unk_id=prot_tok.unk_token_id)
        res_off = np.zeros(len(proteins) + 1, dtype=np.int64)
        np.cumsum(rows_per, out=res_off[1:])
        total = int(res_off[-1])
        hidden = int(prot_model.config.hidden_size)
        gb = total * hidden * 2 / 1e9
        print(f"  per-residue: {total:,} residues x {hidden} @ fp16 = {gb:.2f} GB")
        # A sequence the tokenizer accepted but did not understand produces a
        # valid-looking embedding of nothing. Loud is better than silent.
        if n_unk:
            frac = n_unk / max(total, 1) * 100
            flag = "  <-- INVESTIGATE" if frac > 0.1 else ""
            print(f"  {n_unk:,} <unk> tokens ({frac:.3f}% of residues){flag}")
        # open_memmap writes a real .npy header, so the file is loadable with a
        # plain np.load(..., mmap_mode='r') later. Streaming into it keeps peak
        # RAM at one batch rather than the whole 5 GB.
        res_arr = np.lib.format.open_memmap(
            out / f"prot_residues{tag}.npy", mode="w+",
            dtype=np.float16, shape=(total, hidden))
        np.save(out / f"prot_offsets{tag}.npy", res_off)

    pp, pm = embed_table(proteins["seq"].tolist(), prot_tok, prot_model, args.prot_max,
                         device, args.prot_budget,
                         preprocess=lambda s: prot_preprocess(s, args.prot_model),
                         desc="protein", ckpt_path=out / f"prot_pooler{tag}.partial.npy",
                         dtype=np.float32,   # fp16 would destroy components 2+
                         exclude_special=args.exclude_special,
                         special_ids=set(prot_tok.all_special_ids),
                         res_arr=res_arr, res_off=res_off, struct_ids=struct_ids)
    np.save(out / f"prot_pooler{tag}.npy", pp)
    np.save(out / f"prot_mean{tag}.npy", pm)
    print(f"  wrote prot_pooler{tag}.npy {pp.shape} + prot_mean{tag}.npy {pm.shape}")

    # ------------------------------------------- per-residue integrity check
    # Recompute the mean pool from the stored fp16 ragged array and correlate it
    # against the fp32 mean computed during the forward pass. One number that
    # validates the layout, the offsets, the special-token masking and the
    # precision simultaneously. It is only an exact comparison when
    # --exclude-special is set; otherwise pm includes [CLS]/[SEP] and the two
    # means are legitimately different, so we only warn.
    if res_arr is not None:
        res_arr.flush()
        print(f"  wrote prot_residues{tag}.npy {res_arr.shape} + "
              f"prot_offsets{tag}.npy ({len(res_off):,},)")
        rec = np.zeros_like(pm, dtype=np.float64)
        for i in range(len(proteins)):
            lo, hi = int(res_off[i]), int(res_off[i + 1])
            rec[i] = res_arr[lo:hi].astype(np.float64).mean(axis=0)
        r = float(np.corrcoef(rec.ravel(), pm.astype(np.float64).ravel())[0, 1])
        mad = float(np.abs(rec - pm.astype(np.float64)).max())
        print(f"  integrity: mean-from-residues vs fp32 mean  "
              f"r = {r:.6f}   max|diff| = {mad:.2e}")
        if not args.exclude_special:
            print("    (approximate: pm includes [CLS]/[SEP] but the residue grid "
                  "does not. Re-run with --exclude-special for an exact match.)")
        elif r < 0.999:
            raise RuntimeError(
                f"per-residue integrity check FAILED: r = {r:.6f} < 0.999. The "
                f"ragged array, the offsets or the special-token masking is wrong. "
                f"Do not use prot_residues{tag}.npy.")

    n_trunc = int((proteins["seq_len"] > args.prot_max - 2).sum())
    if n_trunc:
        print(f"  NOTE: {n_trunc:,} / {len(proteins):,} proteins "
              f"({n_trunc/len(proteins)*100:.1f}%) exceed max_length and were truncated")

    # Record what produced these vectors. Future-you will want to know whether a
    # cache on disk came from pooler or mean, fp16 or fp32, which max_length.
    (out / f"embed_meta{tag}.json").write_text(json.dumps({
        "prot_model": args.prot_model, "mol_model": MOL_MODEL,
        "prot_max_length": args.prot_max, "mol_max_length": MOL_MAX,
        "prot_hidden": int(pp.shape[1]),
        "n_proteins": int(len(proteins)), "n_ligands": int(len(ligands)),
        "prot_dtype": "float32", "mol_dtype": "float16",
        "poolings": ["pooler_output", "masked_mean"],
        "exclude_special_from_mean": bool(args.exclude_special),
        "n_proteins_truncated": n_trunc,
        "device": str(device), "limit": args.limit, "tag": tag,
        "residues_saved": bool(args.save_residues),
        "n_residues": int(res_off[-1]) if res_off is not None else 0,
        "residue_dtype": "float16" if args.save_residues else None,
        "residue_layout": ("ragged: prot_residues[offsets[i]:offsets[i+1]] is "
                           "protein i, one row per residue, specials removed"
                           if args.save_residues else None),
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
    checks = [(f"prot_pooler{tag}", pp), (f"prot_mean{tag}", pm)]
    if mp is not None:
        checks += [("mol_pooler", mp), ("mol_mean", mm)]
    for name, arr in checks:
        if arr is None or not np.abs(arr).sum():
            print(f"  {name:16s} (all zeros -- this model exposes no pooler_output)")
            continue
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
