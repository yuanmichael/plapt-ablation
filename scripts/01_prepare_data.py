"""
Stage 1: turn the raw BindingDB-derived dump into a clean, split, deduplicated
experiment dataset -- plus the integer-indexed unique protein / ligand tables
that stage 2 will embed.

Run:
    .venv/Scripts/python.exe -I scripts/01_prepare_data.py --n 200000

Why the file layout is the way it is
------------------------------------
The raw data has 1.84M (protein, ligand, affinity) rows but only ~23k unique
proteins. If we handed stage 2 a flat table of 1.84M rows it would run the
420M-parameter protein encoder 1.84M times to produce ~23k distinct answers.

So we store the data "normalized" (the database sense of the word):

    proteins.parquet   prot_id -> sequence        (one row per unique protein)
    ligands.parquet    mol_id  -> smiles          (one row per unique ligand)
    pairs.parquet      prot_id, mol_id, y, split  (one row per measurement)

Stage 2 embeds each table once. Stage 3 reassembles the full feature matrix
with a fancy-index (emb[pairs.prot_id]), which is instant. This is the single
biggest compute saving in the whole project.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "all.parquet"
OUT = ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000,
                    help="how many deduplicated pairs to sample (0 = keep all)")
    ap.add_argument("--seed", type=int, default=2101,
                    help="2101 is the seed baked into PLAPT's released model filename")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--raw", default=None, help="path to all.parquet")
    ap.add_argument("--out", default=None,
                    help="output dir. DEFAULTS TO data/processed, which OVERWRITES the "
                         "200k tables. Pass a separate dir (e.g. data/processed_full) "
                         "when preparing a different sample, so existing results and any "
                         "derived split stay reproducible.")
    args = ap.parse_args()

    global OUT, RAW
    if args.out:
        OUT = Path(args.out)
    if args.raw:
        RAW = Path(args.raw)
    rng = np.random.default_rng(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"raw: {RAW}\nout: {OUT}")

    df = pd.read_parquet(RAW, columns=["seq", "smiles_can", "neg_log10_affinity_M"])
    df = df.rename(columns={"smiles_can": "smiles", "neg_log10_affinity_M": "y"})
    print(f"raw rows                      {len(df):>12,}")

    # ---------------------------------------------------------------- cleaning
    # Each row is one assay measurement. The same (protein, ligand) pair can be
    # measured many times, in different labs, by different assays. 24.8% of rows
    # are part of such a group, and they disagree by a median 0.62 log units.
    #
    # We collapse each pair to the mean of its measurements. Two reasons:
    #   1. Leakage. A random split would otherwise put measurement A of a pair in
    #      train and measurement B of the SAME pair in validation. The model gets
    #      graded on a question it was already shown the answer to.
    #   2. Noise. Averaging replicates shrinks label noise, so the target the
    #      model chases is a better estimate of the true affinity.
    #
    # Note the tradeoff: the mean is a reasonable summary but it quietly throws
    # away the disagreement, which was itself informative (it told us the noise
    # floor). We keep n_meas so stage 3 can check whether the model does worse on
    # noisily-measured pairs -- a cheap sanity test later.
    before = len(df)
    df = (df.groupby(["seq", "smiles"], as_index=False)
            .agg(y=("y", "mean"), n_meas=("y", "size")))
    print(f"after collapsing dup pairs    {len(df):>12,}   "
          f"(-{before - len(df):,} rows, {(before - len(df)) / before * 100:.1f}%)")

    # ---------------------------------------------------------------- sampling
    # Uniform random sample over PAIRS. This deliberately preserves the natural
    # redundancy structure: proteins that are common in the full dataset stay
    # common here. We want to reproduce PLAPT's conditions, not fix them -- the
    # whole point is to measure what those conditions let a model get away with.
    if args.n and args.n < len(df):
        keep = rng.choice(len(df), size=args.n, replace=False)
        df = df.iloc[np.sort(keep)].reset_index(drop=True)
    print(f"sampled pairs                 {len(df):>12,}")

    # ------------------------------------------------- unique id tables
    # pd.factorize assigns each distinct string an integer 0..k-1 and gives back
    # the lookup table. These integers become row indices into the embedding
    # matrices that stage 2 produces.
    df["prot_id"], prot_vals = pd.factorize(df["seq"])
    df["mol_id"], mol_vals = pd.factorize(df["smiles"])
    proteins = pd.DataFrame({"prot_id": np.arange(len(prot_vals)), "seq": prot_vals})
    ligands = pd.DataFrame({"mol_id": np.arange(len(mol_vals)), "smiles": mol_vals})
    proteins["seq_len"] = proteins["seq"].str.len()
    print(f"unique proteins to embed      {len(proteins):>12,}   "
          f"({len(df) / len(proteins):.1f} pairs each on average)")
    print(f"unique ligands to embed       {len(ligands):>12,}   "
          f"({len(df) / len(ligands):.2f} pairs each on average)")

    # --------------------------------------------------------------- the split
    # Random 90/10 over pairs -- faithful to what PLAPT did, and the thing we are
    # putting on trial. Stage 3 will show how much this split gives away.
    idx = rng.permutation(len(df))
    n_val = int(len(df) * args.val_frac)
    df["split"] = "train"
    df.loc[df.index[idx[:n_val]], "split"] = "val"
    tr = df["split"] == "train"
    print(f"train / val                   {tr.sum():>12,} / {(~tr).sum():,}")

    # ------------------------------------------------------- target scaling
    # Standardize the target to mean 0 / std 1. Neural nets train better when the
    # thing they predict is O(1), and PLAPT's head was trained this way.
    #
    # CRITICAL: statistics come from the TRAINING SPLIT ONLY. If you compute the
    # mean over all the data, information about validation targets leaks into the
    # training signal. It is a small leak, but it is exactly the kind of thing
    # that makes a reported number unreproducible -- and PLAPT itself inherited a
    # scaler (mean 6.51286529169358, scale 1.5614094578916633) fitted over the
    # entire 1.93M-row dataset. We do it properly and record both so stage 3 can
    # report in the original log-unit scale.
    y_mean = float(df.loc[tr, "y"].mean())
    y_std = float(df.loc[tr, "y"].std())
    df["y_scaled"] = (df["y"] - y_mean) / y_std
    print(f"target (train only)           mean {y_mean:.6f}  std {y_std:.6f}")
    print("PLAPT's inherited scaler      mean 6.512865  std 1.561409  (fitted on ALL data)")

    # ------------------------------------------------------------------- write
    pairs = df[["prot_id", "mol_id", "y", "y_scaled", "n_meas", "split"]]
    pairs.to_parquet(OUT / "pairs.parquet", index=False)
    proteins.to_parquet(OUT / "proteins.parquet", index=False)
    ligands.to_parquet(OUT / "ligands.parquet", index=False)
    meta = {
        "n_pairs": int(len(df)), "n_proteins": int(len(proteins)),
        "n_ligands": int(len(ligands)), "seed": args.seed,
        "val_frac": args.val_frac, "y_mean": y_mean, "y_std": y_std,
        "split": "random over pairs (replicating PLAPT)",
        "dedup": "mean of replicate measurements per (seq, smiles)",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))

    # --------------------------------------------- zero-parameter baselines
    # Computed here because they need no embeddings at all, and because every
    # learned number in stage 3 has to be read against them.
    print("\n--- zero-parameter baselines on this exact split ---")
    trd, vad = df[tr], df[~tr]
    gm = trd["y"].mean()

    # Metrics in float64, always. The raw data is float32; summing 20k float32
    # terms leaves ~1e-6 of round-off, which is enough to make np.std() of a
    # genuinely constant array return 9.5e-07 instead of 0 -- and then corrcoef
    # divides by that and hands back a silent nan. Use np.ptp (exact range) to
    # detect a constant prediction rather than comparing a std to a threshold.
    y_true = vad["y"].to_numpy(dtype=np.float64)

    def score(pred, name):
        pred = np.asarray(pred, dtype=np.float64)
        r = 0.0 if np.ptp(pred) == 0 else float(np.corrcoef(pred, y_true)[0, 1])
        rmse = float(np.sqrt(np.mean((pred - y_true) ** 2)))
        print(f"  {name:38s} R = {r:6.4f}   RMSE = {rmse:.4f}")

    score(np.full(len(vad), gm), "predict the global mean")
    score(vad["prot_id"].map(trd.groupby("prot_id")["y"].mean()).fillna(gm).to_numpy(),
          "look up protein's mean affinity")
    score(vad["mol_id"].map(trd.groupby("mol_id")["y"].mean()).fillna(gm).to_numpy(),
          "look up ligand's mean affinity")

    seen_p = vad["prot_id"].isin(trd["prot_id"].unique()).mean()
    seen_m = vad["mol_id"].isin(trd["mol_id"].unique()).mean()
    print(f"\n  val proteins also seen in train: {seen_p * 100:.2f}%")
    print(f"  val ligands  also seen in train: {seen_m * 100:.2f}%")
    print(f"\nwrote -> {OUT}")


if __name__ == "__main__":
    main()
