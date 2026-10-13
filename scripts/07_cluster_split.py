"""
Stage 7: replace the random split over pairs with a cluster split at 30% sequence
identity.

Why this is the experiment that matters
---------------------------------------
Under the random split we currently use, 97.66% of test pairs have a protein the
model trained on -- in a median of 168 other pairs. So the typical test question
is "here is a protein you have seen 168 times, plus a new ligand", and a model
can score well by recalling that protein's average affinity. Measured: the
protein-only arm tracks a zero-parameter lookup table to within 0.013 at every
training-set size.

A cluster split removes that route. Sequences are grouped so that any two
sequences sharing >=30% identity land in the same cluster, and then WHOLE
CLUSTERS are assigned to train, dev or test. No test protein has a close
relative anywhere in training.

Dev is cluster-disjoint too. If dev shared clusters with train, early stopping
would be tuned on homologs of the training set and would stop too late.

Run (needs the `mmseqs` binary on PATH; easiest via modal_app.py::cluster):
    python 07_cluster_split.py --data data/processed --min-seq-id 0.3

Writes pairs_clustered.parquet, with split in {train, dev, test}, alongside the
original pairs.parquet. Nothing is overwritten.

Reading the output
------------------
The number to watch is the protein lookup baseline. Under the random split it is
R = 0.5784 -- better than every trained protein-only arm we have measured. Under
a correct cluster split it must collapse toward 0, because no test protein was
ever seen and the table has nothing to look up. If it does not collapse, the
split is leaking and the rest of the numbers are meaningless.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def write_fasta(proteins: pd.DataFrame, path: Path) -> None:
    with open(path, "w") as f:
        for pid, seq in zip(proteins["prot_id"], proteins["seq"]):
            f.write(f">{pid}\n{seq.upper()}\n")


def run_mmseqs(fasta: Path, workdir: Path, min_seq_id: float, cov: float,
               threads: int) -> Path:
    """easy-cluster writes <prefix>_cluster.tsv: representative <TAB> member."""
    prefix = workdir / "clu"
    cmd = ["mmseqs", "easy-cluster", str(fasta), str(prefix), str(workdir / "tmp"),
           "--min-seq-id", str(min_seq_id), "-c", str(cov),
           "--cov-mode", "0", "--threads", str(threads)]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return Path(str(prefix) + "_cluster.tsv")


def assign_clusters(cluster_of: dict, pairs: pd.DataFrame, dev_frac: float,
                    test_frac: float, seed: int):
    """Shuffle clusters, then fill test and dev to their pair-count targets.

    Assigning by PAIR count rather than protein count matters: cluster sizes span
    four orders of magnitude here (the biggest single protein has 10,162 pairs),
    so splitting evenly by protein would give a test set of almost no pairs.

    Clusters are taken in shuffled order and the last one is allowed to
    overshoot. The alternative -- skipping clusters that overshoot -- would
    silently bias test toward small clusters, i.e. toward less-studied proteins.
    """
    rng = np.random.default_rng(seed)
    pair_prot = pairs["prot_id"].to_numpy()
    pair_clu = np.array([cluster_of[p] for p in pair_prot])
    sizes = pd.Series(pair_clu).value_counts()          # pairs per cluster
    # .copy() because a pandas Index's .to_numpy() can return a read-only view,
    # and rng.shuffle writes in place.
    clusters = sizes.index.to_numpy().copy()
    rng.shuffle(clusters)

    n = len(pairs)
    want = {"test": int(n * test_frac), "dev": int(n * dev_frac)}
    label, got, i = {}, {"test": 0, "dev": 0}, 0
    for split in ("test", "dev"):
        while got[split] < want[split] and i < len(clusters):
            c = clusters[i]
            label[c] = split
            got[split] += int(sizes[c])
            i += 1
    for c in clusters[i:]:
        label[c] = "train"
    return pair_clu, np.array([label[c] for c in pair_clu]), sizes


def lookup_r(train_mask, test_mask, key, y):
    """R of 'predict this entity's mean training affinity', float64 throughout."""
    y = np.asarray(y, dtype=np.float64)
    gm = y[train_mask].mean()
    tbl = pd.Series(y[train_mask]).groupby(key[train_mask]).mean()
    pred = pd.Series(key[test_mask]).map(tbl).fillna(gm).to_numpy(dtype=np.float64)
    true = y[test_mask]
    if np.ptp(pred) == 0:
        return 0.0, float((pd.Series(key[test_mask]).isin(tbl.index)).mean())
    return (float(np.corrcoef(pred, true)[0, 1]),
            float((pd.Series(key[test_mask]).isin(tbl.index)).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--min-seq-id", type=float, default=0.3)
    ap.add_argument("--cov", type=float, default=0.8,
                    help="minimum alignment coverage; 0.8 is the usual companion "
                         "to --min-seq-id 0.3")
    ap.add_argument("--dev-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--cluster-tsv", default=None,
                    help="skip mmseqs and reuse an existing *_cluster.tsv")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    data = Path(args.data) if args.data else root / "data" / "processed"
    work = Path(args.workdir) if args.workdir else data / "cluster_work"
    work.mkdir(parents=True, exist_ok=True)

    proteins = pd.read_parquet(data / "proteins.parquet")
    pairs = pd.read_parquet(data / "pairs.parquet")
    meta = json.loads((data / "meta.json").read_text())
    print(f"{len(proteins):,} proteins, {len(pairs):,} pairs")

    if args.cluster_tsv:
        tsv = Path(args.cluster_tsv)
    else:
        fasta = work / "proteins.fasta"
        write_fasta(proteins, fasta)
        tsv = run_mmseqs(fasta, work, args.min_seq_id, args.cov, args.threads)

    clu = pd.read_csv(tsv, sep="\t", header=None, names=["rep", "member"])
    cluster_of = dict(zip(clu["member"].astype(int), clu["rep"].astype(int)))
    missing = set(proteins["prot_id"]) - set(cluster_of)
    if missing:
        raise RuntimeError(f"{len(missing)} proteins absent from the cluster table")

    n_clu = clu["rep"].nunique()
    csize = clu.groupby("rep").size()
    print(f"\n{n_clu:,} clusters at {args.min_seq_id:.0%} identity / "
          f"{args.cov:.0%} coverage")
    print(f"  {len(proteins)/n_clu:.2f} proteins per cluster on average")
    print(f"  singleton clusters      {(csize == 1).sum():,} ({(csize==1).mean()*100:.1f}%)")
    print(f"  largest cluster         {csize.max():,} proteins")

    pair_clu, split, sizes = assign_clusters(
        cluster_of, pairs, args.dev_frac, args.test_frac, meta["seed"])
    out = pairs.copy()
    out["split"] = split
    out["prot_cluster"] = pair_clu

    print("\n--- realised split (by pairs) ---")
    for s in ("train", "dev", "test"):
        m = split == s
        print(f"  {s:6s} {m.sum():>9,} pairs  ({m.mean()*100:5.1f}%)  "
              f"{out.loc[m,'prot_id'].nunique():>7,} proteins  "
              f"{out.loc[m,'prot_cluster'].nunique():>6,} clusters")

    # ------------------------------------------------------------ leakage check
    tr, te, dv = split == "train", split == "test", split == "dev"
    ctr = set(out.loc[tr, "prot_cluster"])
    print("\n--- leakage ---")
    print(f"  test clusters also in train   "
          f"{len(set(out.loc[te,'prot_cluster']) & ctr)}  (must be 0)")
    print(f"  dev  clusters also in train   "
          f"{len(set(out.loc[dv,'prot_cluster']) & ctr)}  (must be 0)")
    ptr = set(out.loc[tr, "prot_id"])
    seen = out.loc[te, "prot_id"].isin(ptr).mean()
    print(f"  test pairs whose protein was seen   {seen*100:.2f}%   "
          f"(was 97.66% under the random split)")

    # -------------------------------------------------- baselines, the real test
    print("\n--- zero-parameter baselines under THIS split ---")
    y = pairs["y"].to_numpy(dtype=np.float64)
    pr, pcov = lookup_r(tr, te, pairs["prot_id"].to_numpy(), y)
    mr, mcov = lookup_r(tr, te, pairs["mol_id"].to_numpy(), y)
    print(f"  protein lookup   R = {pr:6.4f}   (coverage {pcov*100:5.2f}%)"
          f"   random split: 0.5784")
    print(f"  ligand  lookup   R = {mr:6.4f}   (coverage {mcov*100:5.2f}%)"
          f"   random split: 0.1841")
    if pr > 0.2:
        print("  WARNING: the protein lookup table still works. Either the split "
              "is leaking or affinity correlates with something cluster-level.")

    # Target scaling must be refitted: the train split has changed.
    y_mean = float(y[tr].mean())
    y_std = float(y[tr].std(ddof=1))
    out["y_scaled"] = (y - y_mean) / y_std
    print(f"\ntarget refitted on the new train split: mean {y_mean:.6f} "
          f"std {y_std:.6f}   (was {meta['y_mean']:.6f} / {meta['y_std']:.6f})")

    out.to_parquet(data / "pairs_clustered.parquet", index=False)
    (data / "meta_clustered.json").write_text(json.dumps({
        **meta, "split": f"cluster split, mmseqs2 min-seq-id {args.min_seq_id} "
                         f"cov {args.cov}, whole clusters to train/dev/test",
        "y_mean": y_mean, "y_std": y_std, "n_clusters": int(n_clu),
        "min_seq_id": args.min_seq_id, "cov": args.cov,
        "protein_lookup_r": pr, "ligand_lookup_r": mr,
        "test_pairs_protein_seen_frac": float(seen),
    }, indent=2))
    print(f"\nwrote -> {data/'pairs_clustered.parquet'} and meta_clustered.json")
    print("\nThen, to train on it:")
    print("  03_train_ablation.py --pooling mean --pairs pairs_clustered.parquet "
          "--meta meta_clustered.json --arms full,mol_zero,prot_zero --seeds 3")


if __name__ == "__main__":
    main()
