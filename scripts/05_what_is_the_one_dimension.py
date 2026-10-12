"""
ProtBERT's pooler_output is effectively rank 1: every protein maps to
mu + alpha*u for one fixed direction u. So a protein's entire representation,
as far as PLAPT is concerned, is the single scalar alpha.

Sequence length explains only r^2 = 0.27 of alpha. This script asks what the
other ~73% is, by regressing alpha against simple, interpretable sequence
statistics.

Run:
    .venv/Scripts/python.exe -I scripts/05_what_is_the_one_dimension.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

ROOT = Path(__file__).resolve().parent.parent
AA = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydropathy: positive = hydrophobic, negative = hydrophilic.
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
# Average residue masses in daltons.
MW = {"A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14, "Q": 128.13,
      "E": 129.12, "G": 57.05, "H": 137.14, "I": 113.16, "L": 113.16, "K": 128.17,
      "M": 131.19, "F": 147.18, "P": 97.12, "S": 87.08, "T": 101.10, "W": 186.21,
      "Y": 163.18, "V": 99.13}
CHARGE = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}

# Single-residue "groups" (frac_proline, frac_cysteine, frac_glycine) were removed:
# they are EXACTLY equal to aa_P / aa_C / aa_G, and exact duplicate columns made the
# design matrix singular (condition number 2.3e16), which produced a nonsense
# cross-validated R2 of -2.07 on the first run.
GROUPS = {
    "frac_hydrophobic": "AVILMFWC",
    "frac_aromatic": "FWY",
    "frac_charged": "DEKR",
    "frac_positive": "KRH",
    "frac_negative": "DE",
    "frac_polar": "STNQ",
    "frac_tiny": "AGS",
}


def features(seqs):
    rows = []
    for s in seqs:
        n = len(s)
        counts = {a: s.count(a) for a in AA}
        known = sum(counts.values()) or 1
        f = {"length": n, "log_length": np.log(n)}
        f["mol_weight"] = sum(counts[a] * MW[a] for a in AA)
        f["gravy"] = sum(counts[a] * KD[a] for a in AA) / known       # mean hydropathy
        f["net_charge"] = sum(counts.get(a, 0) * c for a, c in CHARGE.items())
        f["net_charge_per_res"] = f["net_charge"] / n
        for name, members in GROUPS.items():
            f[name] = sum(counts[a] for a in members if a in counts) / known
        for a in AA:
            f[f"aa_{a}"] = counts[a] / known
        rows.append(f)
    return pd.DataFrame(rows)


def cv_r2(X, y, label, alphas=np.logspace(-1, 4, 25)):
    """Cross-validated R^2: how much of alpha can we predict for proteins the
    regression never saw? The in-sample number is inflated because a model with
    many knobs fits noise, so both are printed and only the CV one is believed.

    Two things here were bugs on the first attempt:
      - the ridge penalty started at 1e-3, which is no regularisation at all for
        a near-singular design matrix. It now starts at 1e-1.
      - standardisation was fitted on ALL the data before splitting, leaking fold
        statistics. A Pipeline refits the scaler inside each fold.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    est = RidgeCV(alphas=alphas) if X.shape[1] > 1 else LinearRegression()
    model = make_pipeline(StandardScaler(), est)
    cv = KFold(5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    insample = model.fit(X, y).score(X, y)
    cond = np.linalg.cond(StandardScaler().fit_transform(X)) if X.shape[1] > 1 else 1.0
    flag = "  <-- SINGULAR, distrust" if cond > 1e10 else ""
    print(f"  {label:34s} in-sample R2 {insample:6.3f}   "
          f"cross-validated R2 {scores.mean():6.3f}   cond {cond:8.1e}{flag}")
    return scores.mean()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="0 = use the full prot_pooler.npy table; N = the old 128-protein npz")
    args = ap.parse_args()

    proteins = pd.read_parquet(ROOT / "data" / "processed" / "proteins.parquet")
    full = ROOT / "data" / "processed" / "prot_pooler.npy"

    if full.exists() and not args.sample:
        # Full table from the Colab run: row i corresponds to prot_id i.
        pp = np.load(full).astype(np.float64)
        seqs = proteins.sort_values("prot_id")["seq"].tolist()
        assert len(seqs) == len(pp), (len(seqs), len(pp))
        print(f"using the FULL table: {len(pp):,} proteins")
    else:
        # Fall back to the 128-protein diagnostic sample.
        d = np.load(ROOT / "results" / "pooler_diagnostic.npz")
        pp = d["prot_pooler"].astype(np.float64)
        rng = np.random.default_rng(0)
        pr = proteins[proteins.seq_len <= 1200]
        pr = pr.iloc[rng.choice(len(pr), 128, replace=False)]
        seqs = pr["seq"].tolist()
        assert len(seqs) == len(pp)
        print(f"using the 128-protein diagnostic sample")

    # alpha = each protein's score on the single direction the representation uses.
    X = pp - pp.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    alpha = U[:, 0] * S[0]
    # Sign of a principal component is arbitrary; fix it so longer = larger for
    # readability. This does not change any r^2.
    if np.corrcoef(alpha, [len(s) for s in seqs])[0, 1] < 0:
        alpha = -alpha
    print(f"{len(seqs)} proteins, {min(map(len,seqs))}-{max(map(len,seqs))} aa")
    print(f"component 1 holds {S[0]**2/(S**2).sum()*100:.4f}% of the variance "
          f"-> alpha IS the protein representation\n")

    F = features(seqs)

    print("=== single features, correlation with alpha ===")
    singles = []
    for col in F.columns:
        v = F[col].to_numpy(dtype=np.float64)
        if np.ptp(v) == 0:
            continue
        r = float(np.corrcoef(alpha, v)[0, 1])
        singles.append((abs(r), col, r, r * r))
    singles.sort(reverse=True)
    print(f"  {'feature':24s} {'r':>8s} {'r2':>8s}")
    for _, col, r, r2 in singles[:14]:
        print(f"  {col:24s} {r:>+8.3f} {r2:>8.3f}")
    print(f"  ... {len(singles)-14} more, all |r| < {singles[14][0]:.3f}" if len(singles) > 14 else "")

    print("\n=== how much of alpha can each FEATURE SET explain? ===")
    cv_r2(F[["length"]], alpha, "length alone")
    cv_r2(F[["length", "log_length", "mol_weight"]], alpha, "size only (3 features)")
    cv_r2(F[["gravy", "net_charge_per_res", "frac_hydrophobic",
             "frac_aromatic", "frac_charged", "frac_polar"]], alpha,
          "physicochemical only (6)")
    cv_r2(F[[f"aa_{a}" for a in AA]], alpha, "amino acid composition (20)")
    cv_r2(F[[f"aa_{a}" for a in AA] + ["length", "log_length"]], alpha,
          "composition + size (22)")
    full = cv_r2(F, alpha, f"everything ({F.shape[1]} features)")

    print(f"""
How to read this
----------------
alpha is the ONE number PLAPT gets to describe a protein. If simple sequence
statistics predict alpha well, then PLAPT's protein representation contains
nothing a first-year biochemistry student could not compute by hand -- no fold,
no pocket, no evolutionary context, just bulk composition.

Residual unexplained: {1-full:.1%}. That remainder is the only place anything
structurally interesting could be hiding, and it is an upper bound: a nonlinear
model might explain more of alpha from the same features, pushing it lower.""")


if __name__ == "__main__":
    main()
