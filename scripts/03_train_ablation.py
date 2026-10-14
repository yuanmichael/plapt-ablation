"""
Stage 3: the ablation. Train PLAPT's exact prediction head repeatedly, changing
only WHICH INFORMATION reaches it, and see how much each side of the pair is
actually worth.

Run:
    .venv/Scripts/python.exe -I scripts/03_train_ablation.py --seeds 3

The architecture below is not a reimplementation from the paper's prose -- it was
read directly out of the released affinity_predictor0734-seed2101.onnx graph,
down to dropout=0.2 and the BatchNorm epsilon of 1e-3 (a Wolfram Language
default, which is how we know the original was trained in Mathematica).

The seven arms
--------------
  full          real protein + real ligand            <- reproduces PLAPT
  mol_zero      protein input replaced with zeros      } three different ways
  mol_shuffle   protein vectors randomly re-paired     } to ask "does the
  mol_cut       protein branch deleted entirely        } protein matter?"
  prot_zero     ligand input replaced with zeros       } and the mirror image
  prot_shuffle  ligand vectors randomly re-paired      }
  prot_cut      ligand branch deleted entirely         }

Why three flavours of "remove it" instead of one? Because they are not the same
experiment, and if they disagree that disagreement is the finding:

  zero    keeps all 1,480,320 parameters, so capacity is held constant and only
          information changes. But zeros are off-distribution -- the branch now
          outputs a constant, which is a cleaner signal than real data ever is.
  shuffle keeps the input distribution EXACTLY (same vectors, same norms, same
          covariance) and destroys only the correspondence to the label. This is
          the strictest control, and the one to trust if they diverge.
  cut     removes the parameters too, which answers the engineering question
          ("could I ship a smaller model?") but changes two variables at once.

Read every number against the zero-parameter baselines printed at the end.
A learned model that cannot beat a lookup table has not learned anything useful.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

PROT_DIM, MOL_DIM = 1024, 768   # overridden at run time from the cache
BRANCH, HIDDEN, BOTTLENECK = 512, 512, 64
DROPOUT, BN_EPS = 0.2, 1e-3   # both recovered from the ONNX graph


class PlaptHead(nn.Module):
    """PLAPT's prediction module, transcribed from the released ONNX graph.

    ONNX op order was:
      Slice, Slice, Gemm, Gemm, Relu, Relu, Concat, BatchNormalization,
      Gemm, Relu, Dropout, Gemm, Relu, Gemm

    Passing use_prot=False / use_mol=False deletes that branch (the "cut" arms).
    """

    def __init__(self, use_prot=True, use_mol=True):
        super().__init__()
        self.use_prot, self.use_mol = use_prot, use_mol
        if not (use_prot or use_mol):
            raise ValueError("need at least one branch")
        self.prot_branch = nn.Linear(PROT_DIM, BRANCH) if use_prot else None
        self.mol_branch = nn.Linear(MOL_DIM, BRANCH) if use_mol else None
        merged = BRANCH * (int(use_prot) + int(use_mol))
        self.bn = nn.BatchNorm1d(merged, eps=BN_EPS)
        self.l1 = nn.Linear(merged, HIDDEN)
        self.drop = nn.Dropout(DROPOUT)
        self.l2 = nn.Linear(HIDDEN, BOTTLENECK)
        self.out = nn.Linear(BOTTLENECK, 1)
        self.relu = nn.ReLU()

    def forward(self, p, m):
        parts = []
        if self.use_prot:
            parts.append(self.relu(self.prot_branch(p)))
        if self.use_mol:
            parts.append(self.relu(self.mol_branch(m)))
        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        x = self.bn(x)
        x = self.drop(self.relu(self.l1(x)))
        x = self.relu(self.l2(x))
        return self.out(x).squeeze(1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------- metrics
def metrics(pred_log, true_log):
    """Always compute metrics in float64 -- see the float32 std() trap in stage 1."""
    p = np.asarray(pred_log, dtype=np.float64)
    t = np.asarray(true_log, dtype=np.float64)
    if len(p) < 3 or np.ptp(p) == 0:
        return {"n": len(p), "r": 0.0, "spearman": 0.0,
                "rmse": float(np.sqrt(np.mean((p - t) ** 2))) if len(p) else float("nan"),
                "mae": float(np.mean(np.abs(p - t))) if len(p) else float("nan"),
                "r2": float("nan")}
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return {
        "n": int(len(p)),
        "r": float(np.corrcoef(p, t)[0, 1]),
        "spearman": float(stats.spearmanr(p, t).statistic),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "mae": float(np.mean(np.abs(p - t))),
        # R^2 as 1 - SSres/SStot. Note this is NOT r**2: it penalises bias and
        # bad scale, so a model can have r=0.7 and a NEGATIVE R^2.
        "r2": 1.0 - ss_res / ss_tot,
    }


def run_arm(arm, data, seed, args):
    """Train one arm from scratch and evaluate it on the held-out test split."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    prot_emb, mol_emb, splits, y_mean, y_std = data

    use_prot = arm["prot"] != "none"
    use_mol = arm["mol"] != "none"
    model = PlaptHead(use_prot, use_mol)

    def view(split_name, for_train):
        """Build the (prot_idx, mol_idx, y) tensors this arm sees for a split.

        The embeddings stay as two compact matrices and we index into them per
        batch, so 'shuffle' is a permutation of an index array rather than a
        600 MB copy of the data, and 'zero' is a multiply-by-zero mask.
        """
        pid, mid, y = splits[split_name]
        pid, mid = pid.clone(), mid.clone()
        if arm["prot"] == "shuffle":
            pid = pid[torch.from_numpy(rng.permutation(len(pid)))]
        if arm["mol"] == "shuffle":
            mid = mid[torch.from_numpy(rng.permutation(len(mid)))]
        return pid, mid, y

    views = {s: view(s, s == "train") for s in ("train", "dev", "test")}

    def gather(pid, mid, sl):
        p = prot_emb[pid[sl]] if use_prot else None
        m = mol_emb[mid[sl]] if use_mol else None
        if arm["prot"] == "zero" and p is not None:
            p = torch.zeros_like(p)
        if arm["mol"] == "zero" and m is not None:
            m = torch.zeros_like(m)
        return p, m

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    pid_tr, mid_tr, y_tr = views["train"]
    n_tr = len(y_tr)

    best_dev, best_state, bad_epochs = -np.inf, None, 0

    @torch.no_grad()
    def predict(split_name):
        model.eval()
        pid, mid, y = views[split_name]
        outs = []
        for i in range(0, len(y), 8192):
            sl = slice(i, i + 8192)
            outs.append(model(*gather(pid, mid, sl)))
        pred_scaled = torch.cat(outs).numpy()
        # back to log10 units so numbers are comparable to the paper
        return pred_scaled * y_std + y_mean, y.numpy() * y_std + y_mean

    if getattr(args, "log_epochs", False):
        print(f"  >>> {arm['name']} seed {seed} starting", flush=True)

    for epoch in range(args.max_epochs):
        model.train()
        perm = torch.randperm(n_tr)
        run_loss, n_batch = 0.0, 0
        for i in range(0, n_tr, args.batch_size):
            idx = perm[i:i + args.batch_size]
            # BatchNorm cannot compute batch statistics from a single row, and a
            # trailing batch of size 1 would crash the epoch. Skip it.
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(*gather(pid_tr, mid_tr, idx)), y_tr[idx])
            loss.backward()
            opt.step()
            run_loss += float(loss.detach())
            n_batch += 1

        dev_r = metrics(*predict("dev"))["r"]
        if dev_r > best_dev + 1e-5:
            best_dev, bad_epochs = dev_r, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        # One line per epoch, for scripts/watch_training.py. Train loss falls more
        # or less forever; dev R rises then flattens. Watching those two diverge is
        # watching overfitting happen.
        if getattr(args, "log_epochs", False):
            print(f"    ep {epoch+1:3d}  train_loss {run_loss/max(n_batch,1):.4f}  "
                  f"dev R {dev_r:.4f}  best {best_dev:.4f}  "
                  f"patience {bad_epochs}/{args.patience}", flush=True)

        if bad_epochs >= args.patience:
            break

    # Restore the best-on-dev weights, then touch the test split exactly once.
    model.load_state_dict(best_state)
    pred_test, true_test = predict("test")
    res = metrics(pred_test, true_test)
    res.update(arm=arm["name"], seed=seed, dev_r=float(best_dev),
               epochs=epoch + 1, n_params=model.n_params())

    # The subgroup that actually tests generalisation: test rows whose protein
    # the model never saw during training.
    seen = splits["seen_prot_mask"]
    for label, mask in (("seen_prot", seen), ("unseen_prot", ~seen)):
        # Always define the columns, even when the subgroup is empty. Under a
        # cluster split EVERY test protein is unseen, so "seen_prot" has zero
        # rows -- and a missing column makes the summary aggregation raise
        # KeyError after all the training has already finished.
        if mask.sum() >= 3:
            sub = metrics(pred_test[mask], true_test[mask])
            res[f"r_{label}"] = sub["r"]
            res[f"rmse_{label}"] = sub["rmse"]
            res[f"n_{label}"] = sub["n"]
        else:
            res[f"r_{label}"] = float("nan")
            res[f"rmse_{label}"] = float("nan")
            res[f"n_{label}"] = int(mask.sum())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3,
                    help="repeats per arm; >1 gives you error bars, which you need "
                         "before claiming any arm beat any other")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--dev-frac", type=float, default=0.1,
                    help="carved out of train, used ONLY for early stopping")
    ap.add_argument("--pooling", choices=["pooler", "mean"], default="pooler",
                    help="pooler = what PLAPT uses; mean = the free-upgrade variant")
    ap.add_argument("--prot-tag", default="",
                    help="which protein encoder cache to read, e.g. _esm650. Empty "
                         "means prot_{pooling}.npy (the original ProtBERT run)")
    ap.add_argument("--arms", default="", help="comma-separated subset of arm names")
    ap.add_argument("--proc", default=None,
                    help="dir holding the parquet tables (default data/processed)")
    ap.add_argument("--emb", default=None,
                    help="dir holding the .npy caches, if separate from --proc. "
                         "Avoids having to co-locate a 3 GB cache with the tables.")
    ap.add_argument("--pairs", default="pairs.parquet",
                    help="which pair table to use, e.g. pairs_clustered.parquet")
    ap.add_argument("--meta", default="meta.json",
                    help="matching metadata (target scaling differs per split)")
    ap.add_argument("--train-n", type=int, default=0,
                    help="cap the training set at N pairs (0 = use all). Dev and "
                         "test are unaffected, so points are directly comparable. "
                         "Subsets are nested across values.")
    ap.add_argument("--tag", default="", help="suffix for the output files, so a rerun "
                                              "does not clobber an earlier one")
    ap.add_argument("--log-epochs", action="store_true",
                    help="print one line per epoch, so scripts/watch_training.py can "
                         "render a live loss / dev-R curve")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = auto)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    global PROC
    if args.proc:
        PROC = Path(args.proc)
    print(f"data: {PROC}")
    meta = json.loads((PROC / args.meta).read_text())
    y_mean, y_std = meta["y_mean"], meta["y_std"]
    pairs = pd.read_parquet(PROC / args.pairs)

    suffix = "pooler" if args.pooling == "pooler" else "mean"
    # --prot-tag selects which protein encoder's cache to use, e.g. "_esm650".
    # The ligand cache is shared: swapping the protein encoder does not change it.
    emb = Path(args.emb) if args.emb else PROC
    prot_np = np.load(emb / f"prot_{suffix}{args.prot_tag}.npy")
    mol_np = np.load(emb / f"mol_{suffix}.npy")
    print(f"embeddings: prot {prot_np.shape} {prot_np.dtype}, "
          f"mol {mol_np.shape} {mol_np.dtype}  (pooling={args.pooling})")
    # fp16 on disk to keep the files small; fp32 in RAM because training in fp16
    # on CPU is slower, not faster, and the gradients would be garbage.
    prot_emb = torch.from_numpy(prot_np.astype(np.float32))
    mol_emb = torch.from_numpy(mol_np.astype(np.float32))
    # Different protein encoders have different widths (ProtBERT 1024, ESM-2 650M
    # 1280, ESM-2 150M 640), so the branch input size follows the cache.
    globals()["PROT_DIM"] = int(prot_np.shape[1])
    globals()["MOL_DIM"] = int(mol_np.shape[1])

    # ------------------------------------------------- three splits, not two
    # Stage 1 made PLAPT's 90/10 train/val split. But we will consult the
    # validation set every epoch to decide when to stop training -- which makes
    # it a DEV set, and a number chosen as the maximum over 60 epochs is
    # optimistically biased. So we carve a dev set out of train and keep the
    # original 10% as a test set that each arm is evaluated on exactly once.
    #
    # This is the difference between PLAPT's reported 0.683 and an honest number.
    #
    # If the pairs file already carries a "dev" split -- which pairs_clustered
    # .parquet does -- use it verbatim instead of carving one out. Under a
    # cluster split, carving dev randomly out of train would put homologs of the
    # training proteins in dev, and early stopping would then be tuned against
    # the very leakage the split exists to remove.
    sv = set(pairs["split"].unique())
    if "dev" in sv:
        fit_idx = np.flatnonzero((pairs["split"] == "train").to_numpy())
        dev_idx = np.flatnonzero((pairs["split"] == "dev").to_numpy())
        test_idx = np.flatnonzero((pairs["split"] == "test").to_numpy())
        print("using the pre-specified train/dev/test splits from the pairs file")
    else:
        is_train = (pairs["split"] == "train").to_numpy()
        tr_idx = np.flatnonzero(is_train)
        rng = np.random.default_rng(meta["seed"])
        shuf = rng.permutation(len(tr_idx))
        n_dev = int(len(tr_idx) * args.dev_frac)
        dev_idx = tr_idx[shuf[:n_dev]]
        fit_idx = tr_idx[shuf[n_dev:]]
        test_idx = np.flatnonzero(~is_train)

    # ------------------------------------------------------- learning curve
    # --train-n truncates the TRAINING set only. Dev and test are untouched, so
    # every point on the curve is scored on the identical 20,000 test pairs and
    # stopped on the identical 18,000 dev pairs. Without that, each point would
    # differ in training size AND test composition, and the curve would be
    # uninterpretable.
    #
    # Because `shuf` is seeded from meta["seed"], the subsets are NESTED:
    # the 25k set is a subset of the 50k set, and so on. That removes
    # sample-composition noise between adjacent points -- the only thing that
    # changes from one point to the next is that rows were added.
    #
    # The lookup baselines below are built from fit_idx too, so they shrink with
    # the model. That is the control that matters: if the model's curve and the
    # memorisation curve have the same shape, extra data is buying recall, not
    # understanding.
    if args.train_n and args.train_n < len(fit_idx):
        fit_idx = fit_idx[:args.train_n]

    def pack(idx):
        return (torch.from_numpy(pairs["prot_id"].to_numpy()[idx].astype(np.int64)),
                torch.from_numpy(pairs["mol_id"].to_numpy()[idx].astype(np.int64)),
                torch.from_numpy(pairs["y_scaled"].to_numpy()[idx].astype(np.float32)))

    splits = {"train": pack(fit_idx), "dev": pack(dev_idx), "test": pack(test_idx)}
    train_prots = set(pairs["prot_id"].to_numpy()[fit_idx].tolist())
    splits["seen_prot_mask"] = np.array(
        [p in train_prots for p in pairs["prot_id"].to_numpy()[test_idx]])
    print(f"train {len(fit_idx):,}  dev {len(dev_idx):,}  test {len(test_idx):,}"
          f"   |  test rows with an unseen protein: {(~splits['seen_prot_mask']).sum():,}")

    data = (prot_emb, mol_emb, splits, y_mean, y_std)

    # ------------------------------------------------------- lookup baselines
    print("\n--- zero-parameter baselines, same test split ---")
    y_log = pairs["y"].to_numpy(dtype=np.float64)
    fit_y, test_y = y_log[fit_idx], y_log[test_idx]
    gm = fit_y.mean()
    base = {}
    for name, col in (("protein_mean_lookup", "prot_id"), ("ligand_mean_lookup", "mol_id")):
        key = pairs[col].to_numpy()
        tbl = pd.Series(fit_y).groupby(key[fit_idx]).mean()
        pred = pd.Series(key[test_idx]).map(tbl).fillna(gm).to_numpy()
        base[name] = metrics(pred, test_y)
    base["global_mean"] = metrics(np.full(len(test_y), gm), test_y)
    for k, v in base.items():
        print(f"  {k:24s} R = {v['r']:6.4f}  RMSE = {v['rmse']:.4f}  "
              f"Spearman = {v['spearman']:6.4f}  R2 = {v['r2']:7.4f}")

    # ------------------------------------------------------------------ arms
    all_arms = [
        {"name": "full",         "prot": "real",    "mol": "real"},
        {"name": "mol_zero",     "prot": "zero",    "mol": "real"},
        {"name": "mol_shuffle",  "prot": "shuffle", "mol": "real"},
        {"name": "mol_cut",      "prot": "none",    "mol": "real"},
        {"name": "prot_zero",    "prot": "real",    "mol": "zero"},
        {"name": "prot_shuffle", "prot": "real",    "mol": "shuffle"},
        {"name": "prot_cut",     "prot": "real",    "mol": "none"},
    ]
    if args.arms:
        want = {a.strip() for a in args.arms.split(",")}
        all_arms = [a for a in all_arms if a["name"] in want]

    rows = []
    t0 = time.time()
    for arm in all_arms:
        for seed in range(args.seeds):
            t1 = time.time()
            r = run_arm(arm, data, seed, args)
            rows.append(r)
            print(f"  {arm['name']:13s} seed {seed}  "
                  f"test R = {r['r']:6.4f}  RMSE = {r['rmse']:.4f}  "
                  f"dev R = {r['dev_r']:6.4f}  "
                  f"{r['epochs']:2d} ep  {r['n_params']:>9,} params  "
                  f"{time.time()-t1:5.1f}s", flush=True)

    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    stem = f"{args.pooling}{args.tag}"
    df.to_csv(ROOT / "results" / f"ablation_{stem}.csv", index=False)
    json.dump(base, open(ROOT / "results" / f"baselines_{stem}.json", "w"), indent=2)

    # --------------------------------------------------------------- summary
    print(f"\n{'='*86}\nABLATION SUMMARY  (pooling={args.pooling}, "
          f"{args.seeds} seed(s), max_epochs={args.max_epochs}, "
          f"{time.time()-t0:.0f}s total)\n{'='*86}")
    agg = df.groupby("arm").agg(
        r_mean=("r", "mean"), r_std=("r", "std"),
        rmse_mean=("rmse", "mean"), spearman=("spearman", "mean"),
        r2=("r2", "mean"), params=("n_params", "first"),
        r_seen=("r_seen_prot", "mean"), r_unseen=("r_unseen_prot", "mean"),
    ).reindex([a["name"] for a in all_arms])
    print(f"{'arm':14s} {'test R':>16s} {'RMSE':>7s} {'Spear':>7s} {'R2':>8s} "
          f"{'params':>10s} {'R|seen':>8s} {'R|unseen':>9s}")
    for name, r in agg.iterrows():
        sd = "" if np.isnan(r.r_std) else f" ±{r.r_std:.4f}"
        print(f"{name:14s} {r.r_mean:9.4f}{sd:>7s} {r.rmse_mean:7.4f} "
              f"{r.spearman:7.4f} {r.r2:8.4f} {int(r.params):10,} "
              f"{r.r_seen:8.4f} {r.r_unseen:9.4f}")
    print(f"\nfor reference: protein lookup table R = {base['protein_mean_lookup']['r']:.4f}, "
          f"ligand lookup R = {base['ligand_mean_lookup']['r']:.4f}")
    print(f"               PLAPT's reported validation R = 0.683")
    print(f"\nwrote -> results/ablation_{stem}.csv")


if __name__ == "__main__":
    main()
