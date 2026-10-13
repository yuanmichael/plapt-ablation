"""
Stage 6: replace PLAPT's concatenation with cross-attention between ligand atoms
and protein residues.

Run on Colab (L4), after 02_embed.py --save-residues:
    !python 06_cross_attention.py --data . --emb ./emb --prot-tag _esm650

Smoke-test locally on CPU (minutes, meaningless numbers, proves the plumbing):
    .venv/Scripts/python.exe -I scripts/06_cross_attention.py --smoke

What this changes, and what it deliberately does not
----------------------------------------------------
PLAPT computes a protein vector and a ligand vector independently, concatenates
them, and runs an MLP. The protein vector is a mean over residues, so residue
identity is gone before the ligand is ever consulted. There is no tensor in that
design with a cell for "ligand atom j touches residue i", which is what a binding
contact is.

This script keeps the frozen encoders, the split, the target scaling and the
final MLP's shape identical, and changes exactly one thing: how the protein's
per-residue grid is collapsed.

    PLAPT:  pooled_protein = (1/L) * sum_i residue_i          uniform weights
    here:   context_j      = sum_i w_ij * V(residue_i)        w chosen per atom

Mean pooling is the special case w_ij = 1/L. So this is not a new kind of
operation; it is the same averaging with the weights computed from the ligand
instead of frozen at uniform.

The arms
--------
  full          learned cross-attention, one query per ligand atom
  pooled_query  one query, from the mean-pooled ligand, instead of one per atom.
                The cheap variant: it can learn "this molecule cares about this
                region" but has no cell for a specific atom-residue pair.
  prot_zero     protein grid zeroed -> ligand-only ceiling
  mol_zero      ligand grid zeroed. Note this degenerates gracefully: with zero
                queries every score is equal, so attention becomes uniform and
                the model sees the mean-pooled protein. The protein-only ceiling.

Why the ligand cache is built here instead of in stage 2
--------------------------------------------------------
ChemBERTa is frozen, so its per-atom output is identical every epoch. Computing
it live would repeat ~3 minutes of GPU work 40 times per run. But caching it in
stage 2 would mean carrying an 11 GB artifact around, and our ligands repeat only
1.1 times across the dataset, so there is nothing to amortise WITHIN an epoch.

The resolution is to cache it here, once per invocation, on local disk: one
3-minute pass that is then reused by every epoch, every arm and every seed in the
run, and deleted with the machine. Local disk is large and free; Drive quota is
the thing that is scarce.

Memory placement
----------------
On a 24 GB L4 the two caches do not both fit alongside activations:

    protein residues, ESM-2 650M   3.81M x 1280 fp16  =  9.8 GB
    ligand atoms,     ChemBERTa    7.3M  x  768 fp16  = 11.2 GB

Proteins go to the GPU and ligands stay on disk, because proteins are reused 23x
as often and their rows are ~17x larger (1280 x ~400 residues vs 768 x ~40
tokens). A ligand batch is under 4 MB to transfer; a protein batch would be 167.
Both placements are overridable.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
MOL_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
MOL_MAX = 278          # PLAPT's setting
HIDDEN, BOTTLENECK = 512, 512   # PLAPT's head shape, kept for comparability
DROPOUT, BN_EPS = 0.2, 1e-3


# --------------------------------------------------------------------- metrics
def metrics(pred, true):
    """float64 throughout. Summing 20k float32 terms leaves enough round-off
    that np.std() of a constant array returns 9.5e-07 instead of 0, and then
    corrcoef silently returns nan. np.ptp is exact."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    r = 0.0 if np.ptp(pred) == 0 else float(np.corrcoef(pred, true)[0, 1])
    sp = 0.0 if np.ptp(pred) == 0 else float(
        np.corrcoef(pd.Series(pred).rank(), pd.Series(true).rank())[0, 1])
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ss = float(((true - true.mean()) ** 2).sum())
    return {"r": r, "spearman": sp, "rmse": rmse, "mae": float(np.abs(pred - true).mean()),
            "r2": float(1 - ((pred - true) ** 2).sum() / ss) if ss > 0 else float("nan")}


# ------------------------------------------------------------- ligand caching
def token_counts(texts, tokenizer, max_length, struct_ids, batch=1024):
    """Exact non-special token count per string.

    We cannot predict this from len(smiles): ChemBERTa uses a BPE vocabulary, so
    one token covers a variable number of characters. (Protein tokenizers happen
    to be one-token-per-residue, but relying on that here would be a latent bug
    the first time someone swaps the ligand encoder.) Tokenising twice is cheap
    -- seconds for 182k molecules -- and it makes the preallocated offsets exact
    rather than assumed.
    """
    counts = np.zeros(len(texts), dtype=np.int64)
    for i in range(0, len(texts), batch):
        enc = tokenizer(texts[i:i + batch], truncation=True, max_length=max_length)
        for j, ids in enumerate(enc["input_ids"]):
            counts[i + j] = sum(1 for t in ids if t not in struct_ids)
    return counts


@torch.no_grad()
def build_mol_cache(smiles, out_dir, device, budget=65_536):
    """Per-atom ChemBERTa embeddings, ragged, written once to local disk.

    Reuses stage 2's two tricks: length-sorted dynamic batching (so a batch is
    internally uniform and we do not spend the compute on padding), and a token
    budget rather than a row count (so memory stays flat).
    """
    from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizerFast

    res_p = out_dir / "mol_residues.npy"
    off_p = out_dir / "mol_offsets.npy"
    meta_p = out_dir / "mol_residues_meta.json"
    if res_p.exists() and off_p.exists() and meta_p.exists():
        m = json.loads(meta_p.read_text())
        if m.get("n_ligands") == len(smiles) and m.get("model") == MOL_MODEL:
            print(f"  reusing existing ligand cache ({res_p.stat().st_size/1e9:.2f} GB)")
            return res_p, off_p
        print("  existing ligand cache does not match this ligand table; rebuilding")

    try:
        tok = AutoTokenizer.from_pretrained(MOL_MODEL)
    except Exception:
        tok = RobertaTokenizerFast.from_pretrained(MOL_MODEL)
    try:
        model = AutoModel.from_pretrained(MOL_MODEL)
    except Exception:
        model = RobertaModel.from_pretrained(MOL_MODEL)
    model = model.to(device).eval()

    struct = {getattr(tok, a + "_token_id", None)
              for a in ("cls", "bos", "sep", "eos", "pad")}
    struct = {s for s in struct if s is not None}

    print("  counting tokens...", flush=True)
    counts = token_counts(smiles, tok, MOL_MAX, struct)
    off = np.zeros(len(smiles) + 1, dtype=np.int64)
    np.cumsum(counts, out=off[1:])
    total, dim = int(off[-1]), int(model.config.hidden_size)
    print(f"  ligand atoms: {total:,} x {dim} @ fp16 = {total*dim*2/1e9:.2f} GB"
          f"   (mean {total/len(smiles):.1f} tokens/molecule)", flush=True)

    arr = np.lib.format.open_memmap(res_p, mode="w+", dtype=np.float16,
                                    shape=(total, dim))
    order = np.argsort(counts, kind="stable")
    t0, done, i = time.time(), 0, 0
    while i < len(order):
        longest = int(counts[order[i]]) + 2
        j = i
        while j < len(order):
            cand = max(longest, int(counts[order[j]]) + 2)
            if (j - i + 1) * cand > budget and j > i:
                break
            longest = cand
            j += 1
        idx = order[i:j]
        enc = tok([smiles[k] for k in idx], padding=True, truncation=True,
                  max_length=MOL_MAX, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            hs = model(**enc).last_hidden_state
        hs = hs.float()
        keep = enc["attention_mask"].bool()
        for sid in struct:
            keep &= (enc["input_ids"] != sid)
        for b, k in enumerate(idx):
            rows = hs[b][keep[b]].cpu().numpy().astype(np.float16)
            lo, hi = int(off[k]), int(off[k + 1])
            if rows.shape[0] != hi - lo:
                raise RuntimeError(
                    f"ligand {k}: got {rows.shape[0]} token rows, reserved {hi-lo}. "
                    f"token_counts() and the padded forward pass disagree.")
            arr[lo:hi] = rows
        done += len(idx)
        i = j
        if done % 20000 < len(idx) or i >= len(order):
            el = time.time() - t0
            print(f"  [ligand] {done:,}/{len(order):,}  {done/max(el,1e-9):7.1f}/s"
                  f"  eta {(len(order)-done)/max(done/max(el,1e-9),1e-9)/60:5.1f}m",
                  flush=True)
    arr.flush()
    np.save(off_p, off)
    meta_p.write_text(json.dumps(
        {"model": MOL_MODEL, "n_ligands": len(smiles), "n_tokens": total,
         "dim": dim, "max_length": MOL_MAX, "dtype": "float16"}, indent=2))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res_p, off_p


# ----------------------------------------------------------------- the model
class CrossAttentionHead(nn.Module):
    """Ligand atoms attend over protein residues, then pool.

    The ordering is the whole point. PLAPT pools each side and then combines, so
    the interaction is never computable. Here the interaction is computed first
    and pooled afterwards -- and pooling after is not merely tolerable, it is
    what the physics asks for, since binding free energy is approximately a sum
    over contacts.
    """

    def __init__(self, prot_dim, mol_dim, d=256, heads=4):
        super().__init__()
        assert d % heads == 0, "--attn-dim must be divisible by --heads"
        self.d, self.heads, self.dh = d, heads, d // heads
        self.q = nn.Linear(mol_dim, d)      # what is this atom looking for?
        self.k = nn.Linear(prot_dim, d)     # what does this residue advertise?
        self.v = nn.Linear(prot_dim, d)     # what does it contribute if chosen?
        self.mol_proj = nn.Linear(mol_dim, d)   # the atom's own identity
        self.norm = nn.LayerNorm(2 * d)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, HIDDEN), nn.BatchNorm1d(HIDDEN, eps=BN_EPS),
            nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, BOTTLENECK), nn.BatchNorm1d(BOTTLENECK, eps=BN_EPS),
            nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(BOTTLENECK, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, prot, prot_mask, mol, mol_mask, mode="full"):
        """prot B x L x Dp, mol B x N x Dm, masks are bool (True = real)."""
        B, L, _ = prot.shape
        if mode == "prot_zero":
            prot = torch.zeros_like(prot)
        if mode == "mol_zero":
            mol = torch.zeros_like(mol)

        if mode == "pooled_query":
            # One query from the mean-pooled ligand: the cheap variant.
            w = mol_mask.unsqueeze(-1).float()
            q_src = (mol * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1)
            q_mask = torch.ones(B, 1, dtype=torch.bool, device=mol.device)
        else:
            q_src, q_mask = mol, mol_mask
        N = q_src.shape[1]

        Q = self.q(q_src).view(B, N, self.heads, self.dh).transpose(1, 2)
        K = self.k(prot).view(B, L, self.heads, self.dh).transpose(1, 2)
        V = self.v(prot).view(B, L, self.heads, self.dh).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dh)

        # Never let an atom attend to padding. Every protein has >= 9 residues so
        # no row is fully masked, but clamp the denominator anyway rather than
        # trusting that invariant.
        scores = scores.masked_fill(~prot_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        ctx = (attn @ V).transpose(1, 2).reshape(B, N, self.d)

        # what I am, and what I am touching -- this pair IS the contact
        per_atom = torch.cat([self.mol_proj(q_src), ctx], dim=-1)
        per_atom = self.norm(per_atom)

        m = q_mask.unsqueeze(-1).float()
        pooled = (per_atom * m).sum(1) / m.sum(1).clamp(min=1)
        return self.mlp(pooled).squeeze(-1)


# ----------------------------------------------------------------- batching
class Batcher:
    """Assembles padded (grid, mask) pairs for a list of pair-row indices."""

    def __init__(self, prot_res, prot_off, mol_res, mol_off, pairs, device,
                 prot_on_gpu=True, mol_ram=False):
        self.device = device
        self.prot_off = torch.as_tensor(prot_off, dtype=torch.long, device=device)
        self.prot_len = torch.as_tensor(np.diff(prot_off), dtype=torch.long, device=device)
        self.mol_off, self.mol_len = mol_off, np.diff(mol_off)
        # The ligand cache is read ~162k times per epoch in small random slices,
        # so on a memmap this is disk-seek bound and can dominate the step time.
        # 11 GB fits in a high-RAM Colab runtime (51 GB) but not a standard one
        # (12.7 GB), hence the flag rather than doing it unconditionally.
        if mol_ram:
            print(f"  loading ligand cache into host RAM "
                  f"({mol_res.shape[0]*mol_res.shape[1]*2/1e9:.2f} GB)...", flush=True)
            self.mol_res = np.array(mol_res, dtype=np.float16)
        else:
            self.mol_res = mol_res                  # memmap, stays on disk
        self.n_res = prot_res.shape[0]
        if prot_on_gpu:
            # Upload in chunks. The obvious one-liner --
            #   torch.from_numpy(np.asarray(memmap)).to(device)
            # -- materialises the whole 9.8 GB in HOST ram first, which OOMs a
            # standard 12.7 GB Colab runtime before it ever reaches the GPU.
            # Streaming keeps the host-side peak at one chunk.
            self.prot_res = torch.empty(prot_res.shape, dtype=torch.float16,
                                        device=device)
            CH = 100_000                      # ~256 MB per chunk at 1280 dims
            for i in range(0, prot_res.shape[0], CH):
                blk = np.array(prot_res[i:i + CH], dtype=np.float16)  # writable copy
                self.prot_res[i:i + CH] = torch.from_numpy(blk).to(device)
        else:
            self.prot_res = torch.from_numpy(np.array(prot_res, dtype=np.float16))
        self.prot_on_gpu = prot_on_gpu
        self.pid = pairs["prot_id"].to_numpy()
        self.mid = pairs["mol_id"].to_numpy()
        self.y = pairs["y_scaled"].to_numpy(dtype=np.float32)
        self.prot_dim = prot_res.shape[1]
        self.mol_dim = mol_res.shape[1]

    def batch(self, rows):
        dev = self.device
        pid = torch.as_tensor(self.pid[rows], dtype=torch.long, device=dev)
        plen = self.prot_len[pid]
        Lmax = int(plen.max().item())
        ar = torch.arange(Lmax, device=dev)
        # Gather each protein's slice out of the one long concatenated array.
        # Masked positions read a clamped (valid but wrong) row; the mask makes
        # them unreachable, which is cheaper than a scatter.
        idx = (self.prot_off[pid].unsqueeze(1) + ar.unsqueeze(0)).clamp(max=self.n_res - 1)
        prot = self.prot_res[idx if self.prot_on_gpu else idx.cpu()]
        if not self.prot_on_gpu:
            prot = prot.to(dev, non_blocking=True)
        prot_mask = ar.unsqueeze(0) < plen.unsqueeze(1)

        mids = self.mid[rows]
        mlen = self.mol_len[mids]
        Nmax = int(max(mlen.max(), 1))
        buf = np.zeros((len(rows), Nmax, self.mol_dim), dtype=np.float16)
        for b, (m, n) in enumerate(zip(mids, mlen)):
            if n:
                buf[b, :n] = self.mol_res[self.mol_off[m]:self.mol_off[m] + n]
        mol = torch.from_numpy(buf).to(dev, non_blocking=True)
        mol_mask = (torch.arange(Nmax, device=dev).unsqueeze(0)
                    < torch.as_tensor(mlen, device=dev).unsqueeze(1))

        y = torch.from_numpy(self.y[rows]).to(dev)
        return prot.float(), prot_mask, mol.float(), mol_mask, y


# ----------------------------------------------------------------- training
def run_arm(bt, splits, arm, seed, args, y_mean, y_std, seen_mask):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CrossAttentionHead(bt.prot_dim, bt.mol_dim, args.attn_dim, args.heads).to(bt.device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(seed)

    def evaluate(rows):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(rows), args.batch_size):
                p, pm, m, mm, _ = bt.batch(rows[i:i + args.batch_size])
                preds.append(model(p, pm, m, mm, arm).float().cpu().numpy())
        return np.concatenate(preds) * y_std + y_mean

    best, best_state, bad = np.inf, None, 0
    t0 = time.time()
    for ep in range(args.max_epochs):
        model.train()
        order = rng.permutation(splits["train"])
        tot = 0.0
        for i in range(0, len(order), args.batch_size):
            rows = order[i:i + args.batch_size]
            if len(rows) < 2:      # BatchNorm needs >1 row
                continue
            p, pm, m, mm, y = bt.batch(rows)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(p, pm, m, mm, arm), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(rows)
        dev_pred = evaluate(splits["dev"])
        dev_true = bt.y[splits["dev"]] * y_std + y_mean
        dev_mse = float(np.mean((dev_pred - dev_true) ** 2))
        improved = dev_mse < best - 1e-6
        if improved:
            best, bad = dev_mse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if args.log_epochs:
            # Print AFTER updating, so `patience` is the state you are actually
            # in. Printing it before made the counter lag by an epoch and read
            # as though a best-ever epoch had bad epochs against it.
            print(f"    ep {ep+1:3d}  train_loss {tot/len(order):.4f}  "
                  f"train_rmse {math.sqrt(tot/len(order))*y_std:.4f}  "
                  f"dev_rmse {math.sqrt(dev_mse):.4f}"
                  f"{'  *best' if improved else ''}  "
                  f"patience {bad}/{args.patience}", flush=True)
        if bad >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred = evaluate(splits["test"])
    test_true = bt.y[splits["test"]] * y_std + y_mean
    res = metrics(test_pred, test_true)
    res.update({
        "arm": arm, "seed": seed, "epochs": ep + 1, "params": n_par,
        "dev_r": metrics(evaluate(splits["dev"]),
                         bt.y[splits["dev"]] * y_std + y_mean)["r"],
        "r_seen": metrics(test_pred[seen_mask], test_true[seen_mask])["r"] if seen_mask.any() else float("nan"),
        "r_unseen": metrics(test_pred[~seen_mask], test_true[~seen_mask])["r"] if (~seen_mask).any() else float("nan"),
        "seconds": round(time.time() - t0, 1),
    })
    print(f"  {arm:13s} seed {seed}  test R = {res['r']:.4f}  RMSE = {res['rmse']:.4f}  "
          f"dev R = {res['dev_r']:.4f}  {ep+1:3d} ep  {n_par:,} params  "
          f"{res['seconds']:6.1f}s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dir with pairs/proteins/ligands parquet")
    ap.add_argument("--emb", default=None, help="dir with prot_residues*.npy")
    ap.add_argument("--prot-tag", default="_esm650")
    ap.add_argument("--arms", default="full,prot_zero,mol_zero")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--attn-dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--prot-cpu", action="store_true",
                    help="keep the protein cache in host RAM instead of on the GPU")
    ap.add_argument("--mol-ram", action="store_true",
                    help="load the ~11 GB ligand cache into host RAM instead of "
                         "reading it from disk. Needs a high-RAM runtime, but "
                         "removes what is likely the step-time bottleneck")
    ap.add_argument("--log-epochs", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--results", default=None,
                    help="where to write the CSV (default: <repo>/results). Set this "
                         "when the script runs outside the repo, e.g. on Modal.")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny local run: 2000 pairs, 2 epochs, 1 seed")
    args = ap.parse_args()

    data = Path(args.data) if args.data else ROOT / "data" / "processed"
    emb = Path(args.emb) if args.emb else ROOT / "data" / "processed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    pairs = pd.read_parquet(data / "pairs.parquet")
    ligands = pd.read_parquet(data / "ligands.parquet")
    meta = json.loads((data / "meta.json").read_text())
    y_mean, y_std = meta["y_mean"], meta["y_std"]

    prot_res = np.load(emb / f"prot_residues{args.prot_tag}.npy", mmap_mode="r")
    prot_off = np.load(emb / f"prot_offsets{args.prot_tag}.npy")
    print(f"protein residues {prot_res.shape} fp16 "
          f"({prot_res.shape[0]*prot_res.shape[1]*2/1e9:.2f} GB), "
          f"{len(prot_off)-1:,} proteins")

    print("\nligand per-atom cache:")
    mol_res_p, mol_off_p = build_mol_cache(ligands["smiles"].tolist(), emb, device)
    mol_res = np.load(mol_res_p, mmap_mode="r")
    mol_off = np.load(mol_off_p)

    # --------------------------------------------- the same three splits
    # Identical construction to 03_train_ablation.py, seeded from meta["seed"],
    # so every number here is directly comparable to the published ones.
    is_train = (pairs["split"] == "train").to_numpy()
    tr_idx = np.flatnonzero(is_train)
    rng = np.random.default_rng(meta["seed"])
    shuf = rng.permutation(len(tr_idx))
    n_dev = int(len(tr_idx) * args.dev_frac)
    dev_idx, fit_idx = tr_idx[shuf[:n_dev]], tr_idx[shuf[n_dev:]]
    test_idx = np.flatnonzero(~is_train)

    if args.smoke:
        fit_idx, dev_idx, test_idx = fit_idx[:2000], dev_idx[:500], test_idx[:500]
        args.max_epochs, args.seeds, args.patience = 2, 1, 2

    splits = {"train": fit_idx, "dev": dev_idx, "test": test_idx}
    train_prots = set(pairs["prot_id"].to_numpy()[fit_idx].tolist())
    seen_mask = np.array([p in train_prots for p in pairs["prot_id"].to_numpy()[test_idx]])
    print(f"\ntrain {len(fit_idx):,}  dev {len(dev_idx):,}  test {len(test_idx):,}"
          f"   |  test rows with an unseen protein: {(~seen_mask).sum():,}")

    bt = Batcher(prot_res, prot_off, mol_res, mol_off, pairs, device,
                 prot_on_gpu=(device.type == "cuda" and not args.prot_cpu),
                 mol_ram=args.mol_ram)

    rows = []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        for seed in range(args.seeds):
            rows.append(run_arm(bt, splits, arm, seed, args, y_mean, y_std, seen_mask))

    df = pd.DataFrame(rows)
    res_dir = Path(args.results) if args.results else ROOT / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cross_attention{args.prot_tag}{args.tag}"
    df.to_csv(res_dir / f"{stem}.csv", index=False)

    print("\n" + "=" * 86)
    print(f"CROSS-ATTENTION SUMMARY  (prot{args.prot_tag}, {args.seeds} seed(s), "
          f"d={args.attn_dim}, {args.heads} heads)")
    print("=" * 86)
    print(f"{'arm':14s}{'test R':>16s}{'RMSE':>8s}{'Spear':>8s}{'R2':>9s}"
          f"{'params':>12s}{'R|seen':>9s}{'R|unseen':>10s}")
    for arm, g in df.groupby("arm", sort=False):
        print(f"{arm:14s}{g['r'].mean():8.4f} +-{g['r'].std(ddof=0):6.4f}"
              f"{g['rmse'].mean():8.4f}{g['spearman'].mean():8.4f}{g['r2'].mean():9.4f}"
              f"{int(g['params'].iloc[0]):12,}{g['r_seen'].mean():9.4f}"
              f"{g['r_unseen'].mean():10.4f}")
    print("\nthe numbers to beat, same split and test set, mean pooling + concat:")
    print("  ESM-2 650M   full 0.7350   ligand-only 0.5524   protein-only 0.5705")
    print("  ProtBERT     full 0.7414   ligand-only 0.5507   protein-only 0.5689")
    print("  zero-parameter protein lookup table  0.5784")
    print(f"\nwrote -> {res_dir / (stem + '.csv')}")


if __name__ == "__main__":
    main()
