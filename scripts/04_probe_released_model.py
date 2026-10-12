"""
Probe the RELEASED, ALREADY-TRAINED PLAPT head directly.

This needs no training at all. The authors shipped their trained weights as
affinity_predictor0734-seed2101.onnx, so we can simply ask the actual deployed
model what it does when we vary its inputs.

Two-way sensitivity analysis: take a grid of (protein, ligand) combinations and
ask how much the prediction moves when we change the protein versus when we
change the ligand. If swapping the protein barely moves the output while
swapping the ligand moves it a lot, then the shipped model is effectively a
ligand-only predictor -- and we know that before training a single parameter.

Run:
    .venv/Scripts/python.exe -I scripts/04_probe_released_model.py
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime

ROOT = Path(__file__).resolve().parent.parent
ONNX = ROOT / "data" / "raw" / "affinity_predictor0734-seed2101.onnx"

# PLAPT's own output transform, copied from plapt.py
MEAN, SCALE = 6.51286529169358, 1.5614094578916633


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="data/processed/smoketest",
                   help="dir holding prot_pooler.npy / mol_pooler.npy")
    ap.add_argument("--n", type=int, default=40, help="grid side length")
    args = ap.parse_args()

    emb = ROOT / args.emb
    prot = np.load(emb / "prot_pooler.npy").astype(np.float32)
    mol = np.load(emb / "mol_pooler.npy").astype(np.float32)
    n = min(args.n, len(prot), len(mol))
    prot, mol = prot[:n], mol[:n]
    print(f"probing with {n} real proteins x {n} real ligands = {n*n:,} predictions")

    sess = onnxruntime.InferenceSession(str(ONNX))

    def predict(p, m):
        feat = np.concatenate([p, m]).astype(np.float32)[None, :]
        out = sess.run(None, {"Input": feat, "TrainingMode": np.array(False)})[0][0][0]
        return out * SCALE + MEAN   # -> neg log10 molar, the paper's units

    # full grid: every protein against every ligand
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = predict(prot[i], mol[j])

    print(f"\npredictions span {G.min():.3f} to {G.max():.3f} (neg log10 M)")
    print(f"  for scale: the training data's targets span 0 to 15, std 1.54\n")

    # Hold the ligand fixed, swap the protein 40 ways -> how much moves?
    # Hold the protein fixed, swap the ligand 40 ways -> how much moves?
    across_prot = G.std(axis=0).mean()   # per column (fixed ligand), vary protein
    across_mol = G.std(axis=1).mean()    # per row (fixed protein), vary ligand
    print("=== TWO-WAY SENSITIVITY ===")
    print(f"  fix the ligand, swap among {n} proteins -> prediction std = {across_prot:.6f}")
    print(f"  fix the protein, swap among {n} ligands  -> prediction std = {across_mol:.6f}")
    if across_prot > 0:
        print(f"  the ligand moves the output {across_mol/across_prot:,.0f}x more than the protein")

    print(f"\n  full range when only the protein changes: "
          f"{(G.max(axis=0)-G.min(axis=0)).mean():.6f} log units")
    print(f"  full range when only the ligand  changes: "
          f"{(G.max(axis=1)-G.min(axis=1)).mean():.6f} log units")

    # What happens if we feed a protein vector the model has never seen -- zeros,
    # or pure noise? If the output barely budges, the protein port is decorative.
    print("\n=== FEEDING THE PROTEIN PORT GARBAGE ===")
    ref = np.array([predict(prot[0], mol[j]) for j in range(n)])
    rng = np.random.default_rng(0)
    for label, pv in [
        ("real protein #0", prot[0]),
        ("real protein #17", prot[17]),
        ("all zeros", np.zeros(1024, dtype=np.float32)),
        ("the mean protein vector", prot.mean(0)),
        ("gaussian noise, same scale", rng.normal(prot.mean(), prot.std(), 1024).astype(np.float32)),
        ("all ones", np.ones(1024, dtype=np.float32)),
    ]:
        got = np.array([predict(pv, mol[j]) for j in range(n)])
        print(f"  {label:28s} mean pred {got.mean():7.4f}   "
              f"mean |change| vs real #0: {np.abs(got-ref).mean():.6f}")

    # And the mirror image: scramble the ligand instead.
    print("\n=== MIRROR: FEEDING THE LIGAND PORT GARBAGE ===")
    ref2 = np.array([predict(prot[i], mol[0]) for i in range(n)])
    for label, mv in [
        ("real ligand #0", mol[0]),
        ("real ligand #23", mol[23]),
        ("all zeros", np.zeros(768, dtype=np.float32)),
        ("the mean ligand vector", mol.mean(0)),
    ]:
        got = np.array([predict(prot[i], mv) for i in range(n)])
        print(f"  {label:28s} mean pred {got.mean():7.4f}   "
              f"mean |change| vs real #0: {np.abs(got-ref2).mean():.6f}")

    np.save(ROOT / "results" / "released_model_grid.npy", G)
    print(f"\nwrote results/released_model_grid.npy  ({G.shape})")


if __name__ == "__main__":
    main()
