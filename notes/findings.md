# PLAPT — measurements

Everything here was measured in this repo. Each claim names the script that produced it.

Paper: https://doi.org/10.1101/2024.02.08.575577 · Model: `Bindwell/PLAPT_V1`

---

## 1. Architecture, recovered from the released ONNX

The HF repo ships no training code and no data, only
`models/affinity_predictor0734-seed2101.onnx`. ONNX carries the full graph:

```
Input [1792] = concat( ProtBERT pooler_output [1024], ChemBERTa pooler_output [768] )
  ├─ Slice [0:1024]    → Linear 1024→512 → ReLU ─┐   (authors' name: "ProtOnly")
  └─ Slice [1024:1792] → Linear  768→512 → ReLU ─┤   (authors' name: "MolOnly")
                                       Concat → [1024]
                                       BatchNorm1d(1024, eps=1e-3)
                                       Linear 1024→512 → ReLU
                                       Dropout(p=0.2)
                                       Linear  512→64  → ReLU
                                       Linear   64→1
```

**1,478,273 trainable parameters** (+2,049 BatchNorm running statistics; the ONNX file
holds 1,480,322 numbers in total).

Both encoders are **frozen** — ProtBERT ~420M params, ChemBERTa ~44M. PLAPT trains only
the head. `BatchNorm eps=1e-3` is Wolfram Language's default (PyTorch uses 1e-5), which
with the `WL Notebooks/` directory in their GitHub repo indicates the original was
trained in Mathematica and exported.

## 2. Provenance, and an inherited scaler

PLAPT's README names `jglaser/binding_affinity`. That repo's `data/scaling.json`:

```json
{"n_samples_seen_": 1927202, "mean_": [6.51286529169358], "scale_": [1.5614094578916633]}
```

Byte-identical to the constants hardcoded in `plapt.py`. So the target scaler was not fit
on PLAPT's training split — it was inherited, fit across the entire ~1.93M-row dataset.
Mild target leakage; mainly useful as proof of which data and which column they used.

## 3. The dataset is extremely redundant — `01_prepare_data.py`

`data/all.parquet`:

| | count |
|---|---|
| rows (measurements) | 1,836,729 |
| unique protein sequences | 22,974 |
| unique ligands (canonical SMILES) | 980,310 |
| distinct (protein, ligand) pairs | 1,569,572 |

- Mean **80 measurements per protein**; worst case 12,055. 5,198 proteins (23%) cover
  **98.1%** of rows. Median measurements per *ligand*: **1**.
- **267,157 rows (14.5%) are duplicate (protein, ligand) pairs.** A random split puts the
  same pair on both sides. Collapsed to their mean before splitting here.
- Replicates of the same pair disagree by a **median 0.62 log units** (mean 0.92, p90
  2.23). That is the data's self-consistency and a hard floor on achievable RMSE.

## 4. Zero-parameter baselines

Full dataset, random 90/10 split: **99.35%** of validation rows have a protein that also
appears in training, with ~72 other measurements attached.

| "model" | Pearson R |
|---|---|
| look up this protein's mean affinity in train | **0.579** |
| look up this ligand's mean affinity in train | 0.470 |
| predict the global mean | 0.000 |
| **PLAPT, reported validation** | **0.683** |

A lookup table with no parameters recovers ~85% of the reported correlation. On the 200k
working sample the ligand baseline drops to 0.184, because only 14.6% of test ligands are
seen in training versus 97.8% of proteins.

## 5. `pooler_output` is a collapsed representation — `00_diagnose_pooler.py`

`pooler_output` is the `[CLS]` token through one dense layer + tanh. That layer is trained
by BERT's next-sentence-prediction objective, which neither encoder used. Released
weights, both models declaring `initializer_range = 0.02`:

| | `pooler.dense.weight` std | `pooler.dense.bias` |
|---|---|---|
| ChemBERTa | 0.020011 — *exactly the initialisation* | **exactly all zeros** |
| ProtBERT | 0.001429 — **14× smaller than init** | nonzero (std 0.0878) |
| *ProtBERT `encoder.layer0` attn.out.dense (trained reference)* | 0.044265 | nonzero |

ChemBERTa's pooler was never trained (a bias bit-for-bit zero across 768 values means no
gradient reached it; RoBERTa has no NSP objective). Survivable — a random linear map is
roughly information-preserving. ProtBERT's was trained *toward zero*, so `tanh(Wh+b)`
collapses toward `tanh(b)`.

Measured over all 8,638 proteins / 182,402 ligands in the working set:

| representation | dims | effective rank | dims for 90% var | mean cos(pairs) |
|---|---|---|---|---|
| `prot_pooler` | 1024 | **1.00** | **1** | **1.0000** |
| `prot_mean` | 1024 | 12.71 | 37 | 0.7808 |
| `mol_pooler` | 768 | 81.71 | 127 | 0.5683 |
| `mol_mean` | 768 | 38.84 | 50 | 0.6735 |

Component 1 holds **99.997%** of the variance. Every protein is `μ + α·u` for one fixed
direction: 1,024 numbers carrying one scalar. Vector norms span a 31% range, so α is not
negligible — it is simply one number.

**What α encodes** (`05_what_is_the_one_dimension.py`, 8,638 proteins, 5-fold CV R²):

| predictor set | CV R² |
|---|---|
| length alone | 0.152 |
| size (length, log-length, mass) | 0.254 |
| amino acid composition (20) | 0.088 |
| physicochemical (6) | 0.024 |
| **composition + size (22)** | **0.289** |

Best single predictor is log-length (r = +0.487). So ~29% of α is bulk sequence
statistics, almost entirely size; **71% is unidentified**. Unexplained is not the same as
informative.

For comparison, the [ProtTrans paper](https://arxiv.org/pdf/2007.06225) — which introduced
ProtBERT — also found mean pooling best among the strategies it compared.

**Precision note:** embeddings were computed under `torch.autocast(float16)`, putting a
~1e-3 noise floor on the pooler's components 2+ (real amplitude ~1e-5). Immaterial —
component 1 reproduces at r = 0.99998 against an fp32 CPU computation — but the pooler's
faintest structure was not tested. Proteins are stored fp32 for this reason; ligands fp16,
whose components sit at 0.87/0.78/0.69 and are unaffected.

## 6. Probing the released model — `04_probe_released_model.py`

The shipped ONNX can be interrogated without training anything. 40 real proteins × 40 real
ligands:

| manipulation | prediction std | full range |
|---|---|---|
| fix the ligand, swap among 40 proteins | 0.381 log units | 1.34 |
| fix the protein, swap among 40 ligands | **0.995 log units** | **5.21** |

The ligand moves the output ~3× more. Swapping protein #0 for #17 shifts the prediction by
0.39 — about a quarter of the target's standard deviation (1.54). So the protein port is
used, weakly.

| protein input | mean prediction | shift vs real protein #0 |
|---|---|---|
| real protein #0 | 5.504 | — |
| real protein #17 | 5.360 | 0.390 |
| **the mean protein vector** | **5.504** | **0.002** |
| all zeros | −3.193 | 8.698 |
| all ones | −12.435 | 17.939 |

Two consequences. The mean protein vector is interchangeable with an individual one —
exactly what rank-1 collapse predicts. And zeroing an input is a meaningless probe of an
*already-trained* model: zeros drive the output far outside the data's 0–15 range because
BatchNorm statistics were calibrated on real inputs. Zeroing remains valid in the ablation
below, where the model is retrained with zeros and adapts.

## 7. Ablation — `03_train_ablation.py`

200,000 deduplicated pairs · train 162,000 / dev 18,000 / test 20,000 · random split over
pairs · 3 seeds per cell · test scored once per run. `mol_*` arms keep the LIGAND,
`prot_*` keep the PROTEIN.

**`pooler_output` (what PLAPT ships)**

| arm | test R | RMSE | Spearman | R² | R \| seen | R \| unseen |
|---|---|---|---|---|---|---|
| `full` | **0.5066** ±0.0087 | 1.3510 | 0.4902 | 0.2371 | 0.5071 | 0.4591 |
| `mol_cut` | 0.5021 ±0.0058 | 1.3607 | 0.4884 | 0.2261 | 0.5020 | 0.4729 |
| `mol_zero` | 0.4992 ±0.0050 | 1.3586 | 0.4851 | 0.2285 | 0.4995 | 0.4579 |
| `mol_shuffle` | 0.4989 ±0.0033 | 1.3574 | 0.4838 | 0.2299 | 0.4997 | 0.4397 |
| `prot_zero` | 0.0832 ±0.0302 | 1.5425 | 0.0677 | 0.0055 | 0.0861 | **0.0138** |
| `prot_cut` | 0.0796 ±0.0072 | 1.5429 | 0.0705 | 0.0050 | 0.0818 | 0.0217 |
| `prot_shuffle` | 0.0568 ±0.0173 | 1.5444 | 0.0498 | 0.0030 | 0.0585 | 0.0235 |

**mean pooling (one-line change, identical compute)**

| arm | test R | RMSE | Spearman | R² | R \| seen | R \| unseen |
|---|---|---|---|---|---|---|
| `full` | **0.7414** ±0.0017 | 1.0444 | 0.7356 | 0.5441 | 0.7445 | **0.6317** |
| `prot_cut` | 0.5707 ±0.0034 | 1.2709 | 0.5527 | 0.3249 | 0.5729 | 0.4840 |
| `prot_zero` | 0.5689 ±0.0005 | 1.2726 | 0.5519 | 0.3231 | 0.5712 | 0.4842 |
| `prot_shuffle` | 0.5531 ±0.0026 | 1.2891 | 0.5346 | 0.3055 | 0.5548 | 0.4727 |
| `mol_cut` | 0.5515 ±0.0029 | 1.3016 | 0.5361 | 0.2919 | 0.5515 | 0.5246 |
| `mol_zero` | 0.5507 ±0.0030 | 1.3039 | 0.5337 | 0.2894 | 0.5508 | 0.5212 |
| `mol_shuffle` | 0.5091 ±0.0021 | 1.3408 | 0.4928 | 0.2486 | 0.5087 | 0.4911 |

Baselines, identical split: protein lookup **0.5784** / RMSE 1.2624 · ligand lookup 0.1841
· global mean 0.0000 / RMSE 1.5468. `R | unseen` covers the 469 of 20,000 test pairs whose
protein never appeared in training.

**1. Under PLAPT's pooling the protein contributes nothing.** `full` 0.5066 vs ligand-only
0.4989–0.5021: **+0.005 to +0.007**. Re-running `full` with only the seed changed spans
0.4984–0.5158, a range of 0.0174. The protein contributes less than half of what the
random seed does. All three removal methods agree within 0.003.

**2. A zero-parameter lookup table beats the trained model**: 0.5784 vs 0.5066, and on
RMSE 1.2624 vs 1.3510.

**3. The information exists; the encoding destroys it.** Both receive the same input:

| | R |
|---|---|
| protein identity → zero-parameter lookup table | **0.5784** |
| protein identity → ProtBERT pooler → 524,288-parameter branch | **0.0832** |

**4. Mean pooling recovers it, for free.** Protein-only 0.0832 → 0.5689. Full model 0.5066
→ 0.7414 (RMSE 1.3510 → 1.0444). The protein's contribution, `full` − ligand-only, goes
**+0.005 → +0.190**: an **interaction effect of +0.185**, which is the quantity the 2×2
was built to isolate.

**5. The gain is not better memorisation.** On proteins never seen in training, `full`
goes 0.4591 → **0.6317** and protein-only goes 0.0138 → 0.4842. The seen→unseen drop does
widen (0.113 vs 0.048) — a model that genuinely uses protein identity suffers more when
the protein is unfamiliar — but absolute performance is far better in both regimes.

**6. Which control is strictest depends on the representation.** Under pooler the three
ligand-only arms agree (0.4989/0.4992/0.5021). Under mean pooling they split: zero 0.5507,
cut 0.5515, **shuffle 0.5091**. Permuting *real* protein vectors injects actively
misleading signal rather than merely removing it, so shuffle overstates the contribution
(+0.232 vs the honest +0.190). `cut` is the better estimator once a representation is not
degenerate.

**7. Half the model is dead weight** under pooler: `mol_cut` uses 690,305 parameters
against 1,478,273 and matches `full` (0.5021 vs 0.5066). Under mean pooling the full model
earns them (0.7414 vs 0.5515).

**8. The 40-epoch cap was not binding.** Rerun at 80 (`results/cap80_partial.txt`, stopped
after 6 of 9 runs):

| run | cap 40 | cap 80 | epochs |
|---|---|---|---|
| `full` seeds 0/1/2 | 0.7433 / 0.7398 / 0.7412 | 0.7433 / 0.7398 / 0.7439 | 40 → 42/45/55 |
| `mol_zero` seeds 0/1/2 | 0.5512 / 0.5475 / 0.5535 | 0.5506 / 0.5475 / 0.5535 | 40 → 53/37/46 |
| **means** | 0.7414 / 0.5507 | 0.7423 / 0.5505 | **< 0.001** |

`mol_zero` seed 1 ran exactly 37 epochs both times and returned exactly 0.5475 — the
pipeline is deterministic given a seed, so any difference between configurations is real.

### Predictions registered before running

| prediction | outcome |
|---|---|
| protein-only ≈ full (memorisation) | **Wrong.** Protein-only = 0.083. The memorisation route is real (lookup table 0.578) but PLAPT's representation cannot reach it. |
| ligand-only ≈ full (chemistry dominates) | **Right, wrong reason.** Mean pooling shows the protein is worth +0.190 once it survives encoding. |
| rank-1 collapse is the mechanism | **Supported.** Fixing the collapse recovers the signal — the causal test. |

## 8. Encoder swap — ESM-2 for ProtBERT

Mean pooling, same split, same test set, 3 seeds.

| encoder | dim | full | ligand-only | protein-only | R\|unseen | head params |
|---|---|---|---|---|---|---|
| ProtBERT @3200 | 1024 | **0.7414** ±0.0017 | 0.5507 | 0.5689 | 0.6317 | 1,478,273 |
| ESM-2 150M @1024 | 640 | 0.7366 ±0.0015 | 0.5538 | 0.5706 | 0.6207 | 1,281,665 |
| ESM-2 650M @1024 | 1280 | 0.7350 ±0.0056 | 0.5524 | 0.5705 | 0.6344 | 1,609,345 |

**A 4.3× larger encoder changed the score by −0.0016.** The oldest encoder is nominally best.

Why the swap could not have helped: ESM-2's advantages are per-residue (contact and
structure prediction), and mean pooling deletes the residue axis. Measured:

| | effective rank |
|---|---|
| ProtBERT mean-pooled | 12.7 / 1024 |
| ProtBERT per-residue | 115.9 / 1024 |
| ESM-2 650M mean-pooled | 21.0 / 1280 |
| ESM-2 650M per-residue | **314.4 / 1280** |

Pooling discards ~15× the dimensionality. Effective rank is therefore a **collapse
detector, not a quality metric** — it is anti-correlated with performance across these
three encoders.

Also: ESM-2 exposes no trained pooler (`MISSING: newly initialized`), so
`prot_pooler_esm*` is a random projection. Had we tested the encoder swap *before*
discovering mean pooling, ESM-2 would have appeared to be a large improvement — because
its pooler is random rather than trained-toward-zero, not because it understands proteins
better. The two interventions are **substitutes, not complements**: both route around the
same defect.

## 9. Learning curve — `03_train_ablation.py --train-n`

Dev and test held fixed; only the training set is subsampled, and nested across sizes.

| train_n | full | protein-only | protein lookup | ligand lookup |
|---|---|---|---|---|
| 20,000 | 0.5897 | 0.5104 | 0.5033 | 0.0613 |
| 40,000 | 0.6392 | 0.5374 | **0.5374** | 0.0943 |
| 80,000 | 0.6872 | 0.5566 | 0.5627 | 0.1380 |
| 120,000 | 0.7231 | 0.5590 | 0.5718 | 0.1625 |
| 162,000 | 0.7414 | 0.5689 | 0.5784 | 0.1841 |

`full` is linear in log(data): **R ≈ 0.590 + 0.050 × log₂(n/20,000)**, fitting every
intermediate point to ±0.004, with **no saturation anywhere in range**.

The protein branch, by contrast, *is* saturated — and tracks the zero-parameter lookup
table to within 0.013 at every scale, agreeing to four decimals at 40k. **The protein
pathway's learning curve is the memorisation curve.**

Decomposition: the protein side is data-saturated, the ligand side is nowhere near
(980,310 distinct ligands), so `full`'s continued growth is ligand-driven.

This corrects an earlier over-confident claim in these notes that the architecture rather
than the data was the constraint. Both are constraints.

## 10. Cross-attention — `06_cross_attention.py`

Ligand atoms as queries over protein residues, replacing concatenation. Mean pooling is
the special case where the attention weights are frozen at 1/L.

| | full R | R\|seen | R\|unseen | params |
|---|---|---|---|---|
| cross-attention, ESM-2 650M (1 seed) | **0.7491** | 0.7532 | **0.6089** | 1,610,881 |
| mean pooling + concat, ESM-2 650M | 0.7350 ±0.0056 | 0.7377 | 0.6344 | 1,609,345 |

**+0.0141 at equal parameter count** — larger than the seed spread. But it is a better
*memoriser*: `R|seen` rose +0.0155 while `R|unseen` fell −0.0255.

The training log explains why:

| epoch | train RMSE | dev RMSE | gap |
|---|---|---|---|
| 3 | 1.137 | 1.133 | 0.00 |
| 11 | 0.930 | 1.055 | 0.13 |
| 25 | 0.726 | 1.038 | **0.31** |

Train error approaches the label noise floor (~0.65) while dev error flattens. **A better
inductive bias helps a model that is underfitting; this one is overfitting**, so the new
capability went into fitting the training proteins. No weight decay and no attention
dropout are currently applied.

Three interventions on one pipeline, which is the result worth keeping:

| intervention | cost | ΔR |
|---|---|---|
| fix the collapsed pooler | free | **+0.23** |
| upgrade the encoder 4.3× | 2 h GPU | **−0.006** |
| cross-attention instead of concat | 1 h GPU | **+0.014** |

PLAPT had one catastrophic bug; past that it is limited by label information and data
volume, not by architectural sophistication.

## 11. How much of the test set is already known at training time

| measured per test pair (what the model is graded on) | |
|---|---|
| protein seen in training | **97.66%** |
| ligand seen in training | 13.48% |
| both seen | 12.96% |
| neither seen | 1.82% |

| measured per distinct test entity | |
|---|---|
| distinct test proteins seen (2,776 / 3,222) | 86.16% |
| distinct test ligands seen (2,568 / 19,740) | 13.01% |

For a test pair whose protein was seen, the number of training pairs available for that
same protein: median **168**, mean 261.

This asymmetry explains the memorise/generalise split directly. Proteins recur ~23× so
the protein branch can memorise, and does. Ligands recur ~1.1× so the ligand branch
*cannot* memorise — which is why it reaches 0.5524 against a ligand lookup baseline of
only 0.1841. **The ligand branch does real chemistry; the protein branch does lookup.**
Each branch does the easiest thing its data permits.

`R|unseen` is a weak remedy: "unseen" means exact string mismatch, so a 99%-identical
homolog still counts as unseen.

## 12. Next steps, in priority order

1. **Cluster split at 30% sequence identity.** The blocker on interpreting everything
   above. MMseqs2 `easy-cluster --min-seq-id 0.3 -c 0.8`, then assign whole clusters to
   train/test so no test protein has a close relative in training. Registered prediction:
   `full` falls from ~0.75 toward the ligand-only floor of ~0.55.
2. **Richer supervision than one scalar.** The loss currently compares one predicted
   number to one measured number. AlphaFold 2's distogram term instead predicts a
   distribution over the distance between *every residue pair* and grades it against the
   crystal structure — order 10⁵ supervised quantities per example rather than 1. The
   analogue here is an auxiliary loss on protein–ligand contacts, which would tell the
   attention *where to look* instead of leaving it to infer pocket geometry from a single
   noisy scalar. Sources, cheapest first: an open co-folding model (Boltz, Chai) to
   predict complexes and extract contact residues; BioLiP/PDBbind binding-site residue
   lists, which require sequence→PDB mapping. Note that ESM-2's own `predict_contacts()`
   is free but **circular** — those contacts are derived from the attention that produced
   our cached embeddings, so it adds no information.
3. **Regularisation.** Weight decay and attention dropout are both zero. Cheap, and the
   0.31 train/dev gap is the textbook signature.
4. **More data.** The curve is still linear at 162k. But 49.4% of unused proteins are
   singletons, so added pairs buy ligand coverage rather than protein diversity.

### The label is itself a collapsed representation

The raw dataset's columns are `seq, smiles, affinity_uM, neg_log10_affinity_M,
smiles_can, affinity`. **There is no assay-type column**: Ki, Kd, IC50 and EC50 are
pooled into one scalar before we ever see the data. Those are not interchangeable
quantities, so part of the 0.62 log-unit "noise floor" is definitional heterogeneity
rather than measurement error — and it is unrecoverable from this derived dataset
(BindingDB upstream does retain assay type).

Binding affinity also discards the enthalpy/entropy decomposition and the kinetics
(k_on/k_off; residence time often predicts efficacy better than K_d).

This is the same error as PLAPT's pooling, one level up:

```
PLAPT's input:  400 x 1280 per-residue grid       ->  one 1280-vector
the label:      a thermodynamic + kinetic profile ->  one float
```

Implication: the ~0.91 correlation ceiling is not purely physical. A model given assay
type as a feature, or trained per assay, would face a higher ceiling, because it would no
longer be asked to predict a quantity that is not well defined.

## 13. Limitations

- **200,000 pairs**, ~13% of the 1.57M available.
- **Not a reproduction of the paper's numbers.** 0.5066 is not comparable to their 0.683:
  different data volume, deduplication, and a clean test split. Every comparison here is
  internal — same data, same split, same code, same seeds, one variable changed.
- **One random split**, in which 97.66% of test pairs have a protein the model trained on
  (§11). A cluster-based split at 30% sequence identity is the test that matters, and every
  number here would drop under it. `R | unseen` is a weak proxy: 469 rows, and "unseen" is
  exact string mismatch, so homologs leak.
- **29 of 8,638 sequences were lowercase**, and both vocabularies are uppercase-only. ESM-2
  turned each whole sequence into a single `<unk>`; ProtBERT, which sees one spaced
  character per residue, produced one `[UNK]` per residue — correct token count, wrong
  content, no error raised. So every number in §5–§10 includes 29 proteins encoded as
  garbage (0.34% of proteins, ~0.33% of pairs — below the resolution of the reported R).
  Fixed in `02_embed.py` after the per-residue work tripped a row-count assertion.
- **The mean-pooled head is not a shippable replacement.** The finding is supported; a
  model release would need full-data training.
- **An earlier claim here that PLAPT "cannot distinguish two proteins" was retracted**
  after §6. Cosine ignores magnitude; vector lengths vary 31% and a Linear layer reads
  them. The defensible claim is a coarse global prior, not a recognition mechanism.
