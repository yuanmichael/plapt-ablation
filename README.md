# A replication and modification of PLAPT

A look at [PLAPT](https://doi.org/10.1101/2024.02.08.575577) (Bindwell), a sequence-only
protein–ligand binding affinity model.

The original goal was to play around with PLAPT, to see if removing i) the protein arm or
ii) the ligand arm would impact the results of the model. That turned into three
interventions on the same pipeline — and the two that *didn't* work are as informative as
the one that did.

| intervention | cost | ΔR |
|---|---|---|
| **1.** fix a collapsed pooling layer | free — same forward pass | **+0.23** |
| **2.** swap ProtBERT → ESM-2 650M (4.3× larger encoder) | ~2 h GPU | **−0.006** |
| **3.** replace concatenation with cross-attention | ~1 h GPU | **+0.014** |

PLAPT had one catastrophic bug. Past that, it is limited by the information in its labels
and the structure of its data — not by the sophistication of its architecture.

Everything below holds the data, the split, the seeds and the code fixed, and changes one
variable at a time. All numbers are internal comparisons; none should be read against the
paper's reported figures (see [Caveats](#caveats)).

---

## 1. The pooler is collapsed — and fixing it is free

PLAPT feeds `pooler_output` from a frozen ProtBERT into its prediction head. Measured over
8,638 real proteins, that representation has an **effective rank of 1.00 out of 1024** —
every pair of unrelated proteins has cosine similarity 1.0000.

PLAPT's exact prediction head (1,478,273 parameters, transcribed from the released ONNX
graph), 200,000 deduplicated pairs, random split over pairs, 3 seeds per cell, test set
scored once per run.

| arm | what the head sees | `pooler_output` | mean pooling |
|---|---|---|---|
| `full` | protein + ligand | **0.5066** ±0.0087 | **0.7414** ±0.0017 |
| `mol_cut` | ligand only | 0.5021 ±0.0058 | 0.5515 ±0.0029 |
| `mol_zero` | ligand only | 0.4992 ±0.0050 | 0.5507 ±0.0030 |
| `prot_zero` | protein only | 0.0832 ±0.0302 | 0.5689 ±0.0005 |
| `prot_cut` | protein only | 0.0796 ±0.0072 | 0.5707 ±0.0034 |
| | **protein's contribution** | **+0.005** | **+0.190** |

Zero-parameter baselines on the identical split: protein mean lookup **R = 0.5784**,
ligand mean lookup 0.1841, global mean 0.0000.

**Reading the table.** The left column is PLAPT as published. Removing the protein — by
zeroing it, shuffling it, or deleting the branch outright — barely moves the score: 0.5066
with the protein, 0.4989–0.5021 without it. For scale, re-running the *same* model
changing only the random seed spans 0.4984–0.5158. **The protein contributes less than the
seed does.**

Two explanations are possible: either protein identity isn't much use here, or it is
useful but isn't reaching the model. The lookup-table baseline separates them. "Average
this protein's training affinities and ignore the ligand entirely" scores **0.5784** —
better than the full trained model. So protein identity carries real signal on this split;
it just isn't surviving the trip through ProtBERT's pooler.

The right column tests that. Mean pooling changes nothing but how ProtBERT's per-residue
outputs are collapsed into one vector, and the protein's contribution goes from **+0.005
to +0.190**.

### Why the pooler collapses

ProtBERT emits one 1024-number vector *per amino acid* — a 480-residue protein comes out as
a 480×1024 grid. PLAPT needs one fixed-length vector, so the grid must be collapsed.
`pooler_output` takes only the `[CLS]` row and pushes it through one dense layer plus tanh.
It is the HuggingFace default and the natural thing to reach for.

That dense layer is trained by BERT's next-sentence-prediction objective, which neither
encoder used. Inspecting the released weights (both declare `initializer_range = 0.02`):

| | `pooler.dense.weight` std | `pooler.dense.bias` |
|---|---|---|
| ChemBERTa | 0.020011 — *exactly the initialisation* | **exactly all zeros** |
| ProtBERT | 0.001429 — **14× smaller than init** | nonzero (std 0.0878) |
| *ProtBERT `encoder.layer0` attn.out.dense (trained reference)* | 0.044265 | nonzero |

ChemBERTa's pooler was never trained — a bias bit-for-bit zero across 768 values means no
gradient ever reached it. That is survivable: a random linear map is roughly
information-preserving. ProtBERT's was trained *toward zero*, so `tanh(Wh + b)` collapses
toward `tanh(b)` — nearly the same vector for every protein.

| representation | dims | effective rank | mean cos(pairs) |
|---|---|---|---|
| `prot_pooler` | 1024 | **1.00** | **1.0000** |
| `prot_mean` | 1024 | 12.71 | 0.7808 |
| `mol_pooler` | 768 | 81.71 | 0.5683 |
| `mol_mean` | 768 | 38.84 | 0.6735 |

The [ProtTrans paper](https://arxiv.org/pdf/2007.06225), which introduced ProtBERT, also
found mean pooling best among the strategies it compared.

## 2. A 4.3× larger, more modern encoder changes nothing

If the protein representation was the bottleneck, a better protein encoder should help.
ESM-2 is four years newer than ProtBERT, uses rotary embeddings, and its 650M checkpoint
is the standard choice for protein work. Same head, same split, same seeds, mean pooling
throughout:

| encoder | dim | `full` | ligand-only | protein-only | R\|unseen |
|---|---|---|---|---|---|
| ProtBERT @3200 | 1024 | **0.7414** ±0.0017 | 0.5507 | 0.5689 | 0.6317 |
| ESM-2 150M @1024 | 640 | 0.7366 ±0.0015 | 0.5538 | 0.5706 | 0.6207 |
| ESM-2 650M @1024 | 1280 | 0.7350 ±0.0056 | 0.5524 | 0.5705 | 0.6344 |

**Going from 150M to 650M parameters — 4.3× — moved the score by −0.0016.** The oldest
encoder is nominally the best of the three.

The reason is visible in the representations. ESM-2's advantages are *per-residue* —
contact prediction, secondary structure — and mean pooling deletes the residue axis:

| | effective rank |
|---|---|
| ProtBERT, mean-pooled | 12.7 / 1024 |
| ProtBERT, per-residue | 115.9 / 1024 |
| ESM-2 650M, mean-pooled | 21.0 / 1280 |
| ESM-2 650M, per-residue | **314.4 / 1280** |

Pooling discards roughly 15× the dimensionality. A sharper encoder doesn't help when the
next step averages its output away.

Two further notes. **Effective rank is a collapse detector, not a quality metric** — it is
*anti*-correlated with performance across these three encoders. And ESM-2 exposes no
trained pooler (`MISSING: newly initialized`), so had the encoder swap been tried *before*
the pooling fix, ESM-2 would have looked like a large win — because its pooler is random
rather than trained-toward-zero, not because it understands proteins better. The two
interventions are substitutes, not complements.

## 3. Cross-attention improves the headline number by becoming a better memoriser

PLAPT computes a protein vector and a ligand vector independently and concatenates them,
so no tensor in the model has a cell for "ligand atom *j* contacts residue *i*" — which is
what a binding contact is. Replacing the concatenation with cross-attention (ligand atoms
as queries over protein residues) gives the model that cell. Mean pooling is the special
case where the attention weights are frozen at 1/L.

| | `full` | R\|seen | R\|unseen | params |
|---|---|---|---|---|
| **cross-attention, ESM-2 650M** (1 seed) | **0.7491** | 0.7532 | **0.6089** | 1,610,881 |
| mean pooling + concat, ESM-2 650M | 0.7350 ±0.0056 | 0.7377 | 0.6344 | 1,609,345 |

+0.0141 at equal parameter count, larger than the seed spread. But split by whether the
test protein was seen in training, the gain inverts: `R|seen` **rose** 0.0155 while
`R|unseen` **fell** 0.0255.

The training log says why:

| epoch | train RMSE | dev RMSE | gap |
|---|---|---|---|
| 3 | 1.137 | 1.133 | 0.00 |
| 11 | 0.930 | 1.055 | 0.13 |
| 25 | 0.726 | 1.038 | **0.31** |

Training error reaches the data's own noise floor (~0.65 log units, estimated from
replicate measurements of the same pair) while dev error flattens. **A better inductive
bias helps a model that is underfitting; this one is overfitting**, so the new capability
went into fitting the training proteins harder.

## 4. What the model is actually doing

Measured on the split PLAPT uses — random over pairs:

| per test pair (what the model is graded on) | |
|---|---|
| protein seen in training | **97.66%** |
| ligand seen in training | 13.48% |

For a test pair whose protein was seen, the median number of *other* training pairs for
that same protein is **168**. So the typical test question is "here is a protein you have
seen 168 times, plus a new ligand" — and a model can score well by recalling that
protein's average affinity.

The asymmetry between the two branches follows directly from how the data repeats:

| branch | pairs per entity | lookup-table R | trained branch R | verdict |
|---|---|---|---|---|
| protein | 23.3 | **0.5784** | 0.5705 | memorising — and worse than the dictionary |
| ligand | 1.1 | 0.1841 | 0.5524 | generalising — 3× better than the dictionary |

Ligands recur ~1.1 times, so the ligand branch *cannot* memorise and must read chemical
structure. Proteins recur ~23 times, so the protein branch can memorise — and does. Across
all three encoders and at every training-set size, **the trained protein branch never beats
a zero-parameter lookup table.**

A learning curve confirms it. Holding dev and test fixed and subsampling only the training
set:

| train_n | `full` | protein-only | protein lookup |
|---|---|---|---|
| 20,000 | 0.5897 | 0.5104 | 0.5033 |
| 40,000 | 0.6392 | 0.5374 | **0.5374** |
| 80,000 | 0.6872 | 0.5566 | 0.5627 |
| 162,000 | 0.7414 | 0.5689 | 0.5784 |

`full` is linear in log(data) — `R ≈ 0.590 + 0.050 × log₂(n/20,000)`, fitting every
intermediate point to ±0.004, **with no saturation anywhere in range**. The protein branch,
by contrast, *is* saturated, and tracks the lookup table to within 0.013 at every scale,
agreeing to four decimals at 40k. **The protein pathway's learning curve is the
memorisation curve.**

## Reproducing

```bash
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

mkdir -p data/raw
curl -L -o data/raw/all.parquet \
  "https://huggingface.co/datasets/jglaser/binding_affinity/resolve/main/data/all.parquet"
curl -L -o data/raw/affinity_predictor0734-seed2101.onnx \
  "https://huggingface.co/Bindwell/PLAPT_V1/resolve/main/models/affinity_predictor0734-seed2101.onnx"
```

```bash
# 1. clean, deduplicate, split, print the zero-parameter baselines   (~1 min, CPU)
.venv/Scripts/python.exe -I scripts/01_prepare_data.py --n 200000

# 2. the only GPU step: cache the frozen encoder outputs   (~20 min on an L4)
#    locally this is a smoke test; the real run goes to Modal (modal_app.py)
.venv/Scripts/python.exe -I scripts/02_embed.py --limit 32 --out data/processed/smoketest

# 3. the ablation: 7 arms x 3 seeds x 2 poolings   (~3.5 h, CPU)
.venv/Scripts/python.exe -I scripts/03_train_ablation.py --pooling pooler --seeds 3
.venv/Scripts/python.exe -I scripts/03_train_ablation.py --pooling mean   --seeds 3
```

No data is committed to this repo. `01_prepare_data.py` regenerates the exact
200,000-pair split from the public source, deterministically, from seed 2101 — the seed
baked into Bindwell's released model filename.

GPU stages run on [Modal](https://modal.com) via `modal_app.py` (`embed`, `train`,
`cluster`, `ls`); see `CLAUDE.md` for why, and for the gotchas worth not rediscovering.

## Layout

```
scripts/00_diagnose_pooler.py            is pooler_output an informative representation?
scripts/01_prepare_data.py               raw dump -> clean split + zero-parameter baselines
scripts/02_embed.py                      frozen encoders -> cached .npy (pooled or per-residue)
scripts/03_train_ablation.py             PLAPT's head under 7 information conditions
scripts/04_probe_released_model.py       interrogate Bindwell's shipped ONNX directly
scripts/05_what_is_the_one_dimension.py  regress the surviving scalar on sequence stats
scripts/06_cross_attention.py            ligand atoms attend over protein residues
scripts/07_cluster_split.py              MMseqs2 split at 30% sequence identity
scripts/watch_training.py                live terminal view of a training run
modal_app.py                             the GPU/CPU jobs
notes/findings.md                        full write-up: every result, with provenance
results/                                 the ablation CSVs
```

**[`notes/findings.md`](notes/findings.md)** is the detailed version — the full 7-arm ×
2-pooling grid with Spearman, R² and seen/unseen breakdowns, a direct probe of Bindwell's
shipped ONNX, what the surviving protein dimension encodes, the predictions registered
before running, and the limitations.

## Caveats

These are **not** a reproduction of the paper's reported numbers, and `0.5066` should not
be compared against their `0.683` (against their validation set):

- **200,000 training pairs**, about 13% of the 1.57M deduplicated pairs available.
- **Duplicate pairs collapsed first.** 24.8% of source rows are repeat measurements of the
  same (protein, ligand); a random split otherwise puts the same pair on both sides.
  Averaging replicates removes ~14.5% of rows. (Those replicates disagree by a median of
  **0.62 log units**, the data's own noise floor and a hard limit on any RMSE — it implies
  a correlation ceiling near 0.91, not 1.0.)
- **Three splits, not two.** Dev is used for early stopping; test is scored once. PLAPT's
  0.683 is a best-epoch validation score.
- **One random split**, in which 97.66% of test pairs have a protein the model trained on.
  A cluster split at 30% sequence identity is the test that matters, and every number here
  would drop under it. `R|unseen` is a weak proxy: 469 rows, and "unseen" means exact
  string mismatch, so homologs leak.
- **29 of 8,638 sequences were lowercase**, and both vocabularies are uppercase-only — so
  they encoded as `[UNK]`. 0.34% of proteins, ~0.33% of pairs; below the resolution of the
  reported R, but present in every number above. Fixed in `02_embed.py`.
- **Cross-attention is one seed.** The +0.0141 needs replication.
- **The mean-pooled head is not a shippable replacement for PLAPT.** Trained on 13% of the
  data. The finding is supported; a model release would need the full set.
- **The label is itself a collapsed representation.** The source dataset has no assay-type
  column — Ki, Kd, IC50 and EC50 are pooled into one scalar before we see it. Part of that
  0.62 "noise floor" is definitional heterogeneity, not measurement error.

## Planned next

**1. Cluster split at 30% sequence identity.** The blocker on interpreting everything
above. Under the current random split over pairs, **97.66% of test pairs have a protein
the model trained on**, in a median of 168 other pairs — so a model can score well by
memorizing a protein's average affinity. Implemented in `scripts/07_cluster_split.py`
(MMseqs2, whole clusters to train/dev/test). Registered prediction: `full` falls from
~0.75 toward the ligand-only floor of ~0.55.

**2. Richer supervision than a single scalar.** The loss currently compares one predicted
number to one measured number. AlphaFold 2's distogram term instead predicts a
distribution over the distance between *every residue pair* and grades it against the
crystal structure — roughly 10⁵ supervised quantities per example rather than 1.

The analogue here: run an open co-folding model over our (protein, ligand) pairs to
predict the complex, extract which residues contact the ligand, and add an auxiliary loss
pulling the cross-attention map toward those contacts. That tells the attention *where to
look* rather than leaving it to infer pocket geometry from one noisy scalar — which
finding 3 above suggests it cannot do.

**Prior art to be honest about.** This is not novel territory. **Boltz-2** already does
structure-aware affinity prediction trained with exactly this kind of richer supervision.
**PLINDER** and **LP-PDBbind** exist specifically because the field concluded that random
splits over protein–ligand pairs leak.

**3. Full 1.57M pairs.** The learning curve is still linear at 162k
(`R ≈ 0.590 + 0.050 × log₂(n/20,000)`, no saturation). Cheap for the pooled/concat
pipeline; expensive for cross-attention, which would need a 60.6 GB per-atom ligand cache.

## Acknowledgements

Major thank you to the Bindwell team, who released PLAPT openly including the trained ONNX
weights. Reading the architecture straight out of that file is what made this project
possible.
