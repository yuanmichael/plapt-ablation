# A replication and modification of PLAPT

A look at [PLAPT](https://doi.org/10.1101/2024.02.08.575577) (Bindwell), a sequence-only
protein–ligand binding affinity model.

The original goal was to play around with PLAPT, to see if removing i) the protein arm or
ii) the ligand arm would impact the results of the model. That turned into three
explorations — and the two that didn't work are informative.

| experiment | ΔR |
|---|---|
| **1.** fix a collapsed pooling layer | **+0.23** |
| **2.** swap ProtBERT → ESM-2 650M (4.3× larger encoder) | **−0.006** |
| **3.** replace concatenation with cross-attention | **+0.014** |

PLAPT had one meaningful bug with the pooling layer. Past that, it is constrained by its labels
and the structure of its data — not by its architecture.

All numbers below are internal comparisons; none should be read against the
paper's reported figures (see [Caveats](#caveats)).

---

## 1. The pooler is collapsed 

ProtBERT emits one 1024-number vector *per amino acid* — a 480-residue protein comes out as
a 480×1024 grid. PLAPT needs one fixed-length vector, so the grid must be collapsed.
The default `pooler_output` method takes only the `[CLS]` row from this grid and pushes it through one dense layer plus tanh.
It is the HuggingFace default and the natural thing to reach for.

That dense layer is trained by BERT's next-sentence-prediction objective, which neither
encoder used. 

| representation | dims | effective rank | mean cos(pairs) |
|---|---|---|---|
| `prot_pooler` | 1024 | **1.00** | **1.0000** |
| `prot_mean` | 1024 | 12.71 | 0.7808 |
| `mol_pooler` | 768 | 81.71 | 0.5683 |
| `mol_mean` | 768 | 38.84 | 0.6735 |

Reading this table: Every pair of unrelated proteins has cosine similarity 1.0000. So the structural information from this pooler method is collapsed into essentially 1 dimension. Whereas the "mean pooling" method (taking the average across 480 rows, instead of just taking the one [CLS] row) results in an effective rank of ~13 dimensions.

Here are the results from different training runs: 
| arm | what the head sees | `pooler_output` | mean pooling |
|---|---|---|---|
| `full` | protein + ligand | **0.5066** ±0.0087 | **0.7414** ±0.0017 |
| `mol_cut` | ligand only | 0.5021 ±0.0058 | 0.5515 ±0.0029 |
| `mol_zero` | ligand only | 0.4992 ±0.0050 | 0.5507 ±0.0030 |
| `prot_zero` | protein only | 0.0832 ±0.0302 | 0.5689 ±0.0005 |
| `prot_cut` | protein only | 0.0796 ±0.0072 | 0.5707 ±0.0034 |
| | **protein's contribution** | **+0.005** | **+0.190** |

Baselines using a simple lookup table (no model) on the identical split: protein mean lookup **R = 0.5784**,
ligand mean lookup 0.1841, global mean 0.0000.

**Reading the table.** The left column is PLAPT as published. Removing the protein — by
zeroing it, shuffling it, or deleting the branch outright — barely moves the score: 0.5066
with the protein, 0.4989–0.5021 without it. For scale, re-running the *same* model
changing only the random seed spans 0.4984–0.5158.

The right column is the result of changing the pooling method. Mean pooling changes nothing but how ProtBERT's per-residue
outputs are collapsed into one vector, and the protein's contribution goes from **+0.005
to +0.190**.

## 2. ESM vs. ProtBERT changes nothing

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

## 3. Cross-attention doesn't result in better results either

We then tried cross-attention, to get past #2. When you mean-pool, you collapse the dimensionality in ESM2. 
There's nothing in the model to account for "ligand atom *j* contacts residue *i*" — which is what a binding contact is. 
Introducing cross-attention (ligand atoms as queries over protein residues) gives the model that info. 

| | `full` | R\|seen | R\|unseen | params |
|---|---|---|---|---|
| **cross-attention, ESM-2 650M** (1 seed) | **0.7491** | 0.7532 | **0.6089** | 1,610,881 |
| mean pooling + concat, ESM-2 650M | 0.7350 ±0.0056 | 0.7377 | 0.6344 | 1,609,345 |

No real improvement either vs. #1. 
Instead, the model is likely memorizing the training data better, but not really learning anything new fundamentally about binding
We think it's because 1) our training data is limited (we used 200K pairs out of the full ~1.5M), and 2) the output of PLAPT is 1 scalar number, the binding affinity. That may be compressing a lot of information into 1 scalar.
See the "Planned Next" section for next experiments



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

**[`notes/findings.md`](notes/findings.md)** is the detailed version 

## Caveats

These are **not** a reproduction of the paper's reported numbers, and `0.5066` should not
be compared against their `0.683`:

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
weights. Open-sourcing the weights and method was what made this project possible. 
