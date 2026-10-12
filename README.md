# A replication and modification of PLAPT

A look at [PLAPT](https://doi.org/10.1101/2024.02.08.575577) (Bindwell), a sequence-only
protein–ligand binding affinity model.

The original goal was to play around with PLAPT, to see if removing i) the protein arm or ii) the ligand arm would impact the results of the model. 

**Summary:** PLAPT feeds `pooler_output` from a frozen ProtBERT into its prediction
head. Measured over 8,638 real proteins, that representation has an **effective rank of
1.00 out of 1024** — every pair of unrelated proteins has cosine similarity 1.0000.
Consequently the protein branch contributes little: delete the protein entirely and
performance is unchanged. Swapping `pooler_output` for mask-aware mean pooling — same
encoder, same forward pass, no extra compute — takes the full model from **R = 0.5066 to
R = 0.7414**.

---

## Results

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


**Reading the table.** The left column is PLAPT as published. Removing the protein —
by zeroing it, shuffling it, or deleting the branch outright — barely moves the score:
0.5066 with the protein, 0.4989–0.5021 without it. For scale, re-running the *same*
model changing only the random seed spans 0.4984–0.5158. The protein contributes less
than the seed does.

That leaves two possible explanations. Either protein identity isn't much use for
predicting affinity, or it is useful but isn't reaching the model. The lookup-table
baseline distinguishes them: "average this protein's training affinities and ignore the
ligand entirely" scores **0.5784**, better than the full trained model. So protein
identity does carry real signal on this split — it just isn't surviving the trip through
ProtBERT's pooler.

The right column tests that explanation. Mean pooling changes nothing but the way
ProtBERT's per-residue outputs are collapsed into a single vector, and the protein's
contribution goes from **+0.005 to +0.190**. The gain also holds on proteins the model
has never seen before (R **0.4591 → 0.6317** on the 469 such test pairs), so it isn't
simply a better index for recognising familiar targets.

## Why: the pooler is collapsed

ProtBERT emits one 1024-number vector *per amino acid* — a 480-residue protein comes out
as a 480×1024 grid. PLAPT needs a single fixed-length vector per protein, so that grid
has to be collapsed somehow. PLAPT uses `pooler_output`: take only the `[CLS]` token's
row and push it through one extra dense layer plus tanh. It is the default available in
the HuggingFace API and the natural thing to reach for.

The catch is that this dense layer is trained by BERT's next-sentence-prediction
objective, which neither of these encoders used. Inspecting the released weights (both
models declare `initializer_range = 0.02`):

| | `pooler.dense.weight` std | `pooler.dense.bias` |
|---|---|---|
| ChemBERTa | 0.020011 — *exactly the initialisation* | **exactly all zeros** |
| ProtBERT | 0.001429 — **14× smaller than init** | nonzero (std 0.0878) |
| *ProtBERT `encoder.layer0` attn.out.dense (trained reference)* | 0.044265 | nonzero |

ChemBERTa's pooler was never trained — a bias that is bit-for-bit zero across 768 values
means no gradient ever reached it. That is survivable: a random linear map is roughly
information-preserving. ProtBERT's was trained *toward zero*, so `tanh(Wh + b)` collapses
toward `tanh(b)` — nearly the same vector for every protein.

Measured over 8,638 real proteins:

| representation | dims | effective rank | mean cos(pairs) |
|---|---|---|---|
| `prot_pooler` | 1024 | **1.00** | **1.0000** |
| `prot_mean` | 1024 | 12.71 | 0.7808 |
| `mol_pooler` | 768 | 81.71 | 0.5683 |
| `mol_mean` | 768 | 38.84 | 0.6735 |

The [ProtTrans paper](https://arxiv.org/pdf/2007.06225), which introduced ProtBERT, also
found mean pooling best among the strategies it compared.

## Reproducing

```bash
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install numpy pandas pyarrow scikit-learn matplotlib \
    requests huggingface_hub onnx onnxruntime transformers
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
#    locally this is a smoke test; do the real run via notebooks/stage2_embed_colab.ipynb
.venv/Scripts/python.exe -I scripts/02_embed.py --limit 32 --out data/processed/smoketest

# 3. the ablation: 7 arms x 3 seeds x 2 poolings   (~3.5 h, CPU)
.venv/Scripts/python.exe -I scripts/03_train_ablation.py --pooling pooler --seeds 3
.venv/Scripts/python.exe -I scripts/03_train_ablation.py --pooling mean   --seeds 3
```

No data is committed to this repo. `01_prepare_data.py` regenerates the
exact 200,000-pair split from the public source, deterministically, from seed 2101 — the
seed baked into Bindwell's released model filename.

## Layout

```
scripts/00_diagnose_pooler.py          is pooler_output an informative representation?
scripts/01_prepare_data.py             raw dump -> clean split + zero-parameter baselines
scripts/02_embed.py                    frozen ProtBERT / ChemBERTa -> cached .npy
scripts/03_train_ablation.py           PLAPT's head under 7 information conditions
scripts/04_probe_released_model.py     interrogate Bindwell's shipped ONNX directly
scripts/05_what_is_the_one_dimension.py  regress the surviving scalar on sequence stats
scripts/watch_training.py              live terminal view of a training run
notebooks/stage2_embed_colab.ipynb     the GPU step
notes/findings.md                      full write-up: every result, with provenance
results/                               the ablation CSVs
```

**[`notes/findings.md`](notes/findings.md)** is the detailed version — the full 7-arm ×
2-pooling grid with Spearman, R² and seen/unseen breakdowns, a direct probe of Bindwell's
shipped ONNX, what the surviving protein dimension actually encodes, the predictions
registered before running, and the limitations.

## Caveats

These are **not** a reproduction of the paper's reported numbers, and `0.5066` should not
be compared against their `0.683` (against their validation set):

- **200,000 training pairs**, about 13% of the 1.57M deduplicated pairs available.
- **Duplicate pairs collapsed first.** 24.8% of source rows are repeat measurements of
  the same (protein, ligand); a random split otherwise puts the same pair on both sides.
  Averaging replicates removes ~14.5% of rows. (Those replicates disagree by a median of
  **0.62 log units**, which is the data's own noise floor and a hard limit on any RMSE.)
- **Three splits, not two.** Dev is used for early stopping; test is scored once. PLAPT's
  0.683 is a best-epoch validation score.
- **One random split.** A cluster-based split at 30% sequence identity is the test that
  matters for real-world generalisation, and every number here would drop under it.
- **The mean-pooled head is not a shippable replacement for PLAPT.** Trained on 13% of
  the data. The finding is supported; a model release would need the full set.


## Acknowledgements

Major thank you to the Bindwell team, who released PLAPT openly including the trained ONNX weights. Reading the
architecture straight out of that file is what made this project possible. 