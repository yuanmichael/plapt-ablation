# Working notes

Internal. `README.md` is the public write-up; this is the operational stuff.

## GPU jobs run on Modal, not Colab

Decided 2026-10-12. Colab Pro was the original setup and we moved off it.

| | Colab Pro ($9.99/mo) | Modal |
|---|---|---|
| detached runs | **no** — Pro+ only, $49.99/mo | `--detach`, close the laptop |
| cache between sessions | wiped; 7 min rebuild every time | persistent Volume, built once |
| HF model downloads | re-downloads ESM-650 (2.6 GB) each session | cached in `/vol/hf` |
| host RAM | whatever you're handed | `memory=65536` on request |
| cost for this project | 100 compute units/mo | ~$2.50/mo vs $30/mo free credits |

Note: Colab Pro's "100" is **compute units, not GB**. It includes no storage — Drive
is your Google One quota (15 GB free), which is why a 9.8 GB cache was a problem there.

Modal Volume storage is ~$0.09/GiB-month with 1 TiB/month free, so our ~21 GB is free.

### Commands

```powershell
.venv\Scripts\Activate.ps1        # `modal` is in the venv, not on PATH

# one-time
modal volume create plapt-cache
modal volume put plapt-cache data/processed/proteins.parquet /data/proteins.parquet
modal volume put plapt-cache data/processed/ligands.parquet  /data/ligands.parquet
modal volume put plapt-cache data/processed/pairs.parquet    /data/pairs.parquet
modal volume put plapt-cache data/processed/meta.json        /data/meta.json

# every run
modal run modal_app.py::ls                   # free, no GPU: what's in the Volume
modal run --detach modal_app.py::embed       # ~5 min, once per encoder
modal run --detach modal_app.py::train
modal volume get plapt-cache /results ./results
```

Volume layout: `/data` parquets (~15 MB) · `/emb` encoder caches (~21 GB) ·
`/hf` model downloads (~3 GB) · `/results` CSVs.

**`vol.commit()` is required** after writing, or the writes aren't durable.

The Colab notebooks in `notebooks/` still work and are kept for reference, but
`stage3_xattn_colab.ipynb` is superseded by `modal_app.py`.

## Gotchas, each one found the hard way

**Sequences must be `.upper()`ed before tokenising.** 29 of 8,638 are lowercase.
ESM-2 splits on whitespace, a lowercase sequence has none, so the whole protein
becomes a single `<unk>` — 181 residues as one meaningless vector. ProtBERT is
worse because it fails *silently*: we space out every character, so each lowercase
residue becomes its own `[UNK]` and the token count is still correct. No shape
mismatch, just wrong content. **The published ProtBERT and ESM numbers include 29
proteins encoded as garbage** (0.34% of proteins, ~0.33% of pairs — below the
resolution of the reported R, but it's a real defect).

**Never predict token counts from string length.** `02_embed.py` now tokenises
twice and counts exactly. The prediction `min(seq_len, max_length - n_special)`
assumes one token per character, which is false for BPE vocabularies (ChemBERTa)
and false for any input the vocabulary doesn't recognise.

**Never load a memmap to GPU via `torch.from_numpy(np.asarray(mm)).to(dev)`** —
it materialises the whole thing in *host* RAM first. 9.8 GB would OOM a standard
12.7 GB Colab runtime. Stream it in chunks.

**Metrics in float64, always.** Summing 20k float32 terms leaves enough round-off
that `np.std()` of a genuinely constant array returns 9.5e-07, and `corrcoef` then
divides by it and returns a silent `nan`. Use `np.ptp` to detect constants.

**`AutoTokenizer`/`AutoModel` fail on `Rostlab/prot_bert`** under transformers v5:
no `tokenizer.json`, and no `model_type` in `config.json`. Both loaders fall back
through explicit classes. Don't "simplify" them back to Auto.

**Proteins are cached fp32, ligands fp16.** Not a storage choice. ProtBERT's
`pooler_output` has its 2nd component at 1e-3 of the 1st — exactly the fp16 floor
— so fp16 would destroy the signal the experiment is trying to measure.
Per-residue grids are fine in fp16 (LayerNorm'd, effective rank in the hundreds),
and `02_embed.py` verifies this every run by recomputing the mean pool from the fp16
ragged array and correlating against the fp32 mean. Expect `r = 1.000000`.

**Backslash continuations inside IPython `!` cells get mangled.** Keep shell
commands in notebooks on one line. The notebook generator validates every code
cell with `compile()` for this reason.

## Numbers worth not re-deriving

Full dataset: 1,569,572 deduplicated pairs, **22,974 proteins**, 980,310 ligands.
Our sample: 200k pairs, 8,638 proteins (37%), 182,402 ligands.

The dataset is far less diverse than the protein count suggests:
**49.4% of proteins have exactly one measurement**, 78.1% have fewer than ten, and
the top 1,000 hold 82.7% of all pairs.

Protein lengths: 9 to 7,182, median 375, mean 489. Total 4,227,923 residues
(3,797,618 capped at 1022). Ragged storage is 5.4 GB; padding to the longest
would be 79.4 GB.

Published results (200k pairs, 3 seeds, test scored once, mean pooling):

| encoder | full | ligand-only | protein-only |
|---|---|---|---|
| ProtBERT @3200 | 0.7414 | 0.5507 | 0.5689 |
| ESM-2 150M @1024 | 0.7366 | 0.5538 | 0.5706 |
| ESM-2 650M @1024 | 0.7350 | 0.5524 | 0.5705 |

Zero-parameter protein lookup table: **0.5784** — beats every trained
protein-only arm. Ligand lookup 0.1841.

Learning curve, `full`, mean pooling: `R ≈ 0.590 + 0.050 × log2(n/20000)`,
still climbing at 162k with no saturation. The protein-only arm *is* saturated
and tracks the lookup table to within 0.008 at every size. At 1.27M train the fit
overshoots (predicts 0.89, observed 0.8325), so it does finally bend.

Cluster split, mmseqs2 `--min-seq-id 0.3 -c 0.8` (`full`, mean pooling):

| | random | clustered | protein lookup (clustered) |
|---|---|---|---|
| 200k pairs | 0.7414 | 0.3708 ±0.0262 | 0.0000 |
| 1.57M pairs | 0.8325 | 0.3559 (1 seed) | 0.0000 |

200k: 3,093 clusters / 8,638 proteins. Full: 5,047 / 22,974 — 2.66x the proteins
for only **1.63x the families**, which is why 7.8x the data buys nothing under a
cluster split while buying +0.09 under a random one.

Clustered ligand-only floor at 200k is 0.3788 — *above* `full`. The protein branch
contributes −0.008 once it cannot recognise the family. Don't bother swapping the
protein encoder under this split; there is no contribution to improve.

Seed spread under a cluster split is 15x wider (±0.026 vs ±0.0017) because the unit
of variance is the family and test holds 268–458 of them. Two decimals, not four.

**Dev R sits ABOVE test R under a cluster split** (e.g. 0.4599 vs 0.3559) and this
is not overfitting — neither is a training score. Dev is a maximum over epochs
(early stopping selects on it), and dev/test hold different families.

## Standing preferences

- Succinct. Tables and numbers over prose. No scripts written just to explain a
  concept — put the explanation in the reply.
- Don't run unrequested side investigations. Ask first.
- Minimise complexity; don't add control arms or abstractions that weren't asked
  for.
