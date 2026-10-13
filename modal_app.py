"""
Run the GPU stages on Modal instead of Colab.

Why: a Modal Volume persists between runs, so the 21 GB of cached encoder
outputs is built ONCE and reused forever -- where Colab wipes it every session.
And `modal run --detach` means the job survives closing your laptop.

----------------------------------------------------------------------------
ONE-TIME SETUP
----------------------------------------------------------------------------
    pip install modal
    modal setup                      # opens a browser, links your account

    modal volume create plapt-cache
    modal volume put plapt-cache data/processed/proteins.parquet /data/proteins.parquet
    modal volume put plapt-cache data/processed/ligands.parquet  /data/ligands.parquet
    modal volume put plapt-cache data/processed/pairs.parquet    /data/pairs.parquet
    modal volume put plapt-cache data/processed/meta.json        /data/meta.json

----------------------------------------------------------------------------
EVERY RUN
----------------------------------------------------------------------------
    modal run --detach modal_app.py::embed      # ~5 min, once per encoder
    modal run --detach modal_app.py::train      # the experiment
    modal volume get plapt-cache /results ./results    # pull the CSVs down

`--detach` returns immediately and the job keeps running. Watch it at
modal.com/apps, or re-attach with `modal app logs <id>`.

----------------------------------------------------------------------------
WHAT LIVES WHERE
----------------------------------------------------------------------------
    /data      the parquet tables you uploaded once        ~15 MB
    /emb       cached encoder outputs, built by embed()    ~21 GB
    /hf        HuggingFace model downloads                 ~3 GB
    /results   the output CSVs                             ~5 KB

All four are directories inside the one persistent Volume. Caching /hf matters
more than it looks: ESM-2 650M is a 2.6 GB download that would otherwise repeat
on every single run.
"""

import modal

app = modal.App("plapt-ablation")
vol = modal.Volume.from_name("plapt-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "numpy", "pandas", "pyarrow", "psutil",
                 # 03_train_ablation.py imports scipy for Spearman; 06 does not.
                 "scipy")
    # Mounted at runtime, so editing a script locally takes effect on the next
    # `modal run` with no image rebuild.
    .add_local_file("scripts/01_prepare_data.py", "/app/01_prepare_data.py")
    .add_local_file("scripts/02_embed.py", "/app/02_embed.py")
    .add_local_file("scripts/03_train_ablation.py", "/app/03_train_ablation.py")
    .add_local_file("scripts/06_cross_attention.py", "/app/06_cross_attention.py")
)

# MMseqs2 is a C++ binary, awkward to install on Windows and trivial here. The
# static build is used rather than apt's, because distro packages lag badly.
cluster_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "tar")
    .pip_install("numpy", "pandas", "pyarrow")
    .run_commands(
        "wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /tmp/m.tar.gz",
        "tar xzf /tmp/m.tar.gz -C /opt && rm /tmp/m.tar.gz",
    )
    .env({"PATH": "/opt/mmseqs/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_file("scripts/07_cluster_split.py", "/app/07_cluster_split.py")
)

VOLUMES = {"/vol": vol}
ENV = {"HF_HOME": "/vol/hf"}


def _run(cmd: list[str]) -> None:
    """Stream a subprocess's output so `modal app logs` shows live progress."""
    import subprocess
    import sys
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.run([sys.executable, *cmd])
    if p.returncode != 0:
        raise RuntimeError(f"exited {p.returncode}")


@app.function(image=image, gpu="L4", volumes=VOLUMES, env=ENV, timeout=3600)
def embed(prot_model: str = "facebook/esm2_t33_650M_UR50D",
          tag: str = "_esm650", prot_max: int = 1024):
    """Cache per-residue protein embeddings into the Volume. ~5 min on an L4."""
    import os
    os.makedirs("/vol/emb", exist_ok=True)
    _run(["/app/02_embed.py", "--data", "/vol/data", "--out", "/vol/emb",
          "--save-residues", "--exclude-special", "--skip-ligands",
          "--prot-model", prot_model, "--prot-max", str(prot_max),
          "--tag", tag, "--prot-budget", "8192"])
    vol.commit()          # without this the writes are not durable


@app.function(image=image, gpu="L4", volumes=VOLUMES, env=ENV, timeout=86400,
              memory=65536)
def train(prot_tag: str = "_esm650", arms: str = "full", seeds: int = 1,
          max_epochs: int = 25, patience: int = 4, batch_size: int = 64,
          tag: str = "", log_epochs: bool = True):
    """Train the cross-attention head. Builds the ligand cache on first call.

    memory=65536 (64 GB) is requested so --mol-ram can hold the 11 GB ligand
    cache in host memory. On disk those are ~162k small random reads per epoch,
    which is the step-time bottleneck.
    """
    cmd = ["/app/06_cross_attention.py", "--data", "/vol/data", "--emb", "/vol/emb",
           "--results", "/vol/results", "--prot-tag", prot_tag, "--arms", arms,
           "--seeds", str(seeds), "--max-epochs", str(max_epochs),
           "--patience", str(patience), "--batch-size", str(batch_size),
           "--mol-ram", "--tag", tag]
    if log_epochs:
        cmd.append("--log-epochs")
    _run(cmd)
    vol.commit()


@app.function(image=image, gpu="L4", volumes=VOLUMES, env=ENV, timeout=14400,
              memory=32768)
def full_prepare():
    """Prepare and embed ALL 1,569,572 pairs with the original PLAPT encoders.

    Writes to /vol/data_full and /vol/emb_full -- NOT over the 200k tables, which
    the existing results and the cluster split both depend on.

    Needs /vol/raw/all.parquet (227 MB), uploaded once:
        modal volume put plapt-cache data/raw/all.parquet /raw/all.parquet

    Scale, measured rather than guessed:
        pairs      1,569,572   (7.8x)
        proteins      22,974   (2.7x)   -> ~13 min on an L4
        ligands      980,310   (5.4x)   -> ~16 min
      pooled caches: prot 0.12 GB fp32 + mol 1.51 GB fp16. Trivial.
      (Per-residue caches would be 21.9 + 60.6 GB, which is why cross-attention
       is not part of this run.)
    """
    import os
    os.makedirs("/vol/data_full", exist_ok=True)
    os.makedirs("/vol/emb_full", exist_ok=True)
    _run(["/app/01_prepare_data.py", "--n", "0",
          "--raw", "/vol/raw/all.parquet", "--out", "/vol/data_full"])
    vol.commit()
    # ProtBERT + ChemBERTa, pooled only: the original PLAPT architecture.
    _run(["/app/02_embed.py", "--data", "/vol/data_full", "--out", "/vol/emb_full",
          "--prot-model", "Rostlab/prot_bert", "--prot-max", "3200",
          "--prot-budget", "8192"])
    vol.commit()


@app.function(image=image, volumes=VOLUMES, env=ENV, cpu=16.0, memory=32768,
              timeout=86400)
def full_train(pooling: str = "mean", arms: str = "full,mol_zero,prot_zero",
               seeds: int = 3, max_epochs: int = 60, patience: int = 6,
               train_n: int = 0, tag: str = "_full"):
    """Train PLAPT's original head on the full data. CPU -- the head is a small MLP.

    Pass --train-n to extend the learning curve on a CONSISTENT test set:
    162000, 320000, 640000, then 0 for all of it. Comparing a full-data run
    against the published 0.7414 directly would be wrong, because the test set
    also changes -- the full dataset has far more singleton proteins, so its test
    set is genuinely harder and R can fall while the model improves.
    """
    import shutil
    import os
    # 03_train_ablation.py reads caches and tables from one directory.
    os.makedirs("/vol/run_full", exist_ok=True)
    for f in os.listdir("/vol/data_full"):
        if f.endswith((".parquet", ".json")):
            shutil.copy(f"/vol/data_full/{f}", f"/vol/run_full/{f}")
    for f in os.listdir("/vol/emb_full"):
        if f.endswith((".npy", ".json")):
            shutil.copy(f"/vol/emb_full/{f}", f"/vol/run_full/{f}")
    cmd = ["/app/03_train_ablation.py", "--proc", "/vol/run_full",
           "--pooling", pooling, "--arms", arms, "--seeds", str(seeds),
           "--max-epochs", str(max_epochs), "--patience", str(patience),
           "--threads", "16", "--tag", tag]
    if train_n:
        cmd += ["--train-n", str(train_n)]
    _run(cmd)
    # 03_train_ablation.py writes to <repo>/results, which resolves to /results
    # when the script lives at /app. That is outside the Volume, so copy it in.
    os.makedirs("/vol/results", exist_ok=True)
    if os.path.isdir("/results"):
        for f in os.listdir("/results"):
            shutil.copy(f"/results/{f}", f"/vol/results/{f}")
            print(f"  -> /vol/results/{f}")
    vol.commit()


@app.function(image=cluster_image, volumes=VOLUMES, cpu=8.0, memory=16384,
              timeout=7200)
def cluster(min_seq_id: float = 0.3, cov: float = 0.8):
    """Cluster proteins at `min_seq_id` identity and build a leak-free split.

    CPU only -- no GPU, so this is cheap. Writes pairs_clustered.parquet and
    meta_clustered.json into /vol/data, next to the originals.
    """
    _run(["/app/07_cluster_split.py", "--data", "/vol/data",
          "--min-seq-id", str(min_seq_id), "--cov", str(cov),
          "--threads", "8", "--workdir", "/tmp/cluster_work"])
    vol.commit()


@app.function(image=image, volumes=VOLUMES, timeout=600)
def ls():
    """What is in the Volume, and how big. Cheap -- no GPU."""
    import os
    for d in ("data", "emb", "hf", "results"):
        p = f"/vol/{d}"
        if not os.path.isdir(p):
            print(f"{d+'/':12s} (missing)")
            continue
        tot = 0
        for root, _, fs in os.walk(p):
            tot += sum(os.path.getsize(os.path.join(root, f)) for f in fs)
        print(f"{d+'/':12s} {tot/1e9:8.2f} GB")
        if d in ("data", "emb", "results"):
            for f in sorted(os.listdir(p)):
                fp = os.path.join(p, f)
                if os.path.isfile(fp):
                    print(f"    {os.path.getsize(fp)/1e6:10.1f} MB  {f}")
