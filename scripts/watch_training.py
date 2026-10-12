"""
Live terminal view of a training run. Reads the log a training job is writing and
redraws a dashboard every couple of seconds.

    .venv/Scripts/python.exe -I scripts/watch_training.py
    .venv/Scripts/python.exe -I scripts/watch_training.py --log <path> --expected 9

Ctrl-C stops the view. It does NOT stop the training.

Two levels of detail, depending on what the training job emits:

  completed runs     always available: one row per finished run, with its time
  live epoch curve   only if the job was started with --log-epochs, which makes
                     03_train_ablation.py print a line per epoch

The epoch curve is the one worth watching:

  train_loss falls steadily and more or less forever -- the model can always keep
  fitting the data it is being shown.

  dev R rises, flattens, then stops improving. The divergence between those two
  lines IS overfitting, made visible. Once dev R has failed to improve for
  `patience` epochs, early stopping fires and the run ends, keeping the weights
  from the BEST epoch rather than the last one.

Standard library only, so it starts instantly and cannot disturb the training job.
"""

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _unicode_ok():
    """Windows consoles often default to a codepage that cannot print block glyphs.
    Try to switch stdout to UTF-8; if the glyphs still will not encode, fall back
    to ASCII rather than dying with a UnicodeEncodeError mid-render."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        "▁█─●✓".encode(sys.stdout.encoding or "ascii")
        return True
    except Exception:
        return False


UNI = _unicode_ok()
SPARK = "▁▂▃▄▅▆▇█" if UNI else ".:-=+*#@"
G = ({"bar": "█", "empty": "░", "rule": "─", "run": "●",
      "done": "✓", "arrow": "←"} if UNI else
     {"bar": "#", "empty": ".", "rule": "-", "run": ">", "done": "v", "arrow": "<-"})

RE_DONE = re.compile(
    r"^\s{2}(\w+)\s+seed (\d+)\s+test R = ([-\d.]+)\s+RMSE = ([\d.]+)\s+"
    r"dev R = ([-\d.]+)\s+(\d+) ep\s+([\d,]+) params\s+([\d.]+)s")
RE_START = re.compile(r"^\s*>>> (\w+) seed (\d+) starting")
RE_EPOCH = re.compile(
    r"^\s*ep\s+(\d+)\s+train_loss ([\d.]+)\s+dev R ([-\d.]+)\s+best ([-\d.]+)\s+"
    r"patience (\d+)/(\d+)")
RE_POOL = re.compile(r"pooling=(\w+)")
RE_BASE = re.compile(r"^\s+(protein_mean_lookup|ligand_mean_lookup)\s+R = ([\d.]+)")


def sparkline(vals, width=46):
    if len(vals) < 2:
        return ""
    v = vals[-width:]
    lo, hi = min(v), max(v)
    if hi - lo < 1e-12:
        return SPARK[0] * len(v)
    n = len(SPARK) - 1
    return "".join(SPARK[min(int((x - lo) / (hi - lo) * n), n)] for x in v)


def hms(sec):
    sec = int(max(sec, 0))
    if sec >= 3600:
        return f"{sec//3600}:{(sec%3600)//60:02d}:{sec%60:02d}"
    return f"{sec//60}:{sec%60:02d}"


def newest_log():
    pats = [str(Path(os.environ.get("TEMP", "/tmp")) / "claude" / "**" / "tasks" / "*.output"),
            str(ROOT / "results" / "train_log.txt")]
    cands = []
    for p in pats:
        cands += glob.glob(p, recursive=True)
    cands = [c for c in cands if os.path.getsize(c) > 0]
    if not cands:
        return None
    looks_like_training = []
    for c in cands:
        try:
            if "test R =" in open(c, encoding="utf-8", errors="ignore").read()[-40000:]:
                looks_like_training.append(c)
        except OSError:
            pass
    return max(looks_like_training or cands, key=os.path.getmtime)


def parse(path):
    try:
        txt = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return None
    st = {"done": [], "current": None, "epochs": [], "pooling": "?", "base": {}}
    for line in txt.splitlines():
        m = RE_DONE.match(line)
        if m:
            st["done"].append(dict(
                arm=m.group(1), seed=int(m.group(2)), r=float(m.group(3)),
                rmse=float(m.group(4)), dev=float(m.group(5)), ep=int(m.group(6)),
                secs=float(m.group(8))))
            st["current"], st["epochs"] = None, []
            continue
        m = RE_START.match(line)
        if m:
            st["current"] = (m.group(1), int(m.group(2)))
            st["epochs"] = []
            continue
        m = RE_EPOCH.match(line)
        if m:
            st["epochs"].append(dict(ep=int(m.group(1)), loss=float(m.group(2)),
                                     dev=float(m.group(3)), best=float(m.group(4)),
                                     bad=int(m.group(5)), patience=int(m.group(6))))
            continue
        m = RE_BASE.match(line)
        if m:
            st["base"][m.group(1)] = float(m.group(2))
            continue
        m = RE_POOL.search(line)
        if m:
            st["pooling"] = m.group(1)
    return st


def render(st, path, t0, expected):
    W = 76
    o = []
    o.append(f"\033[1mPLAPT ablation — live\033[0m    pooling={st['pooling']}"
             if UNI else f"\033[1mPLAPT ablation - live\033[0m    pooling={st['pooling']}")
    o.append(f"\033[2m{path}\033[0m")
    o.append(G["rule"] * W)
    if st["base"]:
        b = "  ".join(f"{k.split('_')[0]} lookup R={v:.4f}" for k, v in st["base"].items())
        o.append(f"\033[2mbaselines:  {b}\033[0m")
    o.append("")
    o.append(f"  {'arm':13s} {'seed':>4s}  {'status':11s} {'ep':>4s} "
             f"{'test R':>8s} {'RMSE':>7s} {'time':>7s}")

    for d in st["done"]:
        o.append(f"  {d['arm']:13s} {d['seed']:>4d}  \033[32m{G['done']} done\033[0m"
                 f"{'':6s} {d['ep']:>4d} {d['r']:>8.4f} {d['rmse']:>7.4f} "
                 f"{hms(d['secs']):>7s}")

    if st["current"]:
        arm, seed = st["current"]
        ep = st["epochs"][-1]["ep"] if st["epochs"] else 0
        o.append(f"  {arm:13s} {seed:>4d}  \033[33m{G['run']} running\033[0m"
                 f"{'':3s} {ep:>4d}")

    eps = st["epochs"]
    if len(eps) >= 2:
        last = eps[-1]
        o.append("")
        o.append(f"  \033[1mcurrent run — epoch {last['ep']}\033[0m" if UNI
                 else f"  \033[1mcurrent run - epoch {last['ep']}\033[0m")
        o.append(f"    train loss  {sparkline([e['loss'] for e in eps])}  {last['loss']:.4f}")
        o.append(f"    dev R       {sparkline([e['dev'] for e in eps])}  {last['dev']:.4f}")
        bar = G["bar"] * last["bad"] + G["empty"] * max(last["patience"] - last["bad"], 0)
        warn = (f"  \033[33m{G['arrow']} about to early-stop\033[0m"
                if last["bad"] >= last["patience"] - 1 else "")
        o.append(f"    best dev R  {last['best']:.4f}    patience [{bar}] "
                 f"{last['bad']}/{last['patience']}{warn}")
        devs = [e["dev"] for e in eps]
        if len(devs) > 3:
            msg = ("\033[32mstill improving\033[0m" if devs[-1] >= max(devs[:-1])
                   else "\033[2mplateaued — early stopping is counting down\033[0m" if UNI
                   else "\033[2mplateaued - early stopping is counting down\033[0m")
            o.append(f"    {msg}")

    n = len(st["done"])
    frac = (n / expected) if expected else 0
    filled = int(frac * 30)
    eta = ""
    if st["done"] and expected:
        per = sum(d["secs"] for d in st["done"]) / n
        eta = (f"   eta ~{hms(per * (expected - n))}" if n < expected
               else "   \033[32mcomplete\033[0m")
    o.append("")
    o.append(f"  [{G['bar']*filled}{G['empty']*(30-filled)}] {n}/{expected} runs"
             f"   watching {hms(time.time()-t0)}{eta}")
    o.append("")
    o.append("\033[2m  Ctrl-C stops this view, not the training.\033[0m")
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="log file to watch (default: newest)")
    ap.add_argument("--expected", type=int, default=9, help="total runs expected")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    if sys.platform == "win32":
        os.system("")  # turn on ANSI escape handling in the Windows console

    path = args.log or newest_log()
    if not path:
        print("No training log found. Pass --log <path>.")
        return
    t0 = time.time()
    try:
        while True:
            st = parse(path)
            frame = render(st, path, t0, args.expected) if st else f"waiting for {path}"
            sys.stdout.write("\033[2J\033[H" + frame + "\n")
            sys.stdout.flush()
            if st and len(st["done"]) >= args.expected:
                print("\n  All expected runs finished.")
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n  stopped watching (training continues)")


if __name__ == "__main__":
    main()
