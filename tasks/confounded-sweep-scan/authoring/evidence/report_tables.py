#!/usr/bin/env python3
"""Emit the README's measured tables as markdown, straight from the data.

Printing them from the training replicates rather than retyping them is the
only way the numbers in README.md stay traceable to the simulations they came
from. Run after run_training.sh.
"""
import glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "provenance"))
import config as C

FEATS = ["pi", "tajima_d", "h12", "h2_h1", "nsl_frac"]
BG = {"pi": "bg_pi", "tajima_d": "bg_tajd", "h12": "bg_h12",
      "h2_h1": "bg_h2h1", "nsl_frac": "bg_nsl"}
RATIO = {"pi", "h12", "h2_h1"}
PRETTY = {"pi": "diversity", "tajima_d": "Tajima's D", "h12": "H12",
          "h2_h1": "H2/H1", "nsl_frac": "nSL"}
CLASSES = ["hard_complete", "hard_partial", "soft_complete", "soft_partial"]
AT = 60_000


def load():
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, "training", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        out.setdefault(str(d["realized_class"]), []).append(d)
    return out


def feature_table(data):
    print("| realized class | " + " | ".join(PRETTY[f] for f in FEATS)
          + " | n |")
    print("| --- " * (len(FEATS) + 2) + "|")
    for c in CLASSES:
        reps = data.get(c, [])
        if not reps:
            continue
        cells = []
        for k in FEATS:
            vals = []
            for d in reps:
                mid = d["mid"]
                sel = np.abs(mid - float(d["sweep_pos"])) <= AT
                v = d[k].astype(float)
                m = sel & np.isfinite(v)
                if not m.any():
                    continue
                x = float(np.median(v[m])); bg = float(d[BG[k]])
                vals.append(x / bg if (k in RATIO and bg) else x - bg)
            if not vals:
                cells.append("-"); continue
            med = float(np.median(vals))
            cells.append(f"{med:+.2f}" if k not in RATIO else f"{med:.3f}")
        print(f"| {c} | " + " | ".join(cells) + f" | {len(reps)} |")
    print()
    print("Diversity, H12 and H2/H1 as ratios to each replicate's own "
          "background; Tajima's D and the extreme-nSL fraction as "
          "differences. Median over replicates.")


def ranges(data):
    print("\nPer-class ranges of the two features that are supposed to carry "
          "the two axes on their own:")
    for k in ("h2_h1", "pi", "tajima_d", "nsl_frac"):
        print(f"\n  {PRETTY[k]}:")
        for c in CLASSES:
            vals = []
            for d in data.get(c, []):
                mid = d["mid"]
                sel = np.abs(mid - float(d["sweep_pos"])) <= AT
                v = d[k].astype(float)
                m = sel & np.isfinite(v)
                if not m.any():
                    continue
                x = float(np.median(v[m])); bg = float(d[BG[k]])
                vals.append(x / bg if (k in RATIO and bg) else x - bg)
            if vals:
                print(f"    {c:<16} {min(vals):+.3f} to {max(vals):+.3f}"
                      f"   (n={len(vals)})")


def localisation(data):
    print("\nLocalisation error: distance from the true selected site to the "
          "most extreme window of each statistic, in kb.")
    kind = {"pi": "low", "tajima_d": "low", "h12": "high",
            "h2_h1": "low", "nsl_frac": "high"}
    print(f"\n  {'class':<16}" + "".join(f"{PRETTY[f]:>14}" for f in FEATS))
    for c in CLASSES:
        reps = data.get(c, [])
        if not reps:
            continue
        row = []
        for k in FEATS:
            errs = []
            for d in reps:
                mid = d["mid"]; v = d[k].astype(float)
                ok = np.isfinite(v)
                if not ok.any():
                    continue
                i = int(np.nanargmin(v)) if kind[k] == "low" else int(np.nanargmax(v))
                errs.append(abs(float(mid[i]) - float(d["sweep_pos"])) / 1000.0)
            row.append(f"{np.median(errs):>13.0f}" if errs else f"{'-':>13}")
        print(f"  {c:<16}" + "".join(row))


if __name__ == "__main__":
    data = load()
    n = sum(len(v) for v in data.values())
    print(f"{n} single-sweep replicates, grouped by realized class\n")
    feature_table(data)
    ranges(data)
    localisation(data)
