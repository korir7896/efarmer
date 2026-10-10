#!/usr/bin/env python3
"""Per-class features AT the sweep, which is what a classifier actually sees.

analyze_training.py reports the extreme value of each statistic within a wide
window around the sweep. That is the right summary for asking "is there a
signal here" but the wrong one for classification: taking the MINIMUM H2/H1
over 1.5 Mb finds whatever dip happens to sit at the edge of the footprint,
which made genuine soft sweeps look like hard ones (0.193 against the 0.78 to
1.05 measured at the sweep itself).

Here every statistic is read from the single window containing the selected
site, and from the few windows around it, which is the regime the pipeline
works in once it has localised a peak.
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
AT_SWEEP_BP = 60_000      # windows whose midpoint is within this of the site


def load():
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, "training", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        out.setdefault(str(d["realized_class"]), []).append(d)
    return out


def at_sweep(d, feat):
    """Value of a statistic in the window(s) covering the selected site."""
    mid = d["mid"]; v = d[feat].astype(float)
    sel = (np.abs(mid - float(d["sweep_pos"])) <= AT_SWEEP_BP) & np.isfinite(v)
    if not sel.any():
        return float("nan")
    return float(np.nanmedian(v[sel]))


def main():
    data = load()
    if not data:
        print("no training replicates"); return
    print("=== statistics in the window containing the selected site ===")
    print("(pi, H12 and H2/H1 as ratios to that replicate's background; "
          "Tajima's D and nSL as differences)\n")
    print(f"{'realized class':<24}{'n':>3}" + "".join(f"{f:>12}" for f in FEATS))
    table = {}
    for lab, reps in sorted(data.items()):
        row = {}
        cells = []
        for f in FEATS:
            vals = []
            for d in reps:
                x = at_sweep(d, f)
                bg = float(d[BG[f]])
                if not np.isfinite(x):
                    continue
                vals.append(x / bg if (f in RATIO and bg) else x - bg)
            row[f] = np.array(vals, float)
            cells.append(f"{np.nanmedian(vals):>12.3f}" if vals else f"{'-':>12}")
        table[lab] = row
        print(f"{lab:<24}{len(reps):>3}" + "".join(cells))

    print("\n=== ranges (min to max across replicates) ===")
    for f in FEATS:
        print(f"\n  {f}")
        for lab in sorted(table):
            v = table[lab][f]
            v = v[np.isfinite(v)]
            if v.size:
                print(f"    {lab:<24} {v.min():>9.3f}  to {v.max():>9.3f}   "
                      f"(median {np.median(v):>8.3f}, n={v.size})")

    print("\n=== separability check for the rules the classifier uses ===")
    def report(name, lo_classes, hi_classes, feat):
        lo = np.concatenate([table[c][feat] for c in lo_classes if c in table]) \
            if any(c in table for c in lo_classes) else np.array([])
        hi = np.concatenate([table[c][feat] for c in hi_classes if c in table]) \
            if any(c in table for c in hi_classes) else np.array([])
        lo = lo[np.isfinite(lo)]; hi = hi[np.isfinite(hi)]
        if lo.size == 0 or hi.size == 0:
            print(f"  {name}: not enough data"); return
        print(f"  {name} on {feat}:")
        print(f"      {'+'.join(lo_classes):<46} max {lo.max():>8.3f}")
        print(f"      {'+'.join(hi_classes):<46} min {hi.min():>8.3f}")
        gap = hi.min() - lo.max()
        print(f"      {'SEPARABLE, gap %.3f' % gap if gap > 0 else 'OVERLAPS by %.3f' % -gap}")

    report("soft vs everything else", 
           ["hard_recent_complete", "hard_old_complete", "partial"],
           ["soft_standing"], "h2_h1")
    report("partial vs completed hard",
           ["hard_recent_complete", "hard_old_complete"], ["partial"], "pi")
    report("old vs recent (hard only)",
           ["hard_old_complete"], ["hard_recent_complete"], "h12")


if __name__ == "__main__":
    main()
