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

    # The two axes of the class scheme, one feature at a time. Each is the
    # statistic that is supposed to carry that axis on its own; the point of
    # the report is that none of them does.
    report("soft vs hard, on Tajima's D",
           ["hard_complete", "hard_partial"],
           ["soft_complete", "soft_partial"], "tajima_d")
    report("soft vs hard, on H2/H1",
           ["hard_complete", "hard_partial"],
           ["soft_complete", "soft_partial"], "h2_h1")
    report("complete vs partial, on diversity",
           ["hard_complete", "soft_complete"],
           ["hard_partial", "soft_partial"], "pi")
    report("complete vs partial, on extreme-nSL fraction",
           ["hard_complete", "soft_complete"],
           ["hard_partial", "soft_partial"], "nsl_frac")

    # ------------------------------------------------------------------
    # estimator calibration
    # ------------------------------------------------------------------
    SEARCH = 2_500_000
    print("\n=== s from the width of the diversity trough ===")
    print("w is the span where diversity is below half its background level.")
    print(f"{'realized class':<24}{'n':>3}{'s_true':>9}{'w (kb)':>9}{'w*r/s':>9}")
    allratio = []
    for lab, reps in sorted(data.items()):
        rows = []
        for d in reps:
            mid = d["mid"]; v = d["pi"].astype(float); bg = float(d["bg_pi"])
            pos = float(d["sweep_pos"])
            near = (np.abs(mid - pos) <= SEARCH) & np.isfinite(v)
            below = near & (v < 0.5 * bg)
            if not below.any():
                continue
            w = float(mid[below].max() - mid[below].min())
            rows.append((w, float(d["s_true"])))
        if not rows:
            continue
        w = np.median([r[0] for r in rows]); st = np.median([r[1] for r in rows])
        ratio = w * C.BASE_REC_RATE / st
        allratio.append(ratio)
        print(f"{lab:<24}{len(rows):>3}{st:>9.3f}{w/1000:>9.0f}{ratio:>9.3f}")
    if allratio:
        print(f"{'':<24}{'':>3}{'':>9}{'mean':>9}{np.mean(allratio):>9.3f}")

    print("\n=== completion time from diversity recovery ===")
    print("t_hat = 2*N_current*pi_min/pi_background, before bias removal.")
    print(f"{'realized class':<24}{'n':>3}{'true':>8}{'t_hat':>9}{'bias':>8}")
    for lab, reps in sorted(data.items()):
        rows = []
        for d in reps:
            if "completed" not in d.files or not bool(d["completed"].item()):
                continue
            mid = d["mid"]; v = d["pi"].astype(float); bg = float(d["bg_pi"])
            pos = float(d["sweep_pos"])
            near = (np.abs(mid - pos) <= SEARCH) & np.isfinite(v)
            if not near.any() or not bg:
                continue
            t_hat = 2.0 * C.N_CURRENT * float(v[near].min()) / bg
            rows.append((float(d["completion_gens_ago"]), t_hat))
        if not rows:
            continue
        tr = np.array([r[0] for r in rows]); th = np.array([r[1] for r in rows])
        print(f"{lab:<24}{len(rows):>3}{np.median(tr):>8.0f}{np.median(th):>9.0f}"
              f"{np.median(th / tr):>8.2f}")


if __name__ == "__main__":
    main()
