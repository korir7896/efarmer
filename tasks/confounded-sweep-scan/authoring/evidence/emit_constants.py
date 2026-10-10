#!/usr/bin/env python3
"""Emit the calibration constants the reference pipeline needs.

Prints standardization and per-class centroids for the nearest-centroid
classifier, a robust trough-width calibration for s, and the bias of the
diversity-recovery clock per datable class.
"""
import glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "provenance"))
import config as C

FEATS = ["pi", "tajima_d", "h2_h1", "nsl_frac"]
BG = {"pi": "bg_pi", "tajima_d": "bg_tajd", "h2_h1": "bg_h2h1",
      "nsl_frac": "bg_nsl"}
RATIO = {"pi", "h2_h1"}
AT = 60_000
CLASSES = ["hard_recent_complete", "hard_old_complete", "soft_standing", "partial"]


def contiguous_width(mid, v, bg, centre, frac=0.5):
    """Width of the CONTIGUOUS run of sub-threshold windows around the centre.

    Taking the span between the outermost sub-threshold windows in a wide
    search region is not a measurement of the sweep: a few scattered dips
    anywhere in the region inflate it, which is why that version returned
    3.5-4.5 Mb troughs for sweeps whose footprint should be about 1 Mb.
    """
    order = np.argsort(mid)
    m, vv = mid[order], v[order]
    below = np.isfinite(vv) & (vv < frac * bg)
    if not below.any():
        return 0.0
    i0 = int(np.argmin(np.abs(m - centre)))
    if not below[i0]:
        # nearest sub-threshold window to the centre
        cand = np.where(below)[0]
        i0 = int(cand[np.argmin(np.abs(m[cand] - centre))])
        if abs(m[i0] - centre) > 500_000:
            return 0.0
    lo = i0
    while lo - 1 >= 0 and below[lo - 1]:
        lo -= 1
    hi = i0
    while hi + 1 < len(m) and below[hi + 1]:
        hi += 1
    return float(m[hi] - m[lo])


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "training", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        mid = d["mid"]; sel = np.abs(mid - float(d["sweep_pos"])) <= AT
        feat = {}
        ok = True
        for k in FEATS:
            v = d[k].astype(float); m = sel & np.isfinite(v)
            if not m.any():
                ok = False; break
            x = float(np.median(v[m])); bg = float(d[BG[k]])
            feat[k] = x / bg if (k in RATIO and bg) else x - bg
        if not ok:
            continue
        w = contiguous_width(mid, d["pi"].astype(float), float(d["bg_pi"]),
                             float(d["sweep_pos"]))
        vpi = d["pi"].astype(float)
        near = (np.abs(mid - float(d["sweep_pos"])) <= 1_000_000) & np.isfinite(vpi)
        pi_min_ratio = float(vpi[near].min()) / float(d["bg_pi"]) if near.any() else np.nan
        rows.append(dict(cls=str(d["realized_class"]), feat=feat, width=w,
                         s_true=float(d["s_true"]), pi_min_ratio=pi_min_ratio,
                         completed=bool(d["completed"].item()) if "completed" in d.files else False,
                         t_true=(float(d["completion_gens_ago"])
                                 if "completion_gens_ago" in d.files and
                                 bool(d["completed"].item()) else np.nan)))

    X = np.array([[r["feat"][k] for k in FEATS] for r in rows], float)
    y = np.array([r["cls"] for r in rows])
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)

    print("# --- nearest-centroid classifier, calibrated on "
          f"{len(y)} independent single-sweep replicates ---")
    print(f"CLASS_FEATURES = {FEATS!r}")
    print("FEATURE_MEAN = [" + ", ".join(f"{v:.6g}" for v in mu) + "]")
    print("FEATURE_SD = [" + ", ".join(f"{v:.6g}" for v in sd) + "]")
    print("CLASS_CENTROIDS = {")
    Z = (X - mu) / sd
    for c in CLASSES:
        m = y == c
        if m.sum():
            print(f'    "{c}": [' + ", ".join(f"{v:.4f}" for v in Z[m].mean(0))
                  + f"],   # n={int(m.sum())}")
    print("}")

    print("\n# --- s from the contiguous diversity trough ---")
    print(f"# {'class':<24}{'n':>3}{'s_true':>9}{'width kb':>10}{'w*r/s':>9}")
    ratios = []
    for c in CLASSES:
        rs = [r for r in rows if r["cls"] == c and r["width"] > 0]
        if not rs:
            continue
        w = np.median([r["width"] for r in rs])
        st = np.median([r["s_true"] for r in rs])
        ratio = w * C.BASE_REC_RATE / st
        ratios += [r["width"] * C.BASE_REC_RATE / r["s_true"] for r in rs]
        print(f"# {c:<24}{len(rs):>3}{st:>9.3f}{w/1000:>10.0f}{ratio:>9.3f}")
    print(f"WIDTH_RATE_OVER_S = {np.median(ratios):.4f}   "
          f"# median over all {len(ratios)} replicates, "
          f"spread {np.percentile(ratios,10):.3f}-{np.percentile(ratios,90):.3f}")

    print("\n# --- diversity-recovery clock, bias per datable class ---")
    print("TIME_BIAS = {")
    for c in ("hard_recent_complete", "hard_old_complete"):
        rs = [r for r in rows if r["cls"] == c and r["completed"]
              and np.isfinite(r["t_true"]) and np.isfinite(r["pi_min_ratio"])]
        if not rs:
            continue
        b = np.median([(2.0 * C.N_CURRENT * r["pi_min_ratio"]) / r["t_true"] for r in rs])
        print(f'    "{c}": {b:.3f},   # n={len(rs)}')
    print("}")


if __name__ == "__main__":
    main()
