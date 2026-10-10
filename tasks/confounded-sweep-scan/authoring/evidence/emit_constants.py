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
CLASSES = ["hard_complete", "hard_partial", "soft_complete", "soft_partial"]


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
    # The estimator is  s_hat = width * local_rate / k[class].  Per class,
    # k is the median of width*r/s over that class's replicates; the spread of
    # that ratio is the estimator's multiplicative error, and the rank
    # correlation below is what the verifier's ordering gate measures.
    print(f"# {'class':<16}{'n':>3}{'s range':>16}{'k':>9}"
          f"{'p10/p90 of w*r/s':>20}")
    print("CLASS_WIDTH_CONSTANT = {")
    kk = {}
    for c in CLASSES:
        rs = [r for r in rows if r["cls"] == c and r["width"] > 0]
        if not rs:
            continue
        ratio = np.array([r["width"] * C.BASE_REC_RATE / r["s_true"] for r in rs])
        k = float(np.median(ratio))
        kk[c] = k
        st = sorted(r["s_true"] for r in rs)
        print(f'    "{c}": {k:.4f},'.ljust(30)
              + f"# n={len(rs):<3} s {st[0]:.4f}-{st[-1]:.4f}"
              + f"  spread {np.percentile(ratio,10)/k:.2f}-"
              + f"{np.percentile(ratio,90)/k:.2f}x")
    print("}")

    # Achieved accuracy of that estimator on the calibration set, which is
    # the number the verifier's s gates have to be set below.
    got, tru = [], []
    for r in rows:
        k = kk.get(r["cls"])
        if k is None or r["width"] <= 0:
            continue
        got.append(r["width"] * C.BASE_REC_RATE / k)
        tru.append(r["s_true"])
    if got:
        got_a, tru_a = np.array(got), np.array(tru)
        err = np.maximum(got_a / tru_a, tru_a / got_a)
        for f in (2.0, 2.5, 3.0):
            print(f"#   within a factor of {f}: "
                  f"{int((err <= f).sum())}/{len(err)} "
                  f"= {(err <= f).mean():.3f}")
        print(f"#   rank correlation with truth: {_spearman(got_a, tru_a):.3f} "
              f"over {len(got_a)} replicates")

    print("\n# --- diversity-recovery clock ---")
    # One power law over all hard_complete replicates rather than a constant
    # bias per class. The classes no longer split by age, so a single
    # hard_complete class now spans completion times from about 100 to about
    # 2100 generations and the relation between residual diversity and age is
    # not proportional over that range.
    rs = [r for r in rows if r["cls"] == "hard_complete" and r["completed"]
          and np.isfinite(r["t_true"]) and np.isfinite(r["pi_min_ratio"])
          and r["pi_min_ratio"] > 0]
    if len(rs) >= 4:
        x = np.log(np.array([r["pi_min_ratio"] for r in rs]))
        y = np.log(np.array([r["t_true"] for r in rs]))
        B, logA = np.polyfit(x, y, 1)
        A = float(np.exp(logA)) / C.N_CURRENT
        pred = np.exp(logA + B * x)
        tru_t = np.array([r["t_true"] for r in rs])
        rel = np.abs(pred - tru_t) / tru_t
        within = np.abs(pred - tru_t) <= np.maximum(0.5 * tru_t, 150.0)
        print(f"TIME_CLOCK_A = {A:.5g}   # t_hat = A * N_current * pi_min_ratio**B")
        print(f"TIME_CLOCK_B = {B:.5g}")
        print(f"#   fitted on n={len(rs)} hard_complete replicates, "
              f"t_true {int(tru_t.min())}-{int(tru_t.max())}")
        print(f"#   median relative error {np.median(rel):.3f}; "
              f"inside max(50%, 150 gens) on {int(within.sum())}/{len(rs)}")
        print(f"#   rank correlation with truth: {_spearman(pred, tru_t):.3f}")
    else:
        print("# not enough hard_complete replicates to fit the clock")


def _rankdata(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="stable")
    r = np.empty(v.size, float)
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def _spearman(a, b):
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    main()
