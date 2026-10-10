#!/usr/bin/env python3
"""Choose the estimators of s and of completion time, by held-out score.

The first calibration of the redesigned zoo showed the old estimators were
not good enough for it. Trough width times the local rate, with one constant
per class, put 74% of replicates inside a factor of 2.5 but ranked them at
only 0.32 -- fine when every class had a deep trough and s spanned a factor
of two, hopeless now that soft sweeps barely depress diversity and s spans a
factor of nine. The recovery clock was inside tolerance on 9 of 16.

So the predictors are chosen here rather than assumed, against the measured
leave-one-out score of each candidate on the training replicates. Nothing in
this file looks at the shipped replicate.
"""
import glob, itertools, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "provenance"))
sys.path.insert(0, os.path.join(TASK, "tests"))
import config as C
from grading import spearman, time_tolerance, GATES

AT = 60_000
WIDE = 1_000_000
CLASSES = ["hard_complete", "hard_partial", "soft_complete", "soft_partial"]


def smooth(v, k=15):
    out = np.full(v.size, np.nan); h = k // 2
    for i in range(v.size):
        a, b = max(0, i - h), min(v.size, i + h + 1)
        seg = v[a:b][np.isfinite(v[a:b])]
        if seg.size:
            out[i] = np.median(seg)
    return out


def contiguous_width(mid, v, bg, centre, frac):
    """Span of the contiguous run around the centre below floor+frac*(bg-floor)."""
    vs = smooth(v)
    near = (np.abs(mid - centre) <= 1_200_000) & np.isfinite(vs)
    if not near.any():
        return 0.0
    floor = float(vs[near].min())
    level = floor + frac * (bg - floor)
    order = np.argsort(mid); m, vv = mid[order], vs[order]
    below = np.isfinite(vv) & (vv < level)
    if not below.any():
        return 0.0
    i0 = int(np.argmin(np.abs(m - centre)))
    if not below[i0]:
        cand = np.where(below)[0]
        i0 = int(cand[np.argmin(np.abs(m[cand] - centre))])
        if abs(m[i0] - centre) > 600_000:
            return 0.0
    lo = hi = i0
    while lo - 1 >= 0 and below[lo - 1]:
        lo -= 1
    while hi + 1 < len(m) and below[hi + 1]:
        hi += 1
    return float(m[hi] - m[lo])


def high_width(mid, v, bg, centre, mult):
    """Span of the contiguous run around the centre ABOVE mult*background."""
    vs = smooth(v)
    order = np.argsort(mid); m, vv = mid[order], vs[order]
    above = np.isfinite(vv) & (vv > mult * bg)
    if not above.any():
        return 0.0
    i0 = int(np.argmin(np.abs(m - centre)))
    if not above[i0]:
        cand = np.where(above)[0]
        i0 = int(cand[np.argmin(np.abs(m[cand] - centre))])
        if abs(m[i0] - centre) > 600_000:
            return 0.0
    lo = hi = i0
    while lo - 1 >= 0 and above[lo - 1]:
        lo -= 1
    while hi + 1 < len(m) and above[hi + 1]:
        hi += 1
    return float(m[hi] - m[lo])


def rows(subdir="training"):
    """Feature rows from one or more calibration directories.

    `subdir` may be a single name or an iterable of names. The shipped
    constants are fitted on BOTH training/ and training_holdout/, so the
    configuration space they are validated across is far wider than the ten
    the shipped replicate uses -- which is what makes leave-one-
    configuration-out scoring mean something.
    """
    subdirs = [subdir] if isinstance(subdir, str) else list(subdir)
    out = []
    files = []
    for sd in subdirs:
        files += sorted(glob.glob(os.path.join(HERE, sd, "*.npz")))
    for f in files:
        d = np.load(f, allow_pickle=True)
        # Configuration identity, from the filename tag run_training.sh
        # assigns. Leaving out one REPLICATE when three replicates share a
        # configuration measures almost nothing: the shipped sweeps sit at
        # these very configurations, so a predictor that has memorised the
        # four (s, age) pairs scores perfectly and generalises not at all.
        # Leaving out a whole configuration is the honest test.
        cfg_tag = os.path.basename(f).rsplit("_", 1)[0]
        shipped_cfg = "_ho_" not in cfg_tag
        mid = d["mid"].astype(float)
        centre = float(d["sweep_pos"])
        near = np.abs(mid - centre) <= AT
        wide = np.abs(mid - centre) <= WIDE
        pi = d["pi"].astype(float); bgpi = float(d["bg_pi"])
        h12 = d["h12"].astype(float); bgh12 = float(d["bg_h12"])
        h2h1 = d["h2_h1"].astype(float); bgh2h1 = float(d["bg_h2h1"])
        nsl = d["nsl_frac"].astype(float); bgnsl = float(d["bg_nsl"])
        tajd = d["tajima_d"].astype(float); bgtajd = float(d["bg_tajd"])
        nsnp = d["n_snps"].astype(float)
        far = np.abs(mid - centre) > 2_500_000

        def med(v, m):
            x = v[m & np.isfinite(v)]
            return float(np.median(x)) if x.size else np.nan

        r = dict(
            cfg=cfg_tag,
            shipped_cfg=shipped_cfg,
            cls=str(d["realized_class"]),
            s_true=float(d["s_true"]),
            completed=bool(d["completed"].item()) if "completed" in d.files else False,
            t_true=(float(d["completion_gens_ago"])
                    if "completion_gens_ago" in d.files else np.nan),
            pi_ratio=med(pi, near) / bgpi if bgpi else np.nan,
            pi_min_ratio=(float(np.nanmin(pi[wide])) / bgpi) if bgpi else np.nan,
            h12_ratio=med(h12, near) / bgh12 if bgh12 else np.nan,
            h2h1_ratio=med(h2h1, near) / bgh2h1 if bgh2h1 else np.nan,
            nsl_delta=med(nsl, near) - bgnsl,
            tajd_delta=med(tajd, near) - bgtajd,
            snp_ratio=med(nsnp, near) / med(nsnp, far) if med(nsnp, far) else np.nan,
            w_pi_50=contiguous_width(mid, pi, bgpi, centre, 0.50),
            w_pi_75=contiguous_width(mid, pi, bgpi, centre, 0.75),
            w_h12_2=high_width(mid, h12, bgh12, centre, 2.0),
            w_h12_15=high_width(mid, h12, bgh12, centre, 1.5),
            w_snp_50=contiguous_width(mid, nsnp, med(nsnp, far), centre, 0.50),
        )
        out.append(r)
    return out


def held_out_loglinear(X, y, groups=None):
    """Held-out predictions of a log-linear fit of y on X (both logged).

    With `groups`, every row sharing a group is held out together, so the fit
    never sees another replicate of the same configuration.
    """
    n = len(y)
    pred = np.full(n, np.nan)
    A = np.column_stack([np.ones(n)] + [X[:, j] for j in range(X.shape[1])])
    folds = ([np.array(groups) == g for g in sorted(set(groups))]
             if groups is not None
             else [np.arange(n) == i for i in range(n)])
    for held in folds:
        m = ~held
        if m.sum() <= A.shape[1]:
            continue
        try:
            coef, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred[held] = A[held] @ coef
    return pred


def score_s(cands, data):
    print("\n=== s: leave-one-CONFIGURATION-out, per class, "
          "log-linear in the named predictors ===")
    f = GATES["s_within_factor"]
    best = {}
    for c in CLASSES:
        rs = [r for r in data if r["cls"] == c]
        print(f"\n  {c}  (n={len(rs)}, "
              f"s {min(r['s_true'] for r in rs):.4f}-"
              f"{max(r['s_true'] for r in rs):.4f})")
        results = []
        for combo in cands:
            keep = [r for r in rs
                    if all(np.isfinite(r[k]) and r[k] > 0 for k in combo)]
            if len(keep) < len(combo) + 3:
                continue
            X = np.log(np.array([[r[k] for k in combo] for r in keep], float))
            y = np.log(np.array([r["s_true"] for r in keep], float))
            p = held_out_loglinear(X, y, [r["cfg"] for r in keep])
            ok = np.isfinite(p)
            if ok.sum() < 3:
                continue
            err = np.exp(np.abs(p[ok] - y[ok]))
            rate = float((err <= f).mean())
            rho = spearman(list(np.exp(p[ok])), list(np.exp(y[ok])))
            results.append((rate, rho, combo, int(ok.sum()), len(keep)))
        results.sort(key=lambda t: (-t[0], -t[1]))
        for rate, rho, combo, n_ok, n_keep in results[:6]:
            print(f"    {'+'.join(combo):<30} within {f}: {rate:.2f}  "
                  f"rho {rho:+.2f}  (n={n_ok}/{n_keep})")
        if results:
            best[c] = results[0][2]
    return best


def score_t(cands, data):
    print("\n=== completion time: leave-one-CONFIGURATION-out over hard_complete ===")
    rs = [r for r in data if r["cls"] == "hard_complete" and r["completed"]
          and np.isfinite(r["t_true"])]
    print(f"  n={len(rs)}, t_true {min(r['t_true'] for r in rs):.0f}-"
          f"{max(r['t_true'] for r in rs):.0f}")
    results = []
    for combo in cands:
        keep = [r for r in rs if all(np.isfinite(r[k]) and r[k] > 0 for k in combo)]
        if len(keep) < len(combo) + 3:
            continue
        X = np.log(np.array([[r[k] for k in combo] for r in keep], float))
        y = np.log(np.array([r["t_true"] for r in keep], float))
        p = held_out_loglinear(X, y, [r["cfg"] for r in keep])
        ok = np.isfinite(p)
        if ok.sum() < 3:
            continue
        tt = np.exp(y[ok]); th = np.exp(p[ok])
        inside = np.abs(th - tt) <= np.array([time_tolerance(x) for x in tt])
        results.append((float(inside.mean()), spearman(list(th), list(tt)),
                        combo, int(ok.sum())))
    results.sort(key=lambda t: (-t[0], -t[1]))
    for rate, rho, combo, n_ok in results[:10]:
        print(f"    {'+'.join(combo):<36} inside tolerance: {rate:.2f}  "
              f"rho {rho:+.2f}  (n={n_ok})")


CHOSEN_S = {
    "hard_complete": ("pi_ratio", "w_pi_50"),
    "hard_partial":  ("pi_ratio", "w_h12_15"),
    "soft_complete": ("pi_ratio", "w_pi_75"),
    "soft_partial":  ("pi_ratio", "w_h12_2"),
}
CHOSEN_T = ("h12_ratio", "w_pi_75")


def fit(data, combo, target, cls=None, completed=False):
    rs = [r for r in data if (cls is None or r["cls"] == cls)
          and (not completed or r["completed"])
          and np.isfinite(r[target]) and r[target] > 0
          and all(np.isfinite(r[k]) and r[k] > 0 for k in combo)]
    X = np.log(np.array([[r[k] for k in combo] for r in rs], float))
    y = np.log(np.array([r[target] for r in rs], float))
    A = np.column_stack([np.ones(len(rs))] + [X[:, j] for j in range(X.shape[1])])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    held = held_out_loglinear(X, y, [r["cfg"] for r in rs])
    return rs, coef, y, held


def emit(data):
    print("\n=== CHOSEN ESTIMATORS ===")
    print("# s_hat = exp(a) * prod(feature_i ** b_i), one fit per class.")
    print("S_MODEL = {")
    pooled_hat, pooled_true = [], []
    for c in CLASSES:
        combo = CHOSEN_S[c]
        rs, coef, y, held = fit(data, combo, "s_true", cls=c)
        ok = np.isfinite(held)
        err = np.exp(np.abs(held[ok] - y[ok]))
        rate = float((err <= GATES["s_within_factor"]).mean())
        print(f'    "{c}": dict(feats={list(combo)!r},'
              f' coef=[{", ".join(f"{v:.5g}" for v in coef)}]),'
              f"   # n={len(rs)}, held-out within "
              f"{GATES['s_within_factor']}: {rate:.2f}")
        pooled_hat += list(np.exp(held[ok])); pooled_true += list(np.exp(y[ok]))
    print("}")
    err = np.maximum(np.array(pooled_hat) / np.array(pooled_true),
                     np.array(pooled_true) / np.array(pooled_hat))
    print(f"#   POOLED held-out: within {GATES['s_within_factor']} on "
          f"{int((err <= GATES['s_within_factor']).sum())}/{len(err)} = "
          f"{(err <= GATES['s_within_factor']).mean():.3f}, "
          f"rank correlation {spearman(pooled_hat, pooled_true):+.3f} "
          f"(gates: {GATES['min_s_accuracy']:.2f} and "
          f"{GATES['min_s_rank_corr']:.2f})")

    rs, coef, y, held = fit(data, CHOSEN_T, "t_true",
                            cls="hard_complete", completed=True)
    ok = np.isfinite(held)
    tt = np.exp(y[ok]); th = np.exp(held[ok])
    inside = np.abs(th - tt) <= np.array([time_tolerance(x) for x in tt])
    print(f"\nT_MODEL = dict(feats={list(CHOSEN_T)!r}, "
          f"coef=[{', '.join(f'{v:.5g}' for v in coef)}])")
    print(f"#   n={len(rs)} hard_complete, t_true {int(tt.min())}-{int(tt.max())}; "
          f"held-out inside tolerance on {int(inside.sum())}/{len(tt)} = "
          f"{inside.mean():.2f}, rank correlation {spearman(list(th), list(tt)):+.3f}")


if __name__ == "__main__":
    data = rows()
    print(f"{len(data)} replicates")
    keys = ["pi_ratio", "pi_min_ratio", "h12_ratio", "snp_ratio",
            "w_pi_50", "w_pi_75", "w_h12_2", "w_h12_15", "w_snp_50"]
    singles = [(k,) for k in keys]
    pairs = [c for c in itertools.combinations(keys, 2)]
    if "--emit" in sys.argv:
        emit(data)
    else:
        score_s(singles + pairs, data)
        score_t(singles + pairs, data)
        emit(data)
