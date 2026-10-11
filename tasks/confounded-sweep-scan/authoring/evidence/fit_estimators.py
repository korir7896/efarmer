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


def grid_step(mid) -> float:
    """One step of the scan grid, used as the floor on every width.

    A contiguous run of windows is measured as the distance between the
    centres of its first and last window, so a run ONE window wide measures
    zero, and so does a statistic that never crosses its threshold at all.
    Both mean "narrower than this scan can resolve", which is a measurement,
    not a missing value -- but these estimators are log-linear in the widths,
    and log(0) discards the replicate from the fit and leaves the shipped
    model unable to score that sweep. Reporting one step instead keeps the
    quantity monotone in the right direction (a weaker sweep has a narrower
    footprint) and keeps it finite. The floor is read off the grid rather
    than hard-coded so it stays correct if the scan resolution changes.
    """
    m = np.sort(np.asarray(mid, float))
    d = np.diff(m)
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def contiguous_width(mid, v, bg, centre, frac):
    """Span of the contiguous run around the centre below floor+frac*(bg-floor)."""
    vs = smooth(v)
    step = grid_step(mid)
    near = (np.abs(mid - centre) <= 1_200_000) & np.isfinite(vs)
    if not near.any():
        return step
    floor = float(vs[near].min())
    level = floor + frac * (bg - floor)
    order = np.argsort(mid); m, vv = mid[order], vs[order]
    below = np.isfinite(vv) & (vv < level)
    if not below.any():
        return step
    i0 = int(np.argmin(np.abs(m - centre)))
    if not below[i0]:
        cand = np.where(below)[0]
        i0 = int(cand[np.argmin(np.abs(m[cand] - centre))])
        if abs(m[i0] - centre) > 600_000:
            return step
    lo = hi = i0
    while lo - 1 >= 0 and below[lo - 1]:
        lo -= 1
    while hi + 1 < len(m) and below[hi + 1]:
        hi += 1
    return max(float(m[hi] - m[lo]), step)


def high_width(mid, v, bg, centre, mult):
    """Span of the contiguous run around the centre ABOVE mult*background."""
    vs = smooth(v)
    step = grid_step(mid)
    order = np.argsort(mid); m, vv = mid[order], vs[order]
    above = np.isfinite(vv) & (vv > mult * bg)
    if not above.any():
        return step
    i0 = int(np.argmin(np.abs(m - centre)))
    if not above[i0]:
        cand = np.where(above)[0]
        i0 = int(cand[np.argmin(np.abs(m[cand] - centre))])
        if abs(m[i0] - centre) > 600_000:
            return step
    lo = hi = i0
    while lo - 1 >= 0 and above[lo - 1]:
        lo -= 1
    while hi + 1 < len(m) and above[hi + 1]:
        hi += 1
    return max(float(m[hi] - m[lo]), step)


LOCATE_RADIUS = 1_500_000


def _scan_centre(mid: np.ndarray, h12: np.ndarray, true_pos: float) -> float:
    """The position the shipped locator would report for this sweep."""
    from pipeline.detect import LOCATOR_SMOOTH_WINDOWS, _smooth_median
    order = np.argsort(mid)
    m = mid[order]
    sm = _smooth_median(h12[order].astype(float), LOCATOR_SMOOTH_WINDOWS)
    near = np.abs(m - true_pos) <= LOCATE_RADIUS
    cand = np.where(near & np.isfinite(sm))[0]
    if cand.size == 0:
        return float(true_pos)
    return float(m[cand[int(np.argmax(sm[cand]))]])


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
        # Where the SCAN would say the sweep is, not where it actually is.
        #
        # This matters more than it looks. Every feature below is read within
        # AT base pairs of `centre`, and the shipped pipeline reads them at
        # the position it called -- a median of 78 kb from the true site on
        # the shipped replicate and as much as 302 kb for the strongest
        # sweep, whose footprint is over a megabase. Measuring the
        # calibration features at the true site instead fits the models on a
        # quantity the pipeline never sees, and the gap shows up as a
        # classifier that scores 0.758 held out over these replicates and
        # 0.42 on the shipped genome.
        #
        # It is not a background effect, which was the first guess: the
        # shipped genome's background and these replicates' agree to within
        # 5 to 18 percent on every statistic, so ratios to background
        # transfer fine. It is the offset.
        #
        # The locator is the same one detect._summarize uses -- the maximum
        # of H12 smoothed over LOCATOR_SMOOTH_WINDOWS windows -- restricted
        # to within LOCATE_RADIUS of the true site, which stands in for "the
        # scan formed a cluster around this sweep". Whether a cluster forms
        # at all is a detection question and is not what these models are
        # for.
        centre = _scan_centre(mid, d["h12"].astype(float),
                              float(d["sweep_pos"]))
        near = np.abs(mid - centre) <= AT
        wide = np.abs(mid - centre) <= WIDE
        pi = d["pi"].astype(float); bgpi = float(d["bg_pi"])
        h12 = d["h12"].astype(float); bgh12 = float(d["bg_h12"])
        h2h1 = d["h2_h1"].astype(float); bgh2h1 = float(d["bg_h2h1"])
        nsl = d["nsl_frac"].astype(float); bgnsl = float(d["bg_nsl"])
        tajd = d["tajima_d"].astype(float); bgtajd = float(d["bg_tajd"])
        nsnp = d["n_snps"].astype(float)
        far = np.abs(mid - centre) > 2_500_000
        # local_rate is written by make_training.py; replicates produced
        # before cold-spot calibration existed do not carry it and were all
        # at the base rate.
        local_rate = float(d["local_rate"]) if "local_rate" in d.files \
            else float(C.BASE_REC_RATE)
        rate_scale = local_rate / float(C.BASE_REC_RATE)

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
            # Widths in BASE PAIRS, with the local rate beside them as its
            # own feature. See characterize.shape_features for why dividing
            # by the rate ratio over-corrects on a finite cold spot.
            w_pi_50=contiguous_width(mid, pi, bgpi, centre, 0.50),
            w_pi_75=contiguous_width(mid, pi, bgpi, centre, 0.75),
            w_h12_2=high_width(mid, h12, bgh12, centre, 2.0),
            w_h12_15=high_width(mid, h12, bgh12, centre, 1.5),
            w_snp_50=contiguous_width(mid, nsnp, med(nsnp, far), centre, 0.50),
            rate_ratio=rate_scale,
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
    return results[0][2] if results else None


# Selected by leave-one-configuration-out score over all 37 calibration
# configurations, 20 of which the shipped replicate does not use, preferring
# predictors that are defined on every replicate so the estimator never has
# to fall back. Selecting over the ten shipped configurations alone picked
# (pi_ratio, w_pi_75) for soft_complete, which scores a rank correlation of
# 0.10-0.29 on the wider set: it was chosen on noise, and refitting it away
# from those ten points returns an almost flat model.
# Chosen on held-out accuracy first and rank correlation second, from the
# tables score_s() and score_t() print over all 120 replicates, restricted to
# predictors that are strictly positive on every replicate so the shipped
# estimator never meets a sweep it cannot score.
#
# One of the four carries rate_ratio and three do not, which deserves an
# explanation because three of the fifteen shipped sweeps sit inside a wide
# recombination cold spot and the cold-spot calibration replicates exist for
# exactly this question. Measured, the rate term wins outright for
# soft_complete and buys nothing elsewhere. That is not evidence against the
# rate mattering: the calibration set is 31 base-rate configurations against
# 9 cold-spot ones, so a predictor that is identically 1.0 in three quarters
# of the rows spends a degree of freedom to explain nothing there, and only a
# class whose cold-spot error is large enough pays that back.
#
# Which objective decides is settled by s_gate_design.py, which scores both
# sets over the composition of the SHIPPED zoo -- 3 cold-spot sweeps out of
# 15 -- rather than over the calibration average. CHOSEN_S_RATE below is the
# alternative it is compared against, so the choice is a measurement and the
# losing option stays visible instead of being deleted.
CHOSEN_S = {
    "hard_complete": ("pi_min_ratio", "w_pi_75"),              # 0.93, rho +0.33
    "hard_partial":  ("snp_ratio", "w_h12_15"),                # 0.94, rho +0.75
    "soft_complete": ("w_pi_50", "w_h12_15"),                  # 0.91, rho +0.24
    "soft_partial":  ("h12_ratio", "w_pi_50", "rate_ratio"),   # 0.92, rho +0.41
}
# The best model per class that DOES carry the local rate, for the
# shipped-composition comparison in s_gate_design.py.
CHOSEN_S_RATE = {
    "hard_complete": ("pi_ratio", "w_pi_75", "rate_ratio"),
    "hard_partial":  ("pi_ratio", "w_h12_2", "rate_ratio"),
    "soft_complete": ("w_pi_50", "w_h12_15", "rate_ratio"),
    "soft_partial":  ("h12_ratio", "w_pi_50", "rate_ratio"),
}
CHOSEN_T = ("h12_ratio", "w_pi_50", "rate_ratio")


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


def emit(data, chosen_s=None, chosen_t=None, tag=""):
    chosen_s = chosen_s or CHOSEN_S
    chosen_t = chosen_t or CHOSEN_T
    print(f"\n=== CHOSEN ESTIMATORS{tag} ===")
    print("# s_hat = exp(a) * prod(feature_i ** b_i), one fit per class.")
    print("S_MODEL = {")
    pooled_hat, pooled_true = [], []
    for c in CLASSES:
        combo = chosen_s[c]
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

    rs, coef, y, held = fit(data, chosen_t, "t_true",
                            cls="hard_complete", completed=True)
    ok = np.isfinite(held)
    tt = np.exp(y[ok]); th = np.exp(held[ok])
    inside = np.abs(th - tt) <= np.array([time_tolerance(x) for x in tt])
    print(f"\nT_MODEL = dict(feats={list(chosen_t)!r}, "
          f"coef=[{', '.join(f'{v:.5g}' for v in coef)}])")
    print(f"#   n={len(rs)} hard_complete, t_true {int(tt.min())}-{int(tt.max())}; "
          f"held-out inside tolerance on {int(inside.sum())}/{len(tt)} = "
          f"{inside.mean():.2f}, rank correlation {spearman(list(th), list(tt)):+.3f}")


ALL_DIRS = ("training", "training_holdout")

if __name__ == "__main__":
    dirs = "training" if "--shipped-only" in sys.argv else ALL_DIRS
    data = rows(dirs)
    if "--holdout-only" in sys.argv:
        data = [r for r in data if not r["shipped_cfg"]]
    print(f"calibration set: {len(data)} replicates, "
          f"{len(set(r['cfg'] for r in data))} configurations, "
          f"{len(set(r['cfg'] for r in data if not r['shipped_cfg']))} not "
          f"used by the shipped replicate")
    # Exactly the predictors solution/pipeline/characterize.py computes, and
    # nothing else. A model fitted here on a quantity the pipeline does not
    # produce cannot be used, and the failure is silent: _loglinear() returns
    # None for the missing feature and estimate_s() quietly substitutes a
    # per-class constant, which costs the spread and rank gates rather than
    # raising anything. w_snp_50 is the case in point -- it scored best for
    # soft_partial and is left out, because the fitter measures it against a
    # far-field SNP-density background and the pipeline has no matching
    # definition, so the two would be different quantities under one name.
    #
    # rate_ratio is a candidate here and NOT in fit_classifier: strength is
    # read off footprint geometry, which depends on the local recombination
    # rate, while class is read off haplotype structure, which does not.
    # Offering it to the classifier would let it learn "cold spot implies
    # this class" from the nine cold-spot configurations.
    keys = ["pi_ratio", "pi_min_ratio", "h12_ratio", "snp_ratio",
            "w_pi_50", "w_pi_75", "w_h12_2", "w_h12_15", "rate_ratio"]
    # Drop any predictor that is not strictly positive on every replicate.
    # These models are log-linear, so a feature of zero is not a weak
    # measurement, it is log(0): the fit silently discards that replicate and
    # the shipped estimator cannot score that sweep at all. w_h12_2 and
    # w_h12_15 are exactly this case -- contiguous_high_width() returns 0.0
    # when H12 never rises above its multiple of background around the
    # centre, which happens for 13 of 42 hard_complete replicates at the 2x
    # threshold. A model built on one of them scored 0.90 within tolerance on
    # the 29 replicates where it was defined and could not estimate s for the
    # weakest completed sweep in the shipped zoo. Measuring the candidates
    # rather than listing them keeps this honest if the width code changes.
    if "--allow-zero-width" not in sys.argv:
        dropped = [k for k in keys
                   if not all(np.isfinite(r[k]) and r[k] > 0 for r in data)]
        if dropped:
            counts = {k: sum(1 for r in data
                             if np.isfinite(r[k]) and r[k] > 0) for k in dropped}
            print("  excluded as not strictly positive on every replicate: "
                  + ", ".join(f"{k} ({counts[k]}/{len(data)})"
                              for k in dropped))
        keys = [k for k in keys if k not in set(dropped)]
    singles = [(k,) for k in keys]
    pairs = [c for c in itertools.combinations(keys, 2)]
    # Triples that include rate_ratio only: a (level, width, rate) model is
    # the shape the cold-spot geometry calls for, and the full triple space
    # would be 120 combinations per class with no reason behind most of them.
    triples = [c + ("rate_ratio",)
               for c in itertools.combinations(
                   [k for k in keys if k != "rate_ratio"], 2)]
    if "--cold-transfer" in sys.argv:
        # The question the cold spots were put there to ask, posed the way an
        # analyst would actually meet it: calibrate a width-based strength
        # estimator on BASE-RATE simulations only -- which is what you get if
        # you simulate your null at the genome's average recombination rate,
        # as both agent transcripts on the previous instance did -- and then
        # apply it to the sweeps that sit inside a wide cold spot.
        #
        # This is NOT the same question as --no-rate, and the two answers
        # differ. A rate-blind model FITTED on a mixture of base-rate and
        # cold-spot replicates absorbs the stretching into its coefficients
        # and loses almost nothing. A rate-blind model fitted at the base
        # rate alone has never seen a stretched footprint, and the error it
        # makes on one is the real cost of ignoring the map.
        base = [r for r in data if abs(r["rate_ratio"] - 1.0) < 1e-9]
        cold = [r for r in data if abs(r["rate_ratio"] - 1.0) >= 1e-9]
        print(f"\n### COLD-SPOT TRANSFER: fit on {len(base)} base-rate "
              f"replicates, test on {len(cold)} cold-spot replicates ###")
        f = GATES["s_within_factor"]
        # Two arms, and the difference between them is the point.
        #
        #   base-only   trained on base-rate replicates alone. Note that
        #               rate_ratio is identically 1.0 in every one of those
        #               rows, so log(rate_ratio) is a column of zeros and its
        #               coefficient is not identifiable: a "rate-aware" model
        #               trained here IS the rate-blind model, numerically.
        #               That is why this arm is reported with the rate term
        #               dropped -- carrying it would print the same numbers
        #               under a name that implies a correction is happening.
        #               The lesson is that the rate exponent cannot be learned
        #               without simulating a cold spot.
        #   plus-cold   the reference's own situation: trained on the
        #               base-rate replicates plus the cold-spot replicates
        #               from OTHER configurations, held out by configuration
        #               so no test sweep's own configuration is in its
        #               training set.
        for name, combo_of, use_cold in (
                ("base-rate calibration only, rate term dropped "
                 "(the shortcut)",
                 lambda c: tuple(k for k in CHOSEN_S[c] if k != "rate_ratio"),
                 False),
                ("base-rate plus other configurations' cold spots "
                 "(the reference)", lambda c: CHOSEN_S[c], True)):
            print(f"\n  {name}")
            allerr = []
            for c in CLASSES:
                combo = combo_of(c)
                def usable(rows):
                    return [r for r in rows if r["cls"] == c
                            and np.isfinite(r["s_true"]) and r["s_true"] > 0
                            and all(np.isfinite(r[k]) and r[k] > 0
                                    for k in combo)]
                te = usable(cold)
                if not te:
                    print(f"    {c:<14} (no cold-spot replicates)")
                    continue
                pred, yt = [], []
                for r in te:
                    tr = usable(base)
                    if use_cold:
                        tr = tr + [q for q in usable(cold)
                                   if q["cfg"] != r["cfg"]]
                    if len(tr) < len(combo) + 2:
                        continue
                    X = np.log(np.array([[q[k] for k in combo] for q in tr],
                                        float))
                    y = np.log(np.array([q["s_true"] for q in tr], float))
                    A = np.column_stack([np.ones(len(tr))] + list(X.T))
                    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                    xr = np.log(np.array([r[k] for k in combo], float))
                    pred.append(float(np.concatenate([[1.0], xr]) @ coef))
                    yt.append(float(np.log(r["s_true"])))
                if not pred:
                    print(f"    {c:<14} (not enough training replicates)")
                    continue
                pred = np.array(pred); yt = np.array(yt)
                # Signed, in the direction that matters: a stretched footprint
                # read without the rate looks like a stronger sweep, so the
                # bias should be positive if that is what is happening.
                bias = float(np.exp(np.mean(pred - yt)))
                err = np.exp(np.abs(pred - yt))
                allerr += list(err)
                print(f"    {c:<14} n={len(te):<3} median error "
                      f"{float(np.median(err)):.2f}x  worst "
                      f"{float(err.max()):.2f}x  mean bias {bias:.2f}x  "
                      f"within {f}: {float((err <= f).mean()):.2f}")
            if allerr:
                a = np.array(allerr)
                print(f"    {'POOLED':<14} n={len(a):<3} median error "
                      f"{float(np.median(a)):.2f}x  worst {float(a.max()):.2f}x"
                      f"{'':19}within {f}: {float((a <= f).mean()):.2f}")
        sys.exit(0)
    if "--no-rate" in sys.argv:
        # What the best strength and age models can do when they are NOT
        # allowed to see the local recombination rate. This is the shortcut
        # both agent transcripts on the previous instance took -- read
        # strength off the physical width of the footprint -- and the three
        # sweeps sitting inside wide cold spots are what it costs. Printing
        # it is the only way the claim "the rate correction is required, not
        # decorative" is a measurement rather than an assertion.
        singles = [c for c in singles if "rate_ratio" not in c]
        pairs = [c for c in pairs if "rate_ratio" not in c]
        triples = []
        print("\n### RATE-BLIND: rate_ratio withheld from every model ###")
        best_s = score_s(singles + pairs, data)
        best_t = score_t(singles + pairs, data)
        # The per-class tables above are not the comparison that matters: the
        # verifier scores a single POOLED accuracy and a single POOLED rank
        # correlation over every matched call. So emit the best rate-blind
        # model per class and pool it exactly as the rate-aware one is
        # pooled, which is the only like-for-like number.
        emit(data, chosen_s=best_s, chosen_t=best_t,
             tag=", RATE-BLIND, best available per class")
        sys.exit(0)
    if "--emit" in sys.argv:
        emit(data)
    else:
        score_s(singles + pairs + triples, data)
        score_t(singles + pairs + triples, data)
        emit(data)
