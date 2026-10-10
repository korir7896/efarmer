#!/usr/bin/env python3
"""Classify the two axes separately, and measure whether that beats 4-way.

A nearest-centroid rule over the four classes at once scores 72.9%
leave-one-out on the calibration replicates, against a gate of 70%: too thin
to rely on. The class scheme is two independent binary properties, so the
natural alternative is two binary rules -- complete versus partial, and hard
versus soft -- each free to use the features that carry its own axis.

Scored leave-one-CONFIGURATION-out, so a rule cannot earn its accuracy by
memorising the handful of (s, age) pairs the replicates were simulated at.
"""
import itertools, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fit_estimators import rows

FEATS = ["pi_ratio", "pi_min_ratio", "tajd_delta", "h12_ratio", "nsl_delta",
         "snp_ratio", "w_pi_50", "w_pi_75", "w_h12_2", "w_h12_15", "h2h1_ratio"]


def logistic_fit(X, y, iters=400, lr=0.3, l2=1e-3):
    """Plain gradient-descent logistic regression; no sklearn in this env."""
    n, d = X.shape
    A = np.column_stack([np.ones(n), X])
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-A @ w))
        g = A.T @ (p - y) / n + l2 * np.r_[0.0, w[1:]]
        w -= lr * g * 10.0
    return w


def predict(w, X):
    A = np.column_stack([np.ones(len(X)), X])
    return 1.0 / (1.0 + np.exp(-A @ w))


def held_out(X, y, groups):
    out = np.full(len(y), np.nan)
    for g in sorted(set(groups)):
        held = np.array(groups) == g
        if (~held).sum() < X.shape[1] + 3:
            continue
        if len(set(y[~held])) < 2:
            continue
        w = logistic_fit(X[~held], y[~held])
        out[held] = predict(w, X[held])
    return out


def standardize(X):
    mu = X.mean(0); sd = X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd, mu, sd


def axis_all(data, label_fn, combos, subset_fn=None):
    """Held-out accuracy of every candidate on one axis, unsorted."""
    rs = [r for r in data if all(np.isfinite(r[k]) for k in FEATS)
          and (subset_fn is None or subset_fn(r))]
    y = np.array([1.0 if label_fn(r) else 0.0 for r in rs])
    groups = [r["cfg"] for r in rs]
    out = []
    for combo in combos:
        X = np.array([[r[k] for k in combo] for r in rs], float)
        Xs, _, _ = standardize(X)
        p = held_out(Xs, y, groups)
        ok = np.isfinite(p)
        if ok.sum() < len(rs) * 0.8:
            continue
        out.append((float(((p[ok] >= 0.5) == (y[ok] >= 0.5)).mean()),
                    combo, int(ok.sum())))
    return out


def axis_table(data, name, label_fn, combos):
    rs = [r for r in data if all(np.isfinite(r[k]) for k in FEATS)]
    y = np.array([1.0 if label_fn(r) else 0.0 for r in rs])
    groups = [r["cfg"] for r in rs]
    print(f"\n=== {name}  (n={len(rs)}, positives {int(y.sum())}) ===")
    out = []
    for combo in combos:
        X = np.array([[r[k] for k in combo] for r in rs], float)
        X = np.where(np.isfinite(X), X, 0.0)
        Xs, _, _ = standardize(X)
        p = held_out(Xs, y, groups)
        ok = np.isfinite(p)
        if ok.sum() < len(rs) * 0.8:
            continue
        acc = float(((p[ok] >= 0.5) == (y[ok] >= 0.5)).mean())
        out.append((acc, combo, int(ok.sum())))
    out.sort(key=lambda t: -t[0])
    for acc, combo, n in out[:8]:
        print(f"  {'+'.join(combo):<42} {acc:.3f}  (n={n})")
    return out[0] if out else None


def joint(data, combo_c, combo_h):
    """Four-way accuracy of the two binary rules together, held out."""
    rs = [r for r in data if all(np.isfinite(r[k]) for k in FEATS)]
    groups = [r["cfg"] for r in rs]
    res = {}
    for name, combo, label_fn in (
            ("complete", combo_c, lambda r: r["cls"].endswith("complete")),
            ("soft", combo_h, lambda r: r["cls"].startswith("soft"))):
        X = np.array([[r[k] for k in combo] for r in rs], float)
        Xs, _, _ = standardize(X)
        y = np.array([1.0 if label_fn(r) else 0.0 for r in rs])
        res[name] = (held_out(Xs, y, groups) >= 0.5)
    pred = ["%s_%s" % ("soft" if s else "hard", "complete" if c else "partial")
            for s, c in zip(res["soft"], res["complete"])]
    true = [r["cls"] for r in rs]
    acc = float(np.mean([p == t for p, t in zip(pred, true)]))
    print(f"\n=== four-way accuracy of the two rules together: {acc:.3f} "
          f"over {len(rs)} replicates ===")
    classes = sorted(set(true))
    print(f"  {'true \\\\ predicted':<18}" + "".join(f"{c:>15}" for c in classes))
    for t in classes:
        row = [sum(1 for p, tt in zip(pred, true) if tt == t and p == c)
               for c in classes]
        print(f"  {t:<18}" + "".join(f"{v:>15}" for v in row))
    return acc


def axis_table_within(data, name, subset_fn, combos):
    """The soft/hard axis fitted separately inside one completeness group."""
    rs = [r for r in data if all(np.isfinite(r[k]) for k in FEATS)
          and subset_fn(r)]
    y = np.array([1.0 if r["cls"].startswith("soft") else 0.0 for r in rs])
    groups = [r["cfg"] for r in rs]
    print(f"\n=== SOFT vs HARD, {name}  (n={len(rs)}, "
          f"positives {int(y.sum())}) ===")
    out = []
    for combo in combos:
        X = np.array([[r[k] for k in combo] for r in rs], float)
        Xs, _, _ = standardize(X)
        p = held_out(Xs, y, groups)
        ok = np.isfinite(p)
        if ok.sum() < len(rs) * 0.8:
            continue
        acc = float(((p[ok] >= 0.5) == (y[ok] >= 0.5)).mean())
        out.append((acc, combo, int(ok.sum())))
    out.sort(key=lambda t: -t[0])
    for acc, combo, n in out[:6]:
        print(f"  {'+'.join(combo):<42} {acc:.3f}  (n={n})")
    return out[0] if out else None


def joint_conditional(data, combo_c, combo_soft_complete, combo_soft_partial,
                      quiet=False):
    """Completeness first, then softness with a rule fitted for that group."""
    rs = [r for r in data if all(np.isfinite(r[k]) for k in FEATS)]
    groups = [r["cfg"] for r in rs]
    X = np.array([[r[k] for k in combo_c] for r in rs], float)
    Xs, _, _ = standardize(X)
    yc = np.array([1.0 if r["cls"].endswith("complete") else 0.0 for r in rs])
    pred_c = held_out(Xs, yc, groups) >= 0.5

    pred_s = np.zeros(len(rs), bool)
    for want_complete, combo in ((True, combo_soft_complete),
                                 (False, combo_soft_partial)):
        # Fit on the TRUE group, predict for the PREDICTED group: that is how
        # the pipeline runs, and it keeps the error of the first stage in the
        # measurement instead of hiding it.
        tr = [i for i, r in enumerate(rs)
              if r["cls"].endswith("complete") == want_complete]
        te = [i for i in range(len(rs)) if pred_c[i] == want_complete]
        if not tr or not te:
            continue
        Xtr = np.array([[rs[i][k] for k in combo] for i in tr], float)
        mu, sd = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1.0)
        ytr = np.array([1.0 if rs[i]["cls"].startswith("soft") else 0.0
                        for i in tr])
        gtr = [rs[i]["cfg"] for i in tr]
        # held-out within the training group, then applied to the rest
        for g in sorted(set(gtr)):
            keep = [j for j, i in enumerate(tr) if rs[i]["cfg"] != g]
            if len(set(ytr[keep])) < 2:
                continue
            w = logistic_fit((Xtr[keep] - mu) / sd, ytr[keep])
            for j, i in enumerate(tr):
                if rs[i]["cfg"] == g and pred_c[i] == want_complete:
                    pred_s[i] = predict(w, (Xtr[j:j + 1] - mu) / sd)[0] >= 0.5
        # replicates whose completeness was MISpredicted into this group
        wrong = [i for i in te if (rs[i]["cls"].endswith("complete")
                                   != want_complete)]
        if wrong:
            if len(set(ytr)) > 1:
                w = logistic_fit((Xtr - mu) / sd, ytr)
                Xw = np.array([[rs[i][k] for k in combo] for i in wrong], float)
                pw = predict(w, (Xw - mu) / sd) >= 0.5
                for i, v in zip(wrong, pw):
                    pred_s[i] = v

    pred = ["%s_%s" % ("soft" if s_ else "hard",
                       "complete" if c else "partial")
            for s_, c in zip(pred_s, pred_c)]
    true = [r["cls"] for r in rs]
    acc = float(np.mean([p == t for p, t in zip(pred, true)]))
    if quiet:
        return acc
    print(f"\n=== four-way accuracy, completeness then softness: {acc:.3f} "
          f"over {len(rs)} replicates ===")
    classes = sorted(set(true))
    print(f"  {'true / predicted':<18}" + "".join(f"{c:>15}" for c in classes))
    for t in classes:
        row = [sum(1 for p, tt in zip(pred, true) if tt == t and p == c)
               for c in classes]
        print(f"  {t:<18}" + "".join(f"{v:>15}" for v in row))
    return acc


CHOSEN = {
    # Chosen by a joint search over the top six feature sets per axis, 216
    # triples, scored FOUR-WAY and held out by configuration. Scoring each
    # axis on its own is the wrong objective: the softness rules are fitted
    # inside a completeness group and applied to the PREDICTED group, so the
    # axes interact and a rule that is better on its own axis can feed the
    # next stage replicates it handles worse. Diversity with the SNP count
    # scores 0.941 on completeness against 0.961 for the extreme-nSL
    # fraction with the SNP count, and the triples built on them come out at
    # 0.745 and 0.745 four-way, where this one reaches 0.824. Nothing in the
    # search beats 0.824.
    #
    # completeness: Tajima's D with the number of segregating sites in the
    # window. A completed sweep removed variation and left the spectrum
    # skewed toward rare variants; one still rising did neither as strongly.
    "complete": ("tajd_delta", "snp_ratio"),
    # softness among COMPLETED sweeps: how deep the diversity trough got,
    # with the widths at two recovery levels. One frequent haplotype drives
    # diversity to nothing in a narrow core; several leave it shallow and
    # broad, so the shape of the trough carries the axis where its depth
    # alone does not.
    "soft_given_complete": ("pi_min_ratio", "w_pi_50", "w_pi_75"),
    # softness among PARTIAL sweeps: H2/H1, the textbook discriminator,
    # which works here only because age is not in play -- every partial
    # sweep is still rising. Pooled over completed sweeps of all ages it is
    # confounded, which is why it does not appear above.
    "soft_given_partial": ("h2h1_ratio",),
}


def emit(data):
    rs_all = [r for r in data if all(np.isfinite(r[k]) for k in FEATS)]

    def fit_rule(rs, combo, label_fn):
        X = np.array([[r[k] for k in combo] for r in rs], float)
        mu, sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1.0)
        y = np.array([1.0 if label_fn(r) else 0.0 for r in rs])
        w = logistic_fit((X - mu) / sd, y)
        acc = float(((predict(w, (X - mu) / sd) >= 0.5) == (y >= 0.5)).mean())
        return mu, sd, w, acc, len(rs)

    print("\n=== CHOSEN RULES (paste into solution/pipeline/characterize.py) ===")
    print("CLASS_RULES = {")
    for name, (rs, label_fn) in {
            "complete": (rs_all, lambda r: r["cls"].endswith("complete")),
            "soft_given_complete": ([r for r in rs_all
                                     if r["cls"].endswith("complete")],
                                    lambda r: r["cls"].startswith("soft")),
            "soft_given_partial": ([r for r in rs_all
                                    if not r["cls"].endswith("complete")],
                                   lambda r: r["cls"].startswith("soft")),
    }.items():
        combo = CHOSEN[name]
        mu, sd, w, acc, n = fit_rule(rs, combo, label_fn)
        print(f'    "{name}": dict(')
        print(f'        feats={list(combo)!r},')
        print(f'        mean=[{", ".join(f"{v:.6g}" for v in mu)}],')
        print(f'        sd=[{", ".join(f"{v:.6g}" for v in sd)}],')
        print(f'        weights=[{", ".join(f"{v:.6g}" for v in w)}]),'
              f"   # n={n}, in-sample {acc:.3f}")
    print("}")


def search_joint(data, combos_c, combos_sc, combos_sp, top=6):
    """Pick the triple of feature sets by the metric the verifier uses.

    Choosing each axis on its own accuracy is the wrong objective: the
    softness rules are fitted inside a completeness group and applied to the
    PREDICTED group, so the two interact. Switching the completeness rule to
    the pair that scores best on its own axis, 0.961 against 0.941, dropped
    four-way accuracy from 0.824 to 0.745. So the search is over the triple,
    scored four-way and held out by configuration.
    """
    print("\n=== joint search over the top feature sets per axis ===")
    best = []
    for cc in combos_c[:top]:
        for sc in combos_sc[:top]:
            for sp in combos_sp[:top]:
                acc = joint_conditional(data, cc, sc, sp, quiet=True)
                best.append((acc, cc, sc, sp))
    best.sort(key=lambda t: -t[0])
    for acc, cc, sc, sp in best[:10]:
        print(f"  {acc:.3f}  complete={'+'.join(cc)}  "
              f"soft|complete={'+'.join(sc)}  soft|partial={'+'.join(sp)}")
    return best[0] if best else None


if __name__ == "__main__":
    data = rows()
    singles = [(k,) for k in FEATS]
    pairs = list(itertools.combinations(FEATS, 2))
    triples = list(itertools.combinations(FEATS, 3))
    best_c = axis_table(data, "COMPLETE vs PARTIAL",
                        lambda r: r["cls"].endswith("complete"),
                        singles + pairs + triples)
    best_h = axis_table(data, "SOFT vs HARD",
                        lambda r: r["cls"].startswith("soft"),
                        singles + pairs + triples)
    if best_c and best_h:
        joint(data, best_c[1], best_h[1])
    bsc = axis_table_within(data, "within the COMPLETE sweeps",
                            lambda r: r["cls"].endswith("complete"),
                            singles + pairs + triples)
    bsp = axis_table_within(data, "within the PARTIAL sweeps",
                            lambda r: not r["cls"].endswith("complete"),
                            singles + pairs)
    if best_c and bsc and bsp:
        joint_conditional(data, best_c[1], bsc[1], bsp[1])
    if "--search" in sys.argv:
        cands_c = [c for _, c, _ in sorted(
            axis_all(data, lambda r: r["cls"].endswith("complete"),
                     singles + pairs + triples), key=lambda t: -t[0])]
        cands_sc = [c for _, c, _ in sorted(
            axis_all(data, lambda r: r["cls"].startswith("soft"),
                     singles + pairs + triples,
                     subset_fn=lambda r: r["cls"].endswith("complete")),
            key=lambda t: -t[0])]
        cands_sp = [c for _, c, _ in sorted(
            axis_all(data, lambda r: r["cls"].startswith("soft"),
                     singles + pairs,
                     subset_fn=lambda r: not r["cls"].endswith("complete")),
            key=lambda t: -t[0])]
        search_joint(data, cands_c, cands_sc, cands_sp)
        sys.exit(0)

    print("\n--- held out with the CHOSEN feature sets, not the best-scoring ---")
    joint_conditional(data, CHOSEN["complete"],
                      CHOSEN["soft_given_complete"],
                      CHOSEN["soft_given_partial"])
    emit(data)
