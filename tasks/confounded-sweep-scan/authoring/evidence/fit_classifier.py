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
from fit_estimators import rows  # noqa: F401

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
                      quiet=False, detail=None):
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
    if detail is not None:
        detail.extend(zip(rs, pred, true))
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
    # Chosen by a joint search over the top six feature sets per axis,
    # ranked by SHIPPED-MIX four-way accuracy and held out by configuration
    # over all 40 calibration configurations, 16 of which the shipped
    # replicate does not use.
    #
    # Two things are deliberate about that objective. Scoring each axis on
    # its own is wrong because the softness rules are fitted inside a
    # completeness group and applied to the PREDICTED group, so the axes
    # interact: switching the completeness rule to the pair that scores best
    # on its own axis, 0.961 against 0.941, dropped four-way accuracy from
    # 0.824 to 0.745 on an earlier zoo. And scoring over the calibration set
    # is wrong because the gate is a fraction of the matched calls on one
    # replicate whose class mix is 5/3/4/3, while the calibration set is
    # 42/22/31/25 -- it over-weights hard_complete, which is the easiest
    # class by a wide margin. Measured, the triple that wins on the
    # calibration average scores 0.758 there and 0.667 on the shipped mix;
    # this one scores 0.758 and 0.756. Ranking on the wrong objective was
    # worth nine points of the number the verifier actually computes.
    #
    # completeness: the floor of the diversity trough, the extreme-nSL
    #   fraction and the extent of the H12 elevation. Only a sweep still
    #   rising carries a long young high-frequency haplotype, which is what
    #   nSL responds to.
    "complete": ("pi_min_ratio", "nsl_delta", "w_h12_15"),
    # softness among COMPLETED sweeps: the trough floor, the SNP-density
    #   deficit and H2/H1. One frequent haplotype drives diversity to
    #   nothing and takes the segregating sites with it; several leave both
    #   milder and H2/H1 high. Tajima's D was here instead of snp_ratio and
    #   is the weaker of the two on the shipped mix, which is the whole
    #   nine-point gap above -- this is the axis the shipped zoo loads most.
    "soft_given_complete": ("pi_min_ratio", "snp_ratio", "h2h1_ratio"),
    # softness among PARTIAL sweeps: the extent of the H12 elevation with
    #   H2/H1 -- the textbook discriminator, usable here only because age is
    #   not in play, since every partial sweep is still rising. The extent
    #   is read at 2x background rather than 1.5x.
    "soft_given_partial": ("w_h12_2", "h2h1_ratio"),
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
    print("  Ranked by SHIPPED-MIX accuracy, not by accuracy over the\n"
          "  calibration set. The verifier applies the gate to a fraction of\n"
          "  the matched calls on one replicate carrying five hard_complete,\n"
          "  three soft_complete, four hard_partial and three soft_partial\n"
          "  sweeps, and those classes are not equally easy -- the\n"
          "  calibration set has 42 hard_complete replicates against 22\n"
          "  soft_complete, so it flatters a classifier on exactly the axis\n"
          "  the shipped zoo weights most. Measured, the gap is 0.758 over\n"
          "  the calibration set against 0.667 over the shipped mix, which is\n"
          "  the difference between a class gate of 0.70 passing and failing.")
    weights = shipped_mix_weights(data)
    best = []
    for cc in combos_c[:top]:
        for sc in combos_sc[:top]:
            for sp in combos_sp[:top]:
                detail = []
                acc = joint_conditional(data, cc, sc, sp, quiet=True,
                                        detail=detail)
                best.append((shipped_mix_accuracy(detail, weights), acc,
                             cc, sc, sp))
    best.sort(key=lambda t: -t[0])
    print(f"\n  {'shipped':>8}{'calib':>8}  feature sets")
    for mix, acc, cc, sc, sp in best[:10]:
        print(f"  {mix:>8.3f}{acc:>8.3f}  complete={'+'.join(cc)}  "
              f"soft|complete={'+'.join(sc)}  soft|partial={'+'.join(sp)}")
    return best[0][2:] if best else None


def shipped_mix_weights(data):
    """One weight per calibration configuration, from the shipped zoo.

    Each shipped sweep contributes 1/15 of the gate, spread evenly over the
    replicates of the configuration that matches it. A configuration the
    shipped replicate does not use gets no weight at all.
    """
    import s_gate_design as SG
    sweeps = SG.resolve(SG.shipped_sweeps(), data)
    w = {}
    for sw in sweeps:
        w[sw["cfg"]] = w.get(sw["cfg"], 0.0) + 1.0 / len(sweeps)
    return w


def shipped_mix_accuracy(detail, weights):
    """Expected class accuracy on the shipped zoo, from per-replicate hits."""
    num = den = 0.0
    hits, tot = {}, {}
    for r, pred, true in detail:
        cfg = r["cfg"]
        hits[cfg] = hits.get(cfg, 0) + (1 if pred == true else 0)
        tot[cfg] = tot.get(cfg, 0) + 1
    for cfg, wt in weights.items():
        if tot.get(cfg):
            num += wt * hits[cfg] / tot[cfg]
            den += wt
    return num / den if den else float("nan")


ALL_DIRS = ("training", "training_holdout")

if __name__ == "__main__":
    data = rows(ALL_DIRS if "--shipped-only" not in sys.argv else "training")
    if "--holdout-only" in sys.argv:
        data = [r for r in data if not r["shipped_cfg"]]
    print(f"calibration set: {len(data)} replicates, "
          f"{len(set(r['cfg'] for r in data))} configurations, "
          f"{len(set(r['cfg'] for r in data if not r['shipped_cfg']))} of them "
          f"not used by the shipped replicate")
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

    if "--shipped-composition" in sys.argv:
        # Four-way accuracy over the calibration set is not the number the
        # verifier applies. The gate is a fraction of the MATCHED calls on
        # the shipped replicate, which is fifteen sweeps in a fixed mix of
        # the four classes -- five hard_complete, three soft_complete, four
        # hard_partial, three soft_partial -- and the classes are not equally
        # easy. Averaging over a calibration set with a different mix can
        # put a gate either side of what the reference will actually score.
        # This draws submissions from the shipped mix and reports how often
        # each candidate threshold is met, which is what "passable" means.
        import s_gate_design as SG
        detail = []
        joint_conditional(data, CHOSEN["complete"],
                          CHOSEN["soft_given_complete"],
                          CHOSEN["soft_given_partial"], quiet=True,
                          detail=detail)
        by_cfg = {}
        for r, pred, true in detail:
            by_cfg.setdefault(r["cfg"], []).append(pred == true)
        sweeps = SG.resolve(SG.shipped_sweeps(), data)
        pool, missing = [], []
        for sw in sweeps:
            got = by_cfg.get(sw["cfg"])
            if not got:
                missing.append(f"{sw['sid']} ({sw['cfg']})")
                continue
            pool.append(got)
        if missing:
            raise SystemExit("no classifier prediction for: "
                             + ", ".join(missing))
        print(f"\n=== class gate over the shipped mix of {len(pool)} sweeps "
              f"===")
        for sw, got in zip(sweeps, pool):
            print(f"  {sw['sid']:<6}{sw['cls']:<15}"
                  f"{'correct' if all(got) else ('wrong' if not any(got) else 'mixed'):<9}"
                  f"{sum(got)}/{len(got)} replicates classified correctly")
        rng = np.random.default_rng(20_251_010)
        draws = np.array([
            [pool[i][rng.integers(len(pool[i]))] for i in range(len(pool))]
            for _ in range(4000)])
        accs = draws.mean(axis=1)
        print(f"\n  accuracy over draws: median {float(np.median(accs)):.3f}, "
              f"1st percentile {float(np.percentile(accs, 1)):.3f}, "
              f"mean {float(accs.mean()):.3f}")
        print(f"  {'threshold':>10}{'passes':>9}")
        for thr in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            print(f"  {thr:>10.2f}{float((accs >= thr).mean()):>9.3f}")
        # And the same when only the recall floor is matched: the gate is a
        # FRACTION, so a smaller match set makes each error cost more.
        for k in (10, 12):
            sub = np.array([
                [pool[i][rng.integers(len(pool[i]))]
                 for i in rng.permutation(len(pool))[:k]]
                for _ in range(4000)]).mean(axis=1)
            print(f"\n  with only {k} of {len(pool)} matched: median "
                  f"{float(np.median(sub)):.3f}, "
                  f"1st percentile {float(np.percentile(sub, 1)):.3f}")
            for thr in (0.60, 0.65, 0.70, 0.75):
                print(f"  {thr:>10.2f}{float((sub >= thr).mean()):>9.3f}")
        sys.exit(0)

    print("\n--- held out with the CHOSEN feature sets, not the best-scoring ---")
    joint_conditional(data, CHOSEN["complete"],
                      CHOSEN["soft_given_complete"],
                      CHOSEN["soft_given_partial"])
    emit(data)
