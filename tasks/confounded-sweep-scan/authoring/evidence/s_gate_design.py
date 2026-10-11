#!/usr/bin/env python3
"""Choose the three `s` gates so that they separate the reference from the
shortcut, and show the measurement that chose them.

The verifier grades selection coefficients three ways at once -- magnitude
within a factor, spread across the reported values, and rank correlation
against the truth -- and a threshold is only worth having if some real method
fails it and the reference passes it. This script measures both sides on the
composition of the SHIPPED zoo rather than on the calibration set as a whole,
which matters: the calibration set is 27 cold-spot replicates out of 120,
while the shipped zoo is 3 cold-spot sweeps out of 15, and three of fifteen
is what the gate will actually see.

Two methods are scored, and both are held out by configuration so neither
is reading its own answer:

  REFERENCE   the shipped per-class log-linear models, which carry the local
              recombination rate as a fitted predictor, trained on every
              other configuration.
  SHORTCUT    the same functional form with the rate term dropped, trained
              on BASE-RATE replicates only. This is what you get by
              calibrating against simulations run at the genome's average
              recombination rate, which is what both agent transcripts on the
              previous instance did, and it is a competent method everywhere
              except inside a cold spot.

Both are given the true class, so this isolates estimation from
classification, and both are given a perfect match set, so it isolates
estimation from detection. The shortcut is therefore being flattered in two
ways: whatever margin it keeps here, a real submission making the same
mistake has less.

  s_gate_design.py [--draws N] [--matched K]
"""
import argparse, itertools, sys
import numpy as np

import fit_estimators as F
from fit_estimators import CHOSEN_S, CHOSEN_S_RATE, CLASSES, spearman
import config as C


def shipped_sweeps():
    """Every shipped sweep with the calibration configuration that matches it.

    The three cold-spot sweeps were given their own calibration
    configurations at their own rate factors precisely so this mapping
    exists; without them the cold-spot arm of this comparison would have to
    be extrapolated.
    """
    cold = set(C.SWEEPS_IN_COLDSPOT)
    out = []
    for sid, chrom, _frac, s, _origin, _n, klass in C.SWEEP_SPEC:
        tag = f"_cs_s{round(s * 10000):04d}" if sid in cold \
            else f"_s{round(s * 10000):04d}"
        out.append({"sid": sid, "s": s, "cls": klass, "cold": sid in cold,
                    "cfg_prefix": klass + tag})
        del chrom
    return out


def resolve(sweeps, data):
    """Attach each sweep's calibration replicates, by configuration name.

    Matched by EQUALITY, not by prefix. A prefix match broke as soon as a
    configuration name became a prefix of another: hard_partial_s0075 and a
    leftover hard_partial_s0075p from an earlier revision both matched, and
    the whole script stopped rather than choosing between two sets of
    replicates with identical parameters. The names come from
    provenance/shipped_configs.py now, so equality is exactly right.
    """
    by_cfg = {}
    for r in data:
        by_cfg.setdefault(r["cfg"], []).append(r)
    missing = []
    for sw in sweeps:
        hits = [cfg for cfg in by_cfg if cfg == sw["cfg_prefix"]]
        if not hits:
            missing.append(sw["sid"] + " wants " + sw["cfg_prefix"])
            continue
        if len(hits) > 1:
            raise SystemExit(f"{sw['sid']}: {sw['cfg_prefix']} is ambiguous, "
                             f"matching {sorted(hits)}")
        sw["cfg"] = hits[0]
        sw["reps"] = by_cfg[hits[0]]
    if missing:
        raise SystemExit("no calibration configuration for:\n  "
                         + "\n  ".join(missing))
    return sweeps


def loo_predict(data, combo, cls, train_pool):
    """Held-out log-s prediction for every replicate of class `cls`.

    Returns {(cfg, index_in_cfg): predicted log s}. The training set for a
    replicate is `train_pool` minus every replicate sharing its
    configuration, so a configuration never predicts itself -- the same
    leave-one-configuration-out rule the estimator tables use, because
    leaving out one REPLICATE would let the other two replicates of the same
    configuration stand in for it.
    """
    def usable(rows):
        return [r for r in rows if r["cls"] == cls
                and np.isfinite(r["s_true"]) and r["s_true"] > 0
                and all(np.isfinite(r[k]) and r[k] > 0 for k in combo)]
    pool = usable(train_pool)
    out = {}
    for r in usable(data):
        tr = [q for q in pool if q["cfg"] != r["cfg"]]
        if len(tr) < len(combo) + 2:
            continue
        X = np.log(np.array([[q[k] for k in combo] for q in tr], float))
        y = np.log(np.array([q["s_true"] for q in tr], float))
        A = np.column_stack([np.ones(len(tr))] + list(X.T))
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        xr = np.log(np.array([r[k] for k in combo], float))
        out[r["key"]] = float(np.concatenate([[1.0], xr]) @ coef)
    return out


ARMS = ("shipped", "rate-aware", "shortcut")


def build_predictions(data):
    """Each arm's held-out predictions, keyed by replicate.

      shipped     CHOSEN_S, trained on every other configuration.
      rate-aware  CHOSEN_S_RATE, the best model per class that carries the
                  local recombination rate, trained the same way. This is
                  the arm that decides whether the rate term belongs in the
                  shipped estimator: it wins on the three cold-spot sweeps
                  and loses a little on the other twelve, and the shipped
                  composition is what settles the trade.
      shortcut    CHOSEN_S with the rate term dropped and trained on
                  BASE-RATE replicates only, which is what calibrating
                  against simulations at the genome's average rate gives
                  you.
    """
    base = [r for r in data if abs(r["rate_ratio"] - 1.0) < 1e-9]
    out = {a: {} for a in ARMS}
    for c in CLASSES:
        out["shipped"].update(loo_predict(data, CHOSEN_S[c], c, data))
        out["rate-aware"].update(loo_predict(data, CHOSEN_S_RATE[c], c, data))
        blind = tuple(k for k in CHOSEN_S_RATE[c] if k != "rate_ratio")
        out["shortcut"].update(loo_predict(data, blind, c, base))
    return out


def score_draw(pairs, factor):
    """(accuracy, spread, rho) for one drawn submission.

    `pairs` is [(s_hat, s_true), ...] over the matched calls.
    """
    hat = np.array([p[0] for p in pairs], float)
    true = np.array([p[1] for p in pairs], float)
    err = np.maximum(hat / true, true / hat)
    acc = float((err <= factor).mean())
    spread = float(hat.max() / hat.min())
    return acc, spread, spearman(list(hat), list(true))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--matched", type=int, default=0,
                    help="grade only this many of the fifteen sweeps, drawn "
                         "at random, to mimic a submission that matched only "
                         "the recall floor; 0 grades all fifteen")
    args = ap.parse_args()

    data = F.rows(F.ALL_DIRS)
    for i, r in enumerate(data):
        r["key"] = (r["cfg"], i)
    preds = build_predictions(data)
    sweeps = resolve(shipped_sweeps(), data)

    print(f"calibration set: {len(data)} replicates; shipped zoo: "
          f"{len(sweeps)} sweeps, {sum(s['cold'] for s in sweeps)} of them "
          f"inside a cold spot")
    print("\nthe shipped zoo mapped onto calibration configurations")
    print(f"  {'id':<6}{'class':<15}{'s':>8}  {'cold':<5} configuration")
    for sw in sweeps:
        print(f"  {sw['sid']:<6}{sw['cls']:<15}{sw['s']:>8.4f}  "
              f"{'yes' if sw['cold'] else '-':<5} {sw['cfg']}")

    # Per-sweep pools of (predicted, true) under each arm. A replicate whose
    # realized class differs from the scheduled one is kept under the class
    # it realized, because that is the class the verifier will grade it as.
    pools = {}
    for name, pred in preds.items():
        pool = []
        for sw in sweeps:
            got = [(float(np.exp(pred[r["key"]])), float(r["s_true"]))
                   for r in sw["reps"] if r["key"] in pred]
            if not got:
                raise SystemExit(f"{name}: no held-out prediction for "
                                 f"{sw['sid']} ({sw['cfg']})")
            pool.append(got)
        pools[name] = pool

    rng = np.random.default_rng(20_251_010)
    k = args.matched or len(sweeps)
    print(f"\nscoring {args.draws} drawn submissions, each taking one "
          f"calibration replicate per sweep" + (f", {k} of {len(sweeps)} "
          f"sweeps matched" if args.matched else ", all fifteen matched"))

    grid = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5]
    draws = {}
    for name, pool in pools.items():
        rows = []
        for _ in range(args.draws):
            idx = rng.permutation(len(pool))[:k] if args.matched \
                else np.arange(len(pool))
            pairs = [pool[i][rng.integers(len(pool[i]))] for i in idx]
            rows.append(pairs)
        draws[name] = rows

    print("\n=== magnitude: fraction of drawn submissions reaching each "
          "accuracy, by tolerance ===")
    accs = [0.70, 0.75, 0.80, 0.85, 0.90]
    hdr = "  " + "factor".ljust(8) + "method".ljust(12) \
        + "".join(f"  >={a:.2f}" for a in accs) + "   median acc"
    print(hdr)
    for f in grid:
        for name in ARMS:
            got = np.array([score_draw(p, f)[0] for p in draws[name]])
            cells = "".join(f"  {float((got >= a).mean()):6.2f}" for a in accs)
            print(f"  {f:<8.2f}{name:<12}{cells}   {float(np.median(got)):.3f}")
        print()

    print("=== spread and rank correlation (tolerance-independent) ===")
    for name in ARMS:
        sp = np.array([score_draw(p, 2.0)[1] for p in draws[name]])
        rh = np.array([score_draw(p, 2.0)[2] for p in draws[name]])
        print(f"  {name:<12} spread: median {float(np.median(sp)):.2f}x, "
              f"1st percentile {float(np.percentile(sp, 1)):.2f}x;  "
              f"rho: median {float(np.median(rh)):+.3f}, "
              f"1st percentile {float(np.percentile(rh, 1)):+.3f}")

    print("\n=== all three s gates at once: fraction of drawn submissions "
          "that pass every one ===")
    print("  A submission passes only if magnitude, spread and rank all hold, "
          "so this is\n  the number that decides whether a gate setting is "
          "passable at all. A setting\n  the reference clears in 85% of draws "
          "is not a hard gate, it is a gate that\n  fails one submission in "
          "seven for the draw it got.")
    cand = [(2.00, 0.70, 3.00, 0.50), (2.20, 0.70, 3.00, 0.50),
            (2.20, 0.75, 3.00, 0.50), (2.20, 0.75, 3.00, 0.40),
            (2.50, 0.75, 3.00, 0.50), (2.50, 0.80, 3.00, 0.50),
            (2.50, 0.80, 3.00, 0.40), (2.50, 0.85, 3.00, 0.40),
            (2.80, 0.85, 3.00, 0.40), (2.80, 0.90, 3.00, 0.40)]
    print(f"\n  {'factor':>7}{'min acc':>9}{'min spread':>12}{'min rho':>9}"
          + "".join(f"{a:>13}" for a in ARMS))
    for f, a_min, sp_min, rho_min in cand:
        cells = []
        for name in ARMS:
            ok = 0
            for pairs in draws[name]:
                acc, sp, rho = score_draw(pairs, f)
                if acc >= a_min and sp >= sp_min and rho >= rho_min:
                    ok += 1
            cells.append(ok / len(draws[name]))
        print(f"  {f:>7.2f}{a_min:>9.2f}{sp_min:>12.2f}{rho_min:>9.2f}"
              + "".join(f"{c:>13.3f}" for c in cells))

    print("\n=== cold-spot sweeps alone, which is where the two arms differ ===")
    cold_idx = [i for i, sw in enumerate(sweeps) if sw["cold"]]
    for name, pool in pools.items():
        for i in cold_idx:
            errs = [max(h / t, t / h) for h, t in pool[i]]
            bias = float(np.exp(np.mean([np.log(h / t) for h, t in pool[i]])))
            print(f"  {name:<12}{sweeps[i]['sid']:<6}"
                  f"{sweeps[i]['cls']:<15} error "
                  + "/".join(f"{e:.2f}x" for e in sorted(errs))
                  + f"  mean bias {bias:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
