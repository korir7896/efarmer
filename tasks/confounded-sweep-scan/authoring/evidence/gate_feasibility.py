#!/usr/bin/env python3
"""Two questions about every gate on an estimated quantity.

  FEASIBLE: on independent single-sweep replicates, how often does the
  reference's own estimator land inside the gate? That fraction is what the
  pass threshold has to sit below.

  NOT TRIVIAL: on the shipped truth, how well can an answer that estimates
  nothing do? That is the best single constant, and the best one constant per
  class. Those counts have to sit below the threshold.

The calibrated constants are imported from the reference solution rather than
copied, so this cannot silently drift away from what the reference actually
uses.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "provenance"))
sys.path.insert(0, os.path.join(TASK, "solution"))
sys.path.insert(0, os.path.join(TASK, "tests"))

import config as C
from grading import GATES, TIMED_CLASSES, spearman, time_tolerance
from pipeline.characterize import S_MODEL, T_MODEL, _loglinear

CLASSES = ["hard_complete", "hard_partial", "soft_complete", "soft_partial"]


def predictors():
    """The features the fitted estimators take, per training replicate.

    Measured with the same functions the reference uses, imported from it, so
    this cannot drift away from what the pipeline actually computes.
    """
    sys.path.insert(0, HERE)
    from fit_estimators import rows
    return rows()


def feasibility():
    data = predictors()
    print("=== FEASIBLE: the reference's estimators on independent replicates ===")
    print(f"selection coefficient, within a factor of {GATES['s_within_factor']} "
          f"of truth (gate: {GATES['min_s_accuracy']:.0%} of matched calls):")
    print(f"  {'class':<16}{'pass':>10}{'rate':>8}{'median est/true':>18}")
    all_hat, all_true = [], []
    for c in CLASSES:
        rs = [r for r in data if r["cls"] == c]
        ok = n = 0
        ratios = []
        for r in rs:
            hat = _loglinear(S_MODEL[c], r)
            if hat is None:
                continue
            n += 1
            ratios.append(hat / r["s_true"])
            all_hat.append(hat); all_true.append(r["s_true"])
            if (r["s_true"] / GATES["s_within_factor"] <= hat
                    <= r["s_true"] * GATES["s_within_factor"]):
                ok += 1
        if n:
            print(f"  {c:<16}{ok}/{n:<8}{ok/n:>7.0%}{np.median(ratios):>18.2f}")
    if all_true:
        err = np.maximum(np.array(all_hat) / np.array(all_true),
                         np.array(all_true) / np.array(all_hat))
        n = len(err)
        print(f"  {'ALL':<16}{int((err <= GATES['s_within_factor']).sum())}/{n:<8}"
              f"{(err <= GATES['s_within_factor']).mean():>7.0%}")
        print(f"  rank correlation with truth "
              f"{spearman(all_hat, all_true):.3f} over {n} replicates "
              f"(not gated; reported because an estimator that ranks well "
              f"is doing more than landing in the right decade)")
        print("  NOTE: fitted on these same replicates. The honest, "
              "leave-one-configuration-out\n"
              "        numbers are in authoring/evidence/fit_estimators.py.")

    print(f"\ncompletion time, within max({GATES['time_abs_tol_gens']} gens, "
          f"{GATES['time_rel_tol']:.0%}) of truth "
          f"(gate: {GATES['min_time_correct']} of the matched hard_complete "
          f"sweeps):")
    rs = [r for r in data if r["cls"] in TIMED_CLASSES and r["completed"]
          and np.isfinite(r["t_true"])]
    ok = n = 0
    ratios = []
    for r in rs:
        hat = _loglinear(T_MODEL, r)
        if hat is None:
            continue
        n += 1
        ratios.append(hat / r["t_true"])
        if abs(hat - r["t_true"]) <= time_tolerance(r["t_true"]):
            ok += 1
    if n:
        print(f"  hard_complete   {ok}/{n:<8}{ok/n:>7.0%}"
              f"{np.median(ratios):>18.2f}")


def triviality():
    tp = os.path.join(TASK, "tests", "truth.json")
    if not os.path.exists(tp):
        print("\n(no shipped truth yet; skipping the triviality check)")
        return
    truth = json.load(open(tp))
    print("\n=== NOT TRIVIAL: the best an answer that estimates nothing can do ===")

    f = GATES["s_within_factor"]
    need = GATES["min_s_accuracy"]
    tru_s = [float(s["selection_coefficient"]) for s in truth["sweeps"]]
    print(f"s, true values span {max(tru_s)/min(tru_s):.2f}x: "
          f"{sorted(round(x, 4) for x in tru_s)}")

    # The worst case is not "all ten sweeps": it is whichever match set the
    # recall gates permit that a constant covers best. Enumerate them.
    import itertools
    by_cls = {}
    for sw in truth["sweeps"]:
        by_cls.setdefault(sw["class"], []).append(float(sw["selection_coefficient"]))
    need_per = GATES["min_recall_per_class"]
    choices = []
    for cls, vals in sorted(by_cls.items()):
        k = need_per.get(cls, 1)
        subsets = []
        for size in range(k, len(vals) + 1):
            subsets += [list(c) for c in itertools.combinations(vals, size)]
        choices.append(subsets)

    def best_constant(vals):
        lo, hi = min(vals) / f, max(vals) * f
        best = 0
        for i in range(2001):
            c = lo * (hi / lo) ** (i / 2000)
            best = max(best, sum(1 for t in vals if t / f <= c <= t * f))
        return best

    worst_rate, worst_set = 0.0, None
    for combo in itertools.product(*choices):
        vals = [v for part in combo for v in part]
        rate = best_constant(vals) / len(vals)
        if rate > worst_rate:
            worst_rate, worst_set = rate, sorted(vals)
    n_sets = 1
    for c in choices:
        n_sets *= len(c)
    print(f"s, single constant, worst case over all {n_sets} match sets the "
          f"recall gates permit, by factor:")
    for ff in (1.4, 1.5, 1.6, 1.8, 2.0, 2.5):
        w = 0.0
        for combo in itertools.product(*choices):
            vals = [v for part in combo for v in part]
            lo, hi = min(vals) / ff, max(vals) * ff
            b = 0
            for i in range(1201):
                c = lo * (hi / lo) ** (i / 1200)
                b = max(b, sum(1 for t in vals if t / ff <= c <= t * ff))
            w = max(w, b / len(vals))
        mark = "  <- gate" if abs(ff - f) < 1e-9 else ""
        print(f"    factor {ff}: best constant reaches {w:.3f}{mark}")
    print(f"    so the magnitude gate alone does NOT exclude a constant at "
          f"factor {f}; the spread and ordering gates do.")
    print(f"s, single constant: spread 1.00 against a gate of "
          f"{GATES['min_s_spread']:.2f}, and rank correlation 0.000 against "
          f"{GATES['min_s_rank_corr']:.2f} -- excluded with certainty by "
          f"both, on any match set.")
    print(f"s, constant plus noise: passes magnitude and spread; caught only "
          f"by the ordering gate, which at nine matched calls a random answer "
          f"clears about one time in five. That is the honest limit of what "
          f"ten instances support.")

    dated = [s for s in truth["sweeps"] if s["class"] in TIMED_CLASSES
             and s.get("completion_gens_ago") is not None]
    times = [float(s["completion_gens_ago"]) for s in dated]
    best_t, best_tn = None, -1
    for t in range(1, 6001):
        n = sum(1 for tt in times if abs(t - tt) <= time_tolerance(tt))
        if n > best_tn:
            best_t, best_tn = float(t), n
    print(f"time, single constant: best is {best_t:.0f}, inside tolerance on "
          f"{best_tn}/{len(times)} (gate needs {GATES['min_time_correct']})")

    # One constant per class, which is what the previous age-based class
    # scheme allowed. With age out of the labels every dated sweep is in one
    # class, so this reduces to the line above -- shown explicitly because it
    # is the specific defect this redesign fixes.
    by_class = {}
    for s in dated:
        by_class.setdefault(s["class"], []).append(float(s["completion_gens_ago"]))
    total = 0
    for k, v in sorted(by_class.items()):
        bn = -1
        for t in range(1, 6001):
            n = sum(1 for tt in v if abs(t - tt) <= time_tolerance(tt))
            bn = max(bn, n)
        total += bn
        print(f"time, one constant for class {k}: best covers {bn}/{len(v)}")
    print(f"time, one constant per class: {total}/{len(times)} in total "
          f"(gate needs {GATES['min_time_correct']})")

    # Two constants, which is the limit of this gate and worth stating.
    best2, pair = 0, None
    grid = list(range(20, 3001, 20))
    for a_ in grid:
        for b_ in grid:
            n = sum(1 for tt in times
                    if min(abs(a_ - tt), abs(b_ - tt)) <= time_tolerance(tt))
            if n > best2:
                best2, pair = n, (a_, b_)
    print(f"time, TWO constants: best pair {pair} dates {best2}/{len(times)}. "
          f"With only four dateable sweeps no count-based gate can exclude\n"
          f"    that: a submission reporting two well-chosen numbers passes "
          f"the timing gate without dating anything. The gate excludes one\n"
          f"    constant, which is what it claims, and nothing stronger.")
    print(f"time, true values: {sorted(int(t) for t in times)}")


def design_guarantee():
    """Can a constant EVER date three hard_complete sweeps, for any draw?

    The check above uses the realized completion times of one replicate. This
    one uses the acceptance WINDOWS, so it holds for every replicate the
    generator would accept rather than for the one that happened.

    The tolerance half-width grows with the true time, so over a window
    [lo, hi] the set of constants that could cover a sweep somewhere in it is
    exactly [lo - tol(lo), hi + tol(hi)]. A constant can satisfy a set of
    sweeps only if their intervals share a point, so it suffices to check
    every triple.
    """
    import itertools
    need = GATES["min_time_correct"]
    ids = [sid for sid, r in sorted(C.ACCEPT.items())
           if r["class"] in TIMED_CLASSES and r["must_complete"]]
    span = {}
    for sid in ids:
        r = C.ACCEPT[sid]
        lo, hi = float(r["done_min"]), float(r["done_max"])
        span[sid] = (lo - time_tolerance(lo), hi + time_tolerance(hi))
    print("\n=== DESIGN GUARANTEE: windows, not one realized replicate ===")
    for sid in ids:
        r = C.ACCEPT[sid]
        a, b = span[sid]
        print(f"  {sid}: accepted {r['done_min']}-{r['done_max']} gens ago, "
              f"a constant in [{a:.0f}, {b:.0f}] could date it")
    worst = 0
    for combo in itertools.combinations(ids, need):
        lo = max(span[c][0] for c in combo)
        hi = min(span[c][1] for c in combo)
        if lo <= hi:
            worst = max(worst, len(combo))
            print(f"  *** {' + '.join(combo)} can all be dated by any constant "
                  f"in [{lo:.0f}, {hi:.0f}]")
    if worst == 0:
        print(f"  no constant can date {need} of the {len(ids)} accepted "
              f"hard_complete sweeps, for any draw inside the windows")
    else:
        raise SystemExit("the acceptance windows do not guarantee the timing "
                         "gate is non-trivial")


if __name__ == "__main__":
    feasibility()
    triviality()
    design_guarantee()
