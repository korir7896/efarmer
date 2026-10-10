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
from pipeline.characterize import (CLASS_WIDTH_CONSTANT, RECOVERY_FRAC,
                                   SMOOTH_WINDOWS, TIME_CLOCK_A, TIME_CLOCK_B)

CLASSES = ["hard_complete", "hard_partial", "soft_complete", "soft_partial"]


def smooth(v, k=SMOOTH_WINDOWS):
    out = np.full(v.size, np.nan); h = k // 2
    for i in range(v.size):
        a, b = max(0, i - h), min(v.size, i + h + 1)
        seg = v[a:b][np.isfinite(v[a:b])]
        if seg.size:
            out[i] = np.median(seg)
    return out


def trough_width(mid, v, bg, centre, frac=RECOVERY_FRAC):
    vs = smooth(v)
    near = (np.abs(mid - centre) <= 1_200_000) & np.isfinite(vs)
    if not near.any():
        return 0.0
    vmin = float(vs[near].min()); level = vmin + frac * (bg - vmin)
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
    lo = i0
    while lo - 1 >= 0 and below[lo - 1]:
        lo -= 1
    hi = i0
    while hi + 1 < len(m) and below[hi + 1]:
        hi += 1
    return float(m[hi] - m[lo])


def feasibility():
    per = {c: {"s_ok": 0, "s_n": 0, "t_ok": 0, "t_n": 0,
               "s_est": [], "t_est": []} for c in CLASSES}
    all_s_hat, all_s_true = [], []
    for f in sorted(glob.glob(os.path.join(HERE, "training", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        cls = str(d["realized_class"])
        if cls not in per:
            continue
        mid = d["mid"]; v = d["pi"].astype(float); bg = float(d["bg_pi"])
        centre = float(d["sweep_pos"]); s_true = float(d["s_true"])

        w = trough_width(mid, v, bg, centre)
        if w > 0:
            s_hat = w * C.BASE_REC_RATE / CLASS_WIDTH_CONSTANT[cls]
            per[cls]["s_n"] += 1
            per[cls]["s_est"].append(s_hat / s_true)
            all_s_hat.append(s_hat); all_s_true.append(s_true)
            if s_true / GATES["s_within_factor"] <= s_hat <= s_true * GATES["s_within_factor"]:
                per[cls]["s_ok"] += 1

        if cls in TIMED_CLASSES and "completed" in d.files and bool(d["completed"].item()):
            t_true = float(d["completion_gens_ago"])
            near = (np.abs(mid - centre) <= 1_000_000) & np.isfinite(v)
            if near.any() and bg:
                pir = float(v[near].min()) / bg
                t_hat = TIME_CLOCK_A * C.N_CURRENT * (pir ** TIME_CLOCK_B)
                per[cls]["t_n"] += 1
                per[cls]["t_est"].append(t_hat / t_true)
                if abs(t_hat - t_true) <= time_tolerance(t_true):
                    per[cls]["t_ok"] += 1

    print("=== FEASIBLE: the reference's estimators on independent replicates ===")
    print(f"selection coefficient, within a factor of {GATES['s_within_factor']} "
          f"of truth (gate: {GATES['min_s_accuracy']:.0%} of matched calls):")
    print(f"  {'class':<16}{'pass':>10}{'rate':>8}{'median est/true':>18}")
    for c in CLASSES:
        p = per[c]
        if p["s_n"]:
            print(f"  {c:<16}{p['s_ok']}/{p['s_n']:<8}{p['s_ok']/p['s_n']:>7.0%}"
                  f"{np.median(p['s_est']):>18.2f}")
    tot_ok = sum(per[c]["s_ok"] for c in CLASSES)
    tot_n = sum(per[c]["s_n"] for c in CLASSES)
    if tot_n:
        print(f"  {'ALL':<16}{tot_ok}/{tot_n:<8}{tot_ok/tot_n:>7.0%}")
        rho = spearman(all_s_hat, all_s_true)
        print(f"  rank correlation with truth {rho:.3f} over {tot_n} replicates "
              f"(gate: {GATES['min_s_rank_corr']:.2f})")

    print(f"\ncompletion time, within max({GATES['time_abs_tol_gens']} gens, "
          f"{GATES['time_rel_tol']:.0%}) of truth:")
    print(f"  {'class':<16}{'pass':>10}{'rate':>8}{'median est/true':>18}")
    for c in CLASSES:
        p = per[c]
        if p["t_n"]:
            print(f"  {c:<16}{p['t_ok']}/{p['t_n']:<8}{p['t_ok']/p['t_n']:>7.0%}"
                  f"{np.median(p['t_est']):>18.2f}")


def triviality():
    tp = os.path.join(TASK, "tests", "truth.json")
    if not os.path.exists(tp):
        print("\n(no shipped truth yet; skipping the triviality check)")
        return
    truth = json.load(open(tp))
    print("\n=== NOT TRIVIAL: the best an answer that estimates nothing can do ===")

    f = GATES["s_within_factor"]
    tru_s = [float(s["selection_coefficient"]) for s in truth["sweeps"]]
    lo, hi = min(tru_s) / f, max(tru_s) * f
    best_c, best_n = None, -1
    for i in range(4001):
        c = lo * (hi / lo) ** (i / 4000)
        n = sum(1 for t in tru_s if t / f <= c <= t * f)
        if n > best_n:
            best_c, best_n = c, n
    need = GATES["min_s_accuracy"]
    print(f"s, single constant: best is {best_c:.4f}, inside a factor of {f} "
          f"on {best_n}/{len(tru_s)} = {best_n/len(tru_s):.0%} "
          f"(magnitude gate needs {need:.0%})")
    print(f"s, single constant: rank correlation 0.000 "
          f"(ordering gate needs {GATES['min_s_rank_corr']:.2f}) -- "
          f"a constant cannot pass, whatever its value")
    print(f"s, true values span {max(tru_s)/min(tru_s):.2f}x: "
          f"{sorted(round(x, 4) for x in tru_s)}")

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
    print(f"time, true values: {sorted(int(t) for t in times)}")


if __name__ == "__main__":
    feasibility()
    triviality()
