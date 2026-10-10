#!/usr/bin/env python3
"""Refit every calibrated constant on configurations the shipped replicate
does not use, then regrade the shipped replicate.

The constants in solution/pipeline/characterize.py are fitted on single-sweep
replicates whose (s, origin, n_origins) ARE the shipped sweeps'. That is a
real weakness in the case that the gates are passable: a method calibrated at
the very points it is graded on has an advantage no independent
implementation would have, and leave-one-configuration-out scoring does not
remove it, because the held-out configuration is still one of the ten being
graded.

This script removes it. It refits the three classification rules, the four
selection-coefficient models and the completion-time model on
authoring/evidence/training_holdout/, every configuration of which differs
from all ten shipped ones in both s and origin, and then grades the shipped
replicate with those constants by replaying a dumped scan. The feature SETS
are held fixed, since those are a modelling choice rather than a fitted
quantity; only the numbers are refitted.

Usage:  holdout_calibration.py SCAN.npz
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(TASK, "solution"))
sys.path.insert(0, os.path.join(TASK, "tests"))

from fit_classifier import logistic_fit, predict
from fit_estimators import rows
from grading import GATES, TIMED_CLASSES, spearman, time_tolerance
from pipeline import characterize as CH
from pipeline import detect as DET

CLASSES = ["hard_complete", "hard_partial", "soft_complete", "soft_partial"]


def refit_rules(data):
    """The three logistic rules, on the holdout replicates only."""
    out = {}
    groups = {
        "complete": (data, lambda r: r["cls"].endswith("complete")),
        "soft_given_complete": ([r for r in data if r["cls"].endswith("complete")],
                                lambda r: r["cls"].startswith("soft")),
        "soft_given_partial": ([r for r in data if not r["cls"].endswith("complete")],
                               lambda r: r["cls"].startswith("soft")),
    }
    for name, (rs, label) in groups.items():
        feats = list(CH.CLASS_RULES[name]["feats"])
        rs = [r for r in rs if all(np.isfinite(r[k]) for k in feats)]
        y = np.array([1.0 if label(r) else 0.0 for r in rs])
        if len(rs) < len(feats) + 3 or len(set(y)) < 2:
            print(f"  ! {name}: only {len(rs)} usable replicates, keeping shipped rule")
            out[name] = CH.CLASS_RULES[name]
            continue
        X = np.array([[r[k] for k in feats] for r in rs], float)
        mu = X.mean(0); sd = np.where(X.std(0) > 0, X.std(0), 1.0)
        w = logistic_fit((X - mu) / sd, y)
        acc = float(((predict(w, (X - mu) / sd) >= 0.5) == (y >= 0.5)).mean())
        print(f"  {name}: n={len(rs)}, in-sample {acc:.3f}")
        out[name] = dict(feats=feats, mean=list(mu), sd=list(sd), weights=list(w))
    return out


def refit_loglinear(data, feats, target, cls=None, completed=False):
    rs = [r for r in data if (cls is None or r["cls"] == cls)
          and (not completed or r["completed"])
          and np.isfinite(r[target]) and r[target] > 0
          and all(np.isfinite(r[k]) and r[k] > 0 for k in feats)]
    if len(rs) < len(feats) + 2:
        return None, len(rs)
    X = np.log(np.array([[r[k] for k in feats] for r in rs], float))
    y = np.log(np.array([r[target] for r in rs], float))
    A = np.column_stack([np.ones(len(rs))] + [X[:, j] for j in range(X.shape[1])])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return list(coef), len(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--holdout", default="training_holdout")
    a = ap.parse_args()

    data = rows(a.holdout)
    if not data:
        raise SystemExit(f"no replicates in authoring/evidence/{a.holdout}")
    by_cfg = {}
    for r in data:
        by_cfg.setdefault(r["cfg"], []).append(r["cls"])
    print(f"refitting on {len(data)} replicates, "
          f"{len(by_cfg)} configurations, none of them shipped:")
    for k, v in sorted(by_cfg.items()):
        print(f"  {k:<28}{len(v)} replicate(s), realized {sorted(set(v))}")

    print("\nclassification rules:")
    CH.CLASS_RULES = refit_rules(data)

    print("\nselection-coefficient models:")
    for c in CLASSES:
        feats = list(CH.S_MODEL[c]["feats"])
        coef, n = refit_loglinear(data, feats, "s_true", cls=c)
        if coef is None:
            print(f"  ! {c}: only {n} usable replicates, keeping shipped model")
            continue
        CH.S_MODEL[c] = dict(feats=feats, coef=coef)
        print(f"  {c}: n={n}, coef {[round(v, 4) for v in coef]}")

    feats = list(CH.T_MODEL["feats"])
    coef, n = refit_loglinear(data, feats, "t_true",
                              cls="hard_complete", completed=True)
    if coef is not None:
        CH.T_MODEL = dict(feats=feats, coef=coef)
        print(f"\ncompletion-time model: n={n}, coef {[round(v, 4) for v in coef]}")
    else:
        print(f"\n! completion-time model: only {n} usable replicates, keeping shipped")

    # ---- regrade the shipped replicate with the refitted constants ----
    d = np.load(a.scan, allow_pickle=True)
    obs = {k[4:]: d[k] for k in d.files if k.startswith("obs_")}
    bins = d["bins"]
    regions = json.loads(str(d["regions"]))
    truth = json.load(open(os.path.join(TASK, "tests", "truth.json")))
    bg_by_bin = CH.background_levels(obs, regions, bins)

    rows_out = []
    for r in regions:
        b = int(bins[int(r["peak_index"])])
        lev = bg_by_bin.get(b) or next(iter(bg_by_bin.values()))
        rate = float(obs["rate"][int(r["peak_index"])])
        f = CH.features(obs, r, lev)
        f.update(CH.shape_features(obs, r, lev, rate))
        klass = CH.classify(f)
        best, bd = None, None
        for t in truth["sweeps"]:
            if t["chrom"] != r["chrom"]:
                continue
            dist = abs(t["position_local"] - r["position"])
            if bd is None or dist < bd:
                best, bd = t, dist
        matched = best if (bd is not None and bd <= GATES["match_window_bp"]) else None
        rows_out.append((klass, CH.estimate_s(f, klass),
                         CH.estimate_time(f, klass), matched))

    m = [x for x in rows_out if x[3] is not None]
    print(f"\n=== regraded with holdout-only constants ===")
    print(f"  {len(m)} matched, {len(rows_out) - len(m)} false positive(s) "
          f"(cap {GATES['max_false_positives']})")
    right = sum(1 for x in m if x[0] == x[3]["class"])
    print(f"  class accuracy {right}/{len(m)} = {right/len(m):.2f} "
          f"(need {GATES['min_class_accuracy']:.2f})")
    f_ = GATES["s_within_factor"]
    oks = sum(1 for x in m if x[3]["selection_coefficient"] / f_ <= x[1]
              <= x[3]["selection_coefficient"] * f_)
    print(f"  s within {f_}: {oks}/{len(m)} = {oks/len(m):.2f} "
          f"(need {GATES['min_s_accuracy']:.2f})")
    print(f"  s rank correlation "
          f"{spearman([x[1] for x in m], [x[3]['selection_coefficient'] for x in m]):.3f} "
          f"(need {GATES['min_s_rank_corr']:.2f})")
    dated = [x for x in m if x[3]["class"] in TIMED_CLASSES
             and x[3]["completion_gens_ago"] is not None]
    okt = sum(1 for x in dated if x[2] is not None
              and abs(x[2] - x[3]["completion_gens_ago"])
              <= time_tolerance(x[3]["completion_gens_ago"]))
    print(f"  timing {okt}/{len(dated)} inside tolerance "
          f"(need {GATES['min_time_correct']})")
    print("\nDetection is unchanged: it is calibrated at run time against the "
          "fitted history,\nso recall and the false-positive count do not "
          "depend on these constants.")


if __name__ == "__main__":
    main()
