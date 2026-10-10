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
replicate with those constants by replaying a dumped scan. Both the feature SETS and the coefficients are re-derived: holding the
feature sets fixed is not enough, because selecting them is itself fitting.
The first version of this script refitted only the coefficients and still
regraded at a rank correlation of 0.267, because the pair chosen for
soft_complete on the shipped configurations carries almost no information
about s away from them.

Usage:  holdout_calibration.py SCAN.npz
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(TASK, "solution"))
sys.path.insert(0, os.path.join(TASK, "tests"))

from fit_classifier import (axis_all, logistic_fit, predict, standardize,
                            FEATS as CLS_FEATS)
from fit_estimators import held_out_loglinear, rows
from grading import GATES, TIMED_CLASSES, spearman, time_tolerance
from pipeline import characterize as CH
from pipeline import detect as DET

CLASSES = ["hard_complete", "hard_partial", "soft_complete", "soft_partial"]


def select_rules(data):
    """Re-select each axis's feature set on the holdout replicates alone."""
    import itertools
    singles = [(k,) for k in CLS_FEATS]
    pairs = list(itertools.combinations(CLS_FEATS, 2))
    picks = {}
    specs = {
        "complete": (lambda r: r["cls"].endswith("complete"), None),
        "soft_given_complete": (lambda r: r["cls"].startswith("soft"),
                                lambda r: r["cls"].endswith("complete")),
        "soft_given_partial": (lambda r: r["cls"].startswith("soft"),
                               lambda r: not r["cls"].endswith("complete")),
    }
    for name, (label, subset) in specs.items():
        scored = axis_all(data, label, singles + pairs, subset_fn=subset)
        scored.sort(key=lambda t: -t[0])
        picks[name] = scored[0][1] if scored else CH.CLASS_RULES[name]["feats"]
        if scored:
            print(f"  {name}: picked {'+'.join(picks[name])} "
                  f"(held out {scored[0][0]:.3f})")
    return picks


def refit_rules(data, picks):
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
        feats = list(picks[name])
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


def _select_loglinear(data, cands, target, cls=None, completed=False):
    """Pick a feature set by held-out score on the holdout replicates only."""
    best, best_key = None, None
    for combo in cands:
        rs = [r for r in data if (cls is None or r["cls"] == cls)
              and (not completed or r["completed"])
              and np.isfinite(r[target]) and r[target] > 0
              and all(np.isfinite(r[k]) and r[k] > 0 for k in combo)]
        if len(rs) < len(combo) + 4 or len(set(r["cfg"] for r in rs)) < 3:
            continue
        X = np.log(np.array([[r[k] for k in combo] for r in rs], float))
        y = np.log(np.array([r[target] for r in rs], float))
        p = held_out_loglinear(X, y, [r["cfg"] for r in rs])
        ok = np.isfinite(p)
        if ok.sum() < 3:
            continue
        err = np.exp(np.abs(p[ok] - y[ok]))
        key = (float((err <= GATES["s_within_factor"]).mean()),
               spearman(list(np.exp(p[ok])), list(np.exp(y[ok]))))
        if best_key is None or key > best_key:
            best, best_key = list(combo), key
    return best


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

    print("\nclassification rules (features re-selected on holdout only):")
    picks = select_rules(data)
    CH.CLASS_RULES = refit_rules(data, picks)

    print("\nselection-coefficient models (features re-selected too):")
    import itertools
    keys = ["pi_ratio", "pi_min_ratio", "h12_ratio", "snp_ratio",
            "w_pi_50", "w_pi_75", "w_h12_2", "w_h12_15"]
    cands = [(k,) for k in keys] + list(itertools.combinations(keys, 2))
    for c in CLASSES:
        feats = _select_loglinear(data, cands, "s_true", cls=c) \
            or list(CH.S_MODEL[c]["feats"])
        coef, n = refit_loglinear(data, feats, "s_true", cls=c)
        if coef is None:
            print(f"  ! {c}: only {n} usable replicates, keeping shipped model")
            continue
        CH.S_MODEL[c] = dict(feats=feats, coef=coef)
        print(f"  {c}: {'+'.join(feats)}, n={n}")

    feats = _select_loglinear(data, cands, "t_true", cls="hard_complete",
                              completed=True) or list(CH.T_MODEL["feats"])
    coef, n = refit_loglinear(data, feats, "t_true",
                              cls="hard_complete", completed=True)
    if coef is not None:
        CH.T_MODEL = dict(feats=feats, coef=coef)
        print(f"\ncompletion-time model: {'+'.join(feats)}, n={n}")
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
