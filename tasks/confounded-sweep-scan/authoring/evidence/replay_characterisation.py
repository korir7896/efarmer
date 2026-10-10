#!/usr/bin/env python3
"""Re-derive the characterisation step from a dumped scan.

`solution/pipeline/main.py --dump-scan` saves the windowed scan, the combined
scores and the called regions. This replays everything downstream of
detection against the sealed truth, in a second rather than the eight
minutes a full run takes, so a classifier or an estimator can be judged on
the shipped replicate without re-simulating the null ten times.

It reads the sealed truth, so it is an AUTHORING tool and nothing in the
shipped bundle depends on it.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(TASK, "solution"))
sys.path.insert(0, os.path.join(TASK, "tests"))

from pipeline import characterize as CH
from pipeline import detect as DET
from grading import GATES, TIMED_CLASSES, spearman, time_tolerance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--truth", default=os.path.join(TASK, "tests", "truth.json"))
    a = ap.parse_args()

    d = np.load(a.scan, allow_pickle=True)
    obs = {k[4:]: d[k] for k in d.files if k.startswith("obs_")}
    score = d["score"]
    bins = d["bins"]
    regions = json.loads(str(d["regions"]))
    hist = json.loads(str(d["hist"]))
    truth = json.load(open(a.truth))
    chrom_len = truth["genome"]["chromosome_length_bp"]

    bg_by_bin = CH.background_levels(obs, regions, bins)

    print(f"{'call':<22}{'matched':<8}{'true class':<15}{'called':<15}"
          f"{'s true':>8}{'s hat':>8}{'t true':>8}{'t hat':>8}")
    rows = []
    for r in regions:
        b = int(bins[int(r["peak_index"])])
        lev = bg_by_bin.get(b) or next(iter(bg_by_bin.values()))
        local_rate = float(obs["rate"][int(r["peak_index"])])
        f = CH.features(obs, r, lev)
        f.update(CH.shape_features(obs, r, lev, local_rate))
        klass = CH.classify(f)
        s_hat = CH.estimate_s(f, klass)
        t_hat = CH.estimate_time(f, klass)

        best, bd = None, None
        for t in truth["sweeps"]:
            if t["chrom"] != r["chrom"]:
                continue
            dist = abs(t["position_local"] - r["position"])
            if bd is None or dist < bd:
                best, bd = t, dist
        matched = best if (bd is not None and bd <= GATES["match_window_bp"]) else None
        print(f"chr{r['chrom']}:{r['position']:<16,}"
              f"{(str(round(bd/1000)) + 'kb') if bd is not None else '-':<8}"
              f"{(matched['class'] if matched else 'FALSE POS'):<15}{klass:<15}"
              f"{(matched['selection_coefficient'] if matched else 0):>8.4f}"
              f"{s_hat:>8.4f}"
              f"{str(matched['completion_gens_ago'] if matched else '-'):>8}"
              f"{(round(t_hat) if t_hat else 0):>8}")
        rows.append((r, f, klass, s_hat, t_hat, matched, bd))

    matched_rows = [x for x in rows if x[5] is not None]
    print(f"\n{len(matched_rows)} matched, {len(rows) - len(matched_rows)} "
          f"false positive(s) (cap {GATES['max_false_positives']})")
    found = {}
    for x in matched_rows:
        found[x[5]["class"]] = found.get(x[5]["class"], 0) + 1
    for k, need in GATES["min_recall_per_class"].items():
        tot = sum(1 for t in truth["sweeps"] if t["class"] == k)
        print(f"  recall {k:<15}{found.get(k, 0)}/{tot}  (need {need})")
    if matched_rows:
        right = sum(1 for x in matched_rows if x[2] == x[5]["class"])
        print(f"  class accuracy {right}/{len(matched_rows)} = "
              f"{right/len(matched_rows):.2f} (need "
              f"{GATES['min_class_accuracy']:.2f})")
        okf = sum(1 for x in matched_rows
                  if x[5]["selection_coefficient"] / GATES["s_within_factor"]
                  <= x[3] <= x[5]["selection_coefficient"] * GATES["s_within_factor"])
        print(f"  s within {GATES['s_within_factor']}: {okf}/{len(matched_rows)} "
              f"= {okf/len(matched_rows):.2f} (need {GATES['min_s_accuracy']:.2f})")
        rho = spearman([x[3] for x in matched_rows],
                       [x[5]["selection_coefficient"] for x in matched_rows])
        print(f"  s rank correlation {rho:.3f} (need "
              f"{GATES['min_s_rank_corr']:.2f})")
        dated = [x for x in matched_rows if x[5]["class"] in TIMED_CLASSES
                 and x[5]["completion_gens_ago"] is not None]
        ok_t = sum(1 for x in dated
                   if x[4] is not None
                   and abs(x[4] - x[5]["completion_gens_ago"])
                   <= time_tolerance(x[5]["completion_gens_ago"]))
        print(f"  timing {ok_t}/{len(dated)} inside tolerance (need "
              f"{GATES['min_time_correct']})")

    print("\nfeatures at each call:")
    keys = ["pi_ratio", "snp_ratio", "tajd_delta", "h12_ratio", "h2h1_ratio",
            "nsl_delta", "w_pi_50", "w_pi_75", "w_h12_15", "w_h12_2"]
    print(f"  {'call':<20}" + "".join(f"{k:>12}" for k in keys))
    for r, f, klass, s_hat, t_hat, matched, bd in rows:
        print(f"  chr{r['chrom']}:{r['position']:<14,}"
              + "".join(f"{f.get(k, float('nan')):>12.4g}" for k in keys))
    print(f"\nfitted history: {hist}")


if __name__ == "__main__":
    main()
