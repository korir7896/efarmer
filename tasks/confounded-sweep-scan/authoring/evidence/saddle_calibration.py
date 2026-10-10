#!/usr/bin/env python3
"""Where detect._merge_regions' saddle threshold comes from.

Two clusters close together are either one footprint broken by noise, which
should be fused, or two sweeps, which must not be. Distance cannot tell them
apart here: two selected sites sit 753 and 826 kb apart, and a merge radius
narrow enough to keep them separate is too narrow to repair a split
footprint, which can throw clusters more than a megabase either side of the
sweep.

What does tell them apart is the SADDLE: the lowest combined score between
the two peaks, as a fraction of the weaker peak. This prints that ratio for
every candidate pair the detector considers, labelled by what the sealed
truth says the pair actually is, so the threshold can be read off two
populations instead of guessed.

It reads the sealed truth, so it is an AUTHORING tool and nothing in the
shipped bundle depends on it.

    saddle_calibration.py SCAN.npz [--truth PATH] [--radius BP]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(TASK, "solution"))
sys.path.insert(0, os.path.join(TASK, "tests"))

from grading import GATES
from pipeline import detect as DET


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--truth", default=os.path.join(TASK, "tests", "truth.json"))
    ap.add_argument("--radius", type=int, default=1_500_000)
    a = ap.parse_args()

    d = np.load(a.scan, allow_pickle=True)
    obs = {k[4:]: d[k] for k in d.files if k.startswith("obs_")}
    score = d["score"]
    form = float(d["form_thresh"])
    mass = float(d["mass_thresh"])
    truth = json.load(open(a.truth))

    # Clusters BEFORE any merging, which is what the merge step is handed.
    raw = []
    for run in DET.cluster_runs(obs["chrom"], obs["mid"], score, form,
                               max_gap_windows=10):
        if DET.cluster_mass(score, run, form) < mass:
            continue
        raw.append(DET._summarize(obs, score, list(run),
                                  int(obs["chrom"][run[0]])))
    raw.sort(key=lambda r: (r["chrom"], r["position"]))
    print(f"{len(raw)} cluster(s) clear the mass cut before merging")

    def true_sweeps_between(chrom, lo, hi):
        """True sweeps bracketed by a pair, counting the match window."""
        out = []
        for t in truth["sweeps"]:
            if t["chrom"] != chrom:
                continue
            p = t["position_local"]
            if lo - GATES["match_window_bp"] <= p <= hi + GATES["match_window_bp"]:
                out.append(t["sweep_id"])
        return out

    rows = []
    for left, right in zip(raw, raw[1:]):
        if left["chrom"] != right["chrom"]:
            continue
        gap = right["position"] - left["position"]
        if gap >= a.radius:
            continue
        mid = obs["mid"]
        between = np.where((mid > left["position"]) & (mid < right["position"])
                           & (obs["chrom"] == left["chrom"]))[0]
        v = score[between]
        v = v[np.isfinite(v)]
        weaker = min(float(left["peak_score"]), float(right["peak_score"]))
        ratio = (float(np.min(v)) / weaker) if v.size and weaker > 0 else float("nan")
        ids = true_sweeps_between(int(left["chrom"]),
                                 min(left["position"], right["position"]),
                                 max(left["position"], right["position"]))
        rows.append((ratio, gap, int(left["chrom"]), left["position"],
                     right["position"], ids))

    if not rows:
        print("no candidate pairs inside the radius; nothing to calibrate")
        return 0

    print(f"\n{'saddle':>8}{'gap bp':>11}  chrom  left..right  true sweeps "
          f"bracketed -> verdict")
    one, many = [], []
    for ratio, gap, ch, lp, rp, ids in sorted(rows):
        verdict = "ONE footprint, fuse" if len(ids) <= 1 else "TWO sweeps, keep apart"
        (one if len(ids) <= 1 else many).append(ratio)
        print(f"{ratio:>8.3f}{gap:>11,}  chr{ch + 1}  {lp:,}..{rp:,}  "
              f"{','.join(ids) or '-':<12} -> {verdict}")

    print()
    if one and many:
        lo_one, hi_many = min(one), max(many)
        print(f"split footprints: saddle ratio {min(one):.3f} to {max(one):.3f}")
        print(f"distinct sweeps : saddle ratio {min(many):.3f} to {max(many):.3f}")
        if lo_one > hi_many:
            print(f"the two populations are disjoint; any threshold in "
                  f"({hi_many:.3f}, {lo_one:.3f}) separates them, and the "
                  f"shipped value is {DET._merge_regions.__defaults__[-1]:.2f}")
        else:
            print(f"WARNING: the populations OVERLAP "
                  f"({hi_many:.3f} >= {lo_one:.3f}); no saddle threshold "
                  f"separates every pair on this replicate")
    else:
        which = "split footprints" if many else "distinct-sweep pairs"
        print(f"only one population is present on this replicate (no "
              f"{which}), so the threshold cannot be read off it alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
