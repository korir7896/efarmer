#!/usr/bin/env python3
"""Why detect._merge_regions fuses on the diversity trough and not the saddle.

Two clusters close together are either one footprint broken by noise, which
should be fused, or two sweeps, which must not be. Three rules were tried and
this prints the evidence for all three on one replicate:

  DISTANCE cannot do it. Two selected sites sit 725 and 753 kb apart, and a
    merge radius narrow enough to keep them separate is too narrow to repair
    a split footprint, which can throw clusters more than a megabase either
    side of the sweep.

  The SADDLE -- the lowest combined score between the peaks, as a fraction of
    the weaker peak -- cannot do it either, and that is the measurement this
    script exists to show. On the shipped replicate the saddle inside
    footprints that split overlaps the saddle at the genuine close pair, so
    no threshold separates them. A sweep inside a wide recombination cold
    spot has real structure across two megabases of footprint, as deep as the
    gap between two distinct sweeps.

  The DIVERSITY TROUGH does, because it is one object per sweep rather than a
    noisy field. Two pieces of one footprint lie inside one contiguous
    low-diversity run; two distinct sweeps have two runs that do not contain
    each other's peak.

It reads the sealed truth, so it is an AUTHORING tool and nothing in the
shipped bundle depends on it.

    merge_calibration.py [SCAN.npz] [--selftest] [--truth PATH] [--radius BP]
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

from pipeline import detect as DET          # noqa: E402
from grading import GATES                   # noqa: E402


def selftest() -> int:
    """Check the trough rule on synthetic pairs with known geometry.

    The real-data measurement can only show the two populations the shipped
    replicate happens to contain. This checks the decision itself on pairs
    built by hand, all of them 800 kb apart -- inside the merge radius and
    above the unconditional one -- so the decision has to be made on the
    trough and not on distance.

    Each case sets the diversity between the two peaks. One footprint has a
    continuous trough across the whole span; two sweeps have diversity
    recovering to background between them.
    """
    step = 10_000
    mid = np.arange(0, 4_000_000, step, dtype=float)
    chrom = np.zeros(mid.size, dtype=int)
    frac = DET._merge_regions.__defaults__[-1]
    print(f"trough rule self-test (trough_frac = {frac}, "
          f"unconditional radius = {DET.ALWAYS_MERGE_BP:,} bp)")
    print(f"  {'case':<52}{'merged':>8}{'expected':>10}  verdict")
    # Fraction of background that diversity recovers to between the peaks.
    cases = [("one footprint, trough continuous (recovers to 0.20)", 0.20, True),
             ("one footprint, shallow saddle (recovers to 0.45)", 0.45, True),
             ("two sweeps, diversity back to 0.85 of background", 0.85, False),
             ("two sweeps, diversity back to background", 1.00, False)]
    bad = 0
    left_p, right_p = 1_400_000.0, 2_200_000.0
    for label, recover, expect in cases:
        bg = 1.0
        pi = np.full(mid.size, bg)
        for p_ in (left_p, right_p):
            pi -= 0.95 * bg * np.exp(-0.5 * ((mid - p_) / 160_000.0) ** 2)
        between = (mid > left_p) & (mid < right_p)
        pi[between] = np.minimum(pi[between], recover * bg)
        # A score field with two clean peaks, so only the trough can decide.
        sc = np.full(mid.size, 5.0)
        for p_ in (left_p, right_p):
            sc = np.maximum(sc, 5.0 + 95.0
                            * np.exp(-0.5 * ((mid - p_) / 180_000.0) ** 2))
        obs = {"mid": mid, "chrom": chrom, "h12": sc, "pi": pi}
        left = DET._summarize(obs, sc, list(np.where(
            np.abs(mid - left_p) < 300_000)[0]), 0)
        right = DET._summarize(obs, sc, list(np.where(
            np.abs(mid - right_p) < 300_000)[0]), 0)
        merged = len(DET._merge_regions(obs, sc, [left, right], 1_500_000)) == 1
        ok = merged == expect
        bad += not ok
        print(f"  {label:<52}{str(merged):>8}{str(expect):>10}  "
              f"{'ok' if ok else 'WRONG'}")
    print("  " + ("all four behaved as intended" if not bad
                  else f"{bad} case(s) wrong"))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", nargs="?")
    ap.add_argument("--selftest", action="store_true",
                    help="check the rule on synthetic pairs and stop")
    ap.add_argument("--truth", default=os.path.join(TASK, "tests", "truth.json"))
    ap.add_argument("--radius", type=int, default=1_500_000)
    a = ap.parse_args()

    if a.selftest or not a.scan:
        return selftest()

    rc = selftest()
    print()

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

    def bracketed(chrom, lo, hi):
        """True sweeps a pair brackets, counting the match window."""
        w = GATES["match_window_bp"]
        return [t["sweep_id"] for t in truth["sweeps"]
                if int(t["chrom"]) == chrom
                and lo - w <= t["position_local"] <= hi + w]

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
        saddle = (float(np.min(v)) / weaker) if v.size and weaker > 0 \
            else float("nan")
        one_trough = DET._is_one_footprint(
            obs, score, left, right, DET._merge_regions.__defaults__[-1])
        ids = bracketed(int(left["chrom"]),
                        min(left["position"], right["position"]),
                        max(left["position"], right["position"]))
        rows.append((gap, saddle, one_trough, int(left["chrom"]),
                     left["position"], right["position"], ids))

    if not rows:
        print("no candidate pairs inside the radius; nothing to calibrate")
        return rc

    print(f"\n{'gap bp':>11}{'saddle':>9}{'one trough':>12}  chrom  "
          f"left..right  bracketed -> what it is")
    one, many = [], []
    bad_trough = 0
    for gap, saddle, one_trough, ch, lp, rp, ids in sorted(rows):
        # One sweep bracketed (or none) means the pair is two pieces of one
        # thing or a sweep beside a false call; two means two sweeps.
        two_sweeps = len(ids) >= 2
        want_fuse = not two_sweeps and len(ids) == 1
        label = ("TWO sweeps, keep apart" if two_sweeps
                 else "ONE footprint, fuse" if len(ids) == 1
                 else "no sweep either side, keep apart")
        (many if two_sweeps else one).append(saddle)
        # Below the unconditional radius the trough is not consulted at all.
        decided = ("unconditional" if gap < DET.ALWAYS_MERGE_BP
                   else str(one_trough))
        if gap >= DET.ALWAYS_MERGE_BP and one_trough != want_fuse:
            bad_trough += 1
        print(f"{gap:>11,}{saddle:>9.3f}{decided:>12}  chr{ch}  "
              f"{lp:,}..{rp:,}  {','.join(ids) or '-':<12} -> {label}")

    print()
    if one and many:
        lo_one, hi_many = min(one), max(many)
        print(f"saddle inside split footprints: {min(one):.3f} to {max(one):.3f}")
        print(f"saddle at distinct sweeps     : {min(many):.3f} to {max(many):.3f}")
        if lo_one > hi_many:
            print(f"  the saddle populations are disjoint on this replicate; "
                  f"a threshold in ({hi_many:.3f}, {lo_one:.3f}) would "
                  f"separate them")
        else:
            print(f"  the saddle populations OVERLAP ({hi_many:.3f} >= "
                  f"{lo_one:.3f}): NO saddle threshold separates every pair "
                  f"on this replicate, which is why the shipped rule does not "
                  f"use one")
    else:
        print("only one kind of pair is present, so the saddle populations "
              "cannot be compared on this replicate")

    print(f"\ntrough rule on the pairs it decides: "
          + (f"{bad_trough} wrong" if bad_trough else "every pair correct"))
    return rc or (1 if bad_trough else 0)


if __name__ == "__main__":
    sys.exit(main())
