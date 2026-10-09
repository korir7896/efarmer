#!/usr/bin/env python3
"""Measure realized sweep outcomes to set parameters and acceptance windows.

A sweep's trajectory depends on N and s, not on genome length, so this runs
on a short genome and is therefore cheap enough to replicate properly.  The
acceptance windows in config.ACCEPT are set from the spread measured here,
not from a single run.
"""
import argparse, json, os, statistics, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "provenance"))

import numpy as np
from genome import GenomeConfig
from forward import SweepPlan, run_forward

N_CUR = 9000
GL = 1_000_000


def trial(s, origin, ndesc, t_forward, seed):
    cfg = GenomeConfig(n_chrom=1, chrom_len=GL, cold_spots=[])
    plans = [SweepPlan("x", GL // 2 + 7919, s, 1.0, origin, ndesc, "probe")]
    _, rec, _ = run_forward(cfg, N_CUR, t_forward, plans, seed=seed, verbose=False)
    r = rec.real["x"]
    return dict(fixed=r.fixed, fix_ago=r.fixation_gens_ago if r.fixed else None,
                freq=r.final_freq, attempts=r.establishment_attempts)


def summarize(name, s, origin, ndesc, t_forward, reps, seed0):
    rows = [trial(s, origin, ndesc, t_forward, seed0 + 17 * i) for i in range(reps)]
    nfix = sum(r["fixed"] for r in rows)
    out = {"config": dict(name=name, s=s, origin=origin, ndesc=ndesc), "reps": reps,
           "n_fixed": nfix}
    if nfix:
        fa = sorted(r["fix_ago"] for r in rows if r["fixed"])
        out["fix_ago"] = {"min": fa[0], "p25": int(np.percentile(fa, 25)),
                          "median": int(np.median(fa)), "p75": int(np.percentile(fa, 75)),
                          "max": fa[-1]}
    seg = sorted(r["freq"] for r in rows if not r["fixed"])
    if seg:
        out["seg_freq"] = {"min": round(seg[0], 3), "median": round(float(np.median(seg)), 3),
                           "max": round(seg[-1], 3)}
    out["mean_attempts"] = round(statistics.mean(r["attempts"] for r in rows), 1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()

    # h=1.0 => additive, per-allele advantage s. Expectations used to pick
    # these: P(fix from one copy) ~ 2s; t_fix ~ (2/s)*ln(4Ns);
    # time to reach frequency f from one copy ~ (1/s)*ln(2N*f/(1-f)).
    GRID = [
        # complete, recent: want fixation 150-600 generations ago
        ("recent_a", 0.035,  750, 1),
        ("recent_b", 0.030,  820, 1),
        ("recent_c", 0.040,  680, 1),
        # complete, old: want fixation 1100-1900 generations ago
        ("old_a",    0.028, 2150, 1),
        ("old_b",    0.024, 2300, 1),
        # soft from standing variation: want fixation 150-900 generations ago
        ("soft_a",   0.028,  900, 24),
        ("soft_b",   0.032,  780, 18),
        ("soft_c",   0.026, 1050, 32),
        # partial: want final frequency ~0.4-0.9 and never fixed
        ("part_a",   0.018,  600, 1),
        ("part_b",   0.022,  480, 1),
        ("part_c",   0.018,  520, 1),
        ("part_d",   0.015,  620, 1),
    ]
    res = []
    # Only the span from a sweep's origin to the present affects its
    # trajectory; the constant-size burn-in before it does not. Simulating
    # just that span makes replication affordable.
    for i, (name, s, origin, nd) in enumerate(GRID):
        if i % a.nshards != a.shard:
            continue
        tf = origin + 30
        r = summarize(name, s, origin, nd, tf, a.reps, 900 + 1000 * i)
        res.append(r)
        print(json.dumps(r), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
