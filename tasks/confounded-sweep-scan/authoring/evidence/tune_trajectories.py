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
                completed=r.completed,
                done_ago=r.completion_gens_ago if r.completed else None,
                origins=r.n_origins_surviving,
                freq=r.final_freq, attempts=r.establishment_attempts)


def summarize(name, s, origin, ndesc, t_forward, reps, seed0):
    rows = [trial(s, origin, ndesc, t_forward, seed0 + 17 * i) for i in range(reps)]
    nfix = sum(r["fixed"] for r in rows)
    ndone = sum(r["completed"] for r in rows)
    out = {"config": dict(name=name, s=s, origin=origin, ndesc=ndesc), "reps": reps,
           "n_fixed": nfix, "n_completed": ndone}
    if nfix:
        fa = sorted(r["fix_ago"] for r in rows if r["fixed"])
        out["fix_ago"] = {"min": fa[0], "p25": int(np.percentile(fa, 25)),
                          "median": int(np.median(fa)), "p75": int(np.percentile(fa, 75)),
                          "max": fa[-1]}
    if ndone:
        da = sorted(r["done_ago"] for r in rows if r["completed"])
        out["done_ago"] = {"min": da[0], "p25": int(np.percentile(da, 25)),
                           "median": int(np.median(da)), "p75": int(np.percentile(da, 75)),
                           "max": da[-1]}
    orig = sorted(r["origins"] for r in rows)
    out["origins_surviving"] = {"min": orig[0], "median": int(np.median(orig)),
                                "max": orig[-1],
                                "n_ge2": int(sum(o >= 2 for o in orig))}
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
    ap.add_argument("--grid", default="v3", choices=["v1", "v2", "v3"])
    a = ap.parse_args()

    # h=1.0 => additive, per-allele advantage s. Expectations used to pick
    # these: P(fix from one copy) ~ 2s; t_fix ~ (2/s)*ln(4Ns);
    # time to reach frequency f from one copy ~ (1/s)*ln(2N*f/(1-f)).
    # Second pass: only the partial sweeps needed retuning. Conditioning on
    # establishment skips the slow stochastic phase, so a conditioned sweep
    # rises as if it started near 1/(2Ns) rather than at one copy, and the
    # logistic-from-one-copy estimate of the origin time overshoots badly
    # (measured medians of 0.92-0.95 where 0.65 was intended).
    GRID_V1 = [
        ("part_s018_t430", 0.018, 430, 1),
        ("part_s022_t390", 0.022, 390, 1),
        ("part_s018_t400", 0.018, 400, 1),
        ("part_s022_t360", 0.022, 360, 1),
    ]

    # ---------------- v2 grid: the 2x2 zoo ----------------
    # The shipped zoo was redesigned so that the graded quantities
    # discriminate: the four classes are now mechanism-based
    # ({hard, soft} x {complete, partial}) rather than age-based, s spans a
    # factor of 9.4, and completion time is spread within the one class that
    # is dated. Both changes need realizability measured per configuration.
    #
    # Deterministic expectations used to seed the grid (N=9000, h=1.0):
    #   t_fix ~ (2/s)*ln(4Ns) from one copy;
    #   time from k copies to frequency f ~ (1/s)*ln(2N*f/((1-f)*k)).
    # Conditioning on establishment skips the stochastic phase, so realized
    # sweeps run faster than these and the measured values below are what
    # the shipped origins are set from.
    GRID_V2 = [
        # hard_complete: four widely spread completion times, s deliberately
        # NOT monotone in age so neither can be read off the other.
        ("hc_a_s0085_t1400", 0.0085, 1400, 1),
        ("hc_a_s0085_t1500", 0.0085, 1500, 1),
        ("hc_b_s080_t520",   0.080,   520, 1),
        ("hc_b_s080_t580",   0.080,   580, 1),
        ("hc_c_s011_t2150",  0.011,  2150, 1),
        ("hc_c_s011_t2250",  0.011,  2250, 1),
        ("hc_d_s030_t2330",  0.030,  2330, 1),
        ("hc_d_s030_t2420",  0.030,  2420, 1),
        # soft_complete: many independent origins, recent enough that more
        # than one survives to the present.
        ("sc_a_s018_t650",   0.018,   650, 100),
        ("sc_a_s018_t720",   0.018,   720, 100),
        ("sc_b_s045_t380",   0.045,   380, 100),
        ("sc_b_s045_t450",   0.045,   450, 100),
        # hard_partial: one origin, still segregating at the present.
        ("hp_a_s016_t430",   0.016,   430, 1),
        ("hp_a_s016_t470",   0.016,   470, 1),
        ("hp_b_s035_t190",   0.035,   190, 1),
        ("hp_b_s035_t230",   0.035,   230, 1),
        # soft_partial: several origins AND incomplete -- the weakest cell,
        # so the origin count is moderate rather than 100 to leave a signal.
        ("sp_a_s014_t500",   0.014,   500, 40),
        ("sp_a_s014_t560",   0.014,   560, 40),
        ("sp_b_s025_t260",   0.025,   260, 40),
        ("sp_b_s025_t310",   0.025,   310, 40),
    ]

    # ---------------- v3 grid: the shipped 2x2 zoo ----------------
    # v2 showed two things that fixed the assignment of s to class.
    #
    # A weak sweep's completion time is far less predictable than a strong
    # one's: s=0.0085 from one copy completed anywhere between 93 and 460
    # generations ago from a fixed origin 1500 generations back, a spread of
    # 367, where s=0.080 held inside 45. So the YOUNGEST completion slot, the
    # one with the tightest absolute tolerance, goes to the strongest sweep,
    # and the weak coefficients go to the older slots where the tolerance is
    # hundreds of generations wide.
    #
    # s=0.0085 also completed (passed 0.98) in 8 of 8 while fixing in only 3:
    # the last few ancestral haplotypes take a long time to go. Completion,
    # not fixation, is both the graded quantity and the estimable one.
    GRID_V3 = [
        # hard_complete: four completion times spread over a factor of ~16,
        # with s deliberately NOT monotone in age.
        ("hc1_s080_t280",   0.080,  280, 1),
        ("hc2_s030_t710",   0.030,  710, 1),
        ("hc3_s0085_t2300", 0.0085, 2300, 1),
        ("hc4_s025_t2500",  0.025, 2500, 1),
        ("hc1_s080_t300",   0.080,  300, 1),
        ("hc2_s030_t760",   0.030,  760, 1),
        # soft_complete: many independent origins, recent enough that more
        # than one survives to the present.
        ("sc_a_s018_t650",  0.018,  650, 100),
        ("sc_b_s045_t380",  0.045,  380, 100),
        # hard_partial: one origin, still segregating.
        ("hp_a_s014_t450",  0.014,  450, 1),
        ("hp_b_s035_t210",  0.035,  210, 1),
        ("hp_b_s035_t230",  0.035,  230, 1),
        # soft_partial: the weakest cell. Twelve origins rather than a
        # hundred, so the surviving beneficial haplotypes are few enough to
        # leave a haplotype signal while still being more than one.
        ("sp_a_s011_t740",  0.011,  740, 12),
        ("sp_b_s060_t140",  0.060,  140, 12),
        ("sp_b_s060_t170",  0.060,  170, 12),
    ]
    GRID = {"v1": GRID_V1, "v2": GRID_V2, "v3": GRID_V3}[a.grid]
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
