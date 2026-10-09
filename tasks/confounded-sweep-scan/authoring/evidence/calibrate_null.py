#!/usr/bin/env python3
"""Null distribution of every scan statistic, with no selection anywhere.

Any window that looks like a sweep in these replicates is a false positive by
construction.  The cold-spot columns are the point of the exercise: they show
what a scan does when it applies a genome-wide threshold to a region whose
recombination rate is an order of magnitude below average.
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "provenance"))

import config as C
import sweepstats as SS
from nullsim import simulate_null, ts_to_chromdata

WIN = 50_000
STEP = 10_000


def coldspot_mask(cfg, chrom_idx, mids):
    m = np.zeros(mids.size, dtype=bool)
    for spot in cfg.cold_spots:
        if spot.chrom != chrom_idx:
            continue
        m |= (mids >= spot.start) & (mids < spot.end)
    return m


def near_coldspot_mask(cfg, chrom_idx, mids, pad):
    m = np.zeros(mids.size, dtype=bool)
    for spot in cfg.cold_spots:
        if spot.chrom != chrom_idx:
            continue
        m |= (mids >= spot.start - pad) & (mids < spot.end + pad)
    return m


def run_replicate(cfg, demog, seed):
    ts = simulate_null(cfg, demog, C.N_SAMPLE_DIPLOIDS, seed)
    cds = ts_to_chromdata(ts, cfg)
    rows = {k: [] for k in ("pi", "tajima_d", "h12", "h2_h1", "nsl_frac",
                            "n_sites", "cold", "nearcold", "chrom", "mid")}
    for ci in range(cfg.n_chrom):
        name = f"chr{ci + 1}"
        if name not in cds:
            continue
        cd = cds[name]
        sfs = SS.windowed_sfs_stats(cd, cfg.chrom_len, WIN, STEP)
        gar = SS.garud_scan_bp(cd, cfg.chrom_len, WIN, STEP, min_snps=20)
        ns = SS.nsl_scan(cd)
        nsw = SS.windowed_max_abs(ns["pos"], ns["std"], cfg.chrom_len, WIN,
                                  STEP, kind="frac_extreme")
        mids = sfs["mid"]
        rows["pi"].append(sfs["pi"]); rows["tajima_d"].append(sfs["tajima_d"])
        rows["n_sites"].append(sfs["n_sites"])
        rows["h12"].append(gar["h12"]); rows["h2_h1"].append(gar["h2_h1"])
        rows["nsl_frac"].append(nsw["value"])
        rows["cold"].append(coldspot_mask(cfg, ci, mids))
        rows["nearcold"].append(near_coldspot_mask(cfg, ci, mids, 300_000))
        rows["chrom"].append(np.full(mids.size, ci + 1))
        rows["mid"].append(mids)
    return {k: np.concatenate(v) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "null_calibration.npz"))
    a = ap.parse_args()
    cfg = C.genome_config(); demog = C.demography()
    acc = None
    for i in range(a.reps):
        r = run_replicate(cfg, demog, seed=7_000_000 + 1013 * i)
        r["rep"] = np.full(r["pi"].size, i)
        acc = r if acc is None else {k: np.concatenate([acc[k], r[k]]) for k in r}
        print(f"replicate {i} done: {r['pi'].size} windows", flush=True)
    np.savez_compressed(a.out, **acc)
    print("saved", a.out)


if __name__ == "__main__":
    main()
