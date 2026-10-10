#!/usr/bin/env python3
"""Detection score at each true sweep, against the calling threshold.

Says exactly which sweeps the scan can and cannot see, and by how much, so a
recall failure can be attributed to a specific class rather than guessed at.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(TASK, "solution"))

from pipeline import demography as DEM
from pipeline import detect as DET
from pipeline import sweepstats as SS

WIN, STEP, MU = 50_000, 10_000, 2.0e-8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(TASK, "environment", "data"))
    ap.add_argument("--truth", default=os.path.join(TASK, "tests", "truth.json"))
    ap.add_argument("--n-null", type=int, default=10)
    a = ap.parse_args()

    truth = json.load(open(a.truth))
    chroms = SS.load_vcf(os.path.join(a.data, "variants.vcf.gz"))
    rm = DET.load_rate_map(os.path.join(a.data, "recombination_map.bed"))
    med = rm.median_rate()
    n_dip = next(iter(chroms.values())).n_hap // 2

    obs = DET.scan_genome(chroms, rm.chrom_len, rm, WIN, STEP)
    obs_sum = DEM.observed_summaries(chroms, rm.chrom_len)
    hist, _ = DEM.fit_demography(obs_sum, mu=MU, rate=med, n_dip=n_dip,
                                 seed=20_240_101, log=lambda m: None)
    null = DET.build_null(hist, rm, MU, n_dip, WIN, STEP, n_reps=a.n_null,
                          seed=20_240_101 + 5000, log=lambda m: None)
    score = DET.sweep_scores(obs, null, med)
    thr = DET.null_score_threshold(null, med, a.n_null)
    print(f"calling threshold {thr:.2f}\n")
    print(f"{'sweep':<7}{'class':<22}{'chrom':>6}{'position':>11}"
          f"{'best score':>12}{'offset kb':>11}  seen?")
    for s in truth["sweeps"]:
        same = obs["chrom"] == s["chrom"]
        near = same & (np.abs(obs["mid"] - s["position_local"]) <= 400_000)
        if not near.any():
            print(f"{s['sweep_id']:<7}{s['class']:<22}{s['chrom']:>6}"
                  f"{s['position_local']:>11,}{'-':>12}{'-':>11}  no windows")
            continue
        idx = np.where(near)[0]
        best = idx[np.argmax(np.where(np.isfinite(score[idx]), score[idx], -np.inf))]
        off = (obs["mid"][best] - s["position_local"]) / 1000.0
        print(f"{s['sweep_id']:<7}{s['class']:<22}{s['chrom']:>6}"
              f"{s['position_local']:>11,}{score[best]:>12.1f}{off:>11.0f}"
              f"  {'yes' if score[best] >= thr else 'NO'}")

    fin = score[np.isfinite(score)]
    print(f"\ngenome-wide score: median {np.median(fin):.1f}, "
          f"p99 {np.percentile(fin, 99):.1f}, max {fin.max():.1f}")


if __name__ == "__main__":
    main()
