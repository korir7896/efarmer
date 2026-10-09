#!/usr/bin/env python3
"""What the scan statistics do with no selection anywhere."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "provenance"))
import config as C

d = np.load(os.path.join(HERE, "null_calibration.npz"))
cold = d["cold"].astype(bool)
nearcold = d["nearcold"].astype(bool)
plain = ~nearcold

print(f"windows: {cold.size} total, {cold.sum()} in cold spots, {plain.sum()} away from them")
print(f"replicates: {d['rep'].max()+1}\n")

STATS = [("pi", "low"), ("tajima_d", "low"), ("h12", "high"),
         ("h2_h1", "low"), ("nsl_frac", "high")]

print(f"{'stat':<10}{'dir':>5} | {'median':>10}{'p1':>10}{'p99':>10}{'p99.9':>10} | "
      f"{'cold med':>10}{'cold p1':>10}{'cold p99':>10}")
for name, direction in STATS:
    v = d[name].astype(float)
    ok = np.isfinite(v)
    a = v[ok & plain]; c = v[ok & cold]
    if a.size == 0 or c.size == 0:
        print(f"{name:<10} no data"); continue
    print(f"{name:<10}{direction:>5} | {np.median(a):>10.4g}{np.percentile(a,1):>10.4g}"
          f"{np.percentile(a,99):>10.4g}{np.percentile(a,99.9):>10.4g} | "
          f"{np.median(c):>10.4g}{np.percentile(c,1):>10.4g}{np.percentile(c,99):>10.4g}")

print("\n--- false positives from a genome-wide threshold ---")
print("A naive scan takes the genome-wide 99.9th percentile as its cutoff.")
print("These are the fractions of windows that cross it, inside cold spots")
print("versus away from them. A ratio far above 1 is the confound.\n")
print(f"{'stat':<10}{'thresh':>12}{'frac outside':>14}{'frac in cold':>14}{'enrichment':>12}")
for name, direction in STATS:
    v = d[name].astype(float)
    ok = np.isfinite(v)
    a = v[ok & plain]; c = v[ok & cold]
    if a.size == 0 or c.size == 0: continue
    if direction == "low":
        thr = np.percentile(a, 0.1)
        fo = float(np.mean(a <= thr)); fc = float(np.mean(c <= thr))
    else:
        thr = np.percentile(a, 99.9)
        fo = float(np.mean(a >= thr)); fc = float(np.mean(c >= thr))
    enr = (fc / fo) if fo > 0 else float("inf")
    print(f"{name:<10}{thr:>12.4g}{fo:>14.4f}{fc:>14.4f}{enr:>12.1f}x")

print("\n--- SNP density ---")
ns = d["n_sites"].astype(float)
print(f"median SNPs per 50kb window: away from cold spots {np.median(ns[plain]):.0f}, "
      f"in cold spots {np.median(ns[cold]):.0f}")
