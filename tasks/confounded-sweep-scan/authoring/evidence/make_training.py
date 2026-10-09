#!/usr/bin/env python3
"""Single-sweep training replicates, one chromosome each.

Produces the per-class feature distributions used to:
  * set the classification rules in the reference solution,
  * calibrate the estimators of s and of fixation time,
  * measure localisation spread, which is what sets the verifier's match
    window (the window must come from how precisely correct methods can
    locate a sweep, not from how precise the reference happens to be).

Each replicate is one 12 Mb chromosome at uniform recombination rate with
exactly one sweep, so the features describe the sweep alone. Cold spots are
covered separately by calibrate_null.py.
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "provenance"))

import config as C
import sweepstats as SS
from export import present_day_individuals, scrub_metadata, subsample
from forward import SweepPlan, run_forward
from genome import GenomeConfig
from nullsim import ts_to_chromdata
from recap import overlay_mutations, recapitate
from sweeps import realized_class

WIN, STEP = 50_000, 10_000
CHROM_LEN = C.CHROM_LEN
SWEEP_POS = 6_143_879          # a non-round position near the middle

# (label, s, origin_gens_ago, ndescendants)
CLASS_CONFIGS = {
    "hard_recent_complete": (0.035, 750, 1),
    "hard_old_complete":    (0.028, 2150, 1),
    "soft_standing":        (0.032, 780, 18),
    "partial":              (0.018, 430, 1),
}


def one_replicate(label, seed):
    s, origin, ndesc = CLASS_CONFIGS[label]
    cfg = GenomeConfig(n_chrom=1, chrom_len=CHROM_LEN,
                       base_rec_rate=C.BASE_REC_RATE, mut_rate=C.MUT_RATE,
                       cold_spots=[])
    demog = C.demography()
    plans = [SweepPlan("sw", SWEEP_POS, s, 1.0, origin, ndesc, label)]
    ts, rec, _ = run_forward(cfg, C.N_CURRENT, C.T_FORWARD, plans,
                             seed=seed, verbose=False)
    r = rec.real["sw"]
    rts = recapitate(ts, cfg, demog, seed=seed + 1)
    mts = overlay_mutations(rts, cfg, seed=seed + 2)
    sts, _ = subsample(mts, C.N_SAMPLE_DIPLOIDS, seed=seed + 3)
    sts = scrub_metadata(sts)
    cds = ts_to_chromdata(sts, cfg)
    cd = cds["chr1"]

    sfs = SS.windowed_sfs_stats(cd, cfg.chrom_len, WIN, STEP)
    gar = SS.garud_scan_bp(cd, cfg.chrom_len, WIN, STEP, min_snps=20)
    ns = SS.nsl_scan(cd)
    nsw = SS.windowed_max_abs(ns["pos"], ns["std"], cfg.chrom_len, WIN, STEP,
                              kind="frac_extreme")

    mids = sfs["mid"]
    far = np.abs(mids - SWEEP_POS) > 2_500_000     # background, outside footprint
    out = {
        "label": label,
        "seed": seed,
        "realized_class": realized_class(r),
        "fixed": bool(r.fixed),
        "fixation_gens_ago": r.fixation_gens_ago if r.fixed else None,
        "final_freq": float(r.final_freq),
        "s_true": float(r.s),
        "origin_gens_ago": int(r.origin_gens_ago),
        "sweep_pos": SWEEP_POS,
        "mid": mids,
        "pi": sfs["pi"], "tajima_d": sfs["tajima_d"],
        "h12": gar["h12"], "h2_h1": gar["h2_h1"],
        "nsl_frac": nsw["value"], "n_snps": gar["n_snps"],
        "bg_pi": float(np.nanmedian(sfs["pi"][far])),
        "bg_tajd": float(np.nanmedian(sfs["tajima_d"][far])),
        "bg_h12": float(np.nanmedian(gar["h12"][far])),
        "bg_h2h1": float(np.nanmedian(gar["h2_h1"][far])),
        "bg_nsl": float(np.nanmedian(nsw["value"][far])),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=sorted(CLASS_CONFIGS))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=500_000)
    ap.add_argument("--outdir", default=os.path.join(HERE, "training"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for i in range(a.n):
        seed = a.seed0 + 97 * i
        try:
            r = one_replicate(a.label, seed)
        except Exception as e:
            print(f"{a.label} seed {seed}: FAILED {e}", flush=True)
            continue
        path = os.path.join(a.outdir, f"{a.label}_{seed}.npz")
        np.savez_compressed(path, **{k: v for k, v in r.items() if v is not None})
        print(f"{a.label} seed {seed}: realized={r['realized_class']} "
              f"fix_ago={r['fixation_gens_ago']} freq={r['final_freq']:.3f} "
              f"-> {os.path.basename(path)}", flush=True)


if __name__ == "__main__":
    main()
