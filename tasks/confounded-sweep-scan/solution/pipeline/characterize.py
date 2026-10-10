"""Classify each detected sweep and estimate its strength and age.

Rules and estimators are calibrated on independent single-sweep replicates
simulated under the same history, grouped by what each sweep ACTUALLY turned
out to be rather than what it was asked to be (see
authoring/evidence/analyze_features.py). Each threshold sits in a measured
gap between classes.

Statistics read from the window containing the selected site, as ratios to
that replicate's background for diversity, H12 and H2/H1, and as differences
for Tajima's D and the extreme-nSL fraction:

    realized class          pi   Tajima D     H12    H2/H1     nSL     n
    hard, old, done      0.154     -2.365   1.346    0.797   0.030     8
    hard, recent, done   0.073     -2.645   8.455    0.071   0.044    11
    partial              0.528     -1.438   7.457    0.111   0.389     8
    soft                 0.743     +0.886   4.518    0.436   0.016     4

Two of these are cleanly separable and are what the rules use.

Tajima's D at a soft sweep is POSITIVE, from +0.21 to +1.15, while every
other class is negative with a maximum of -0.25. Several frequent haplotypes
leave an excess of intermediate-frequency variants, which is the textbook
soft-sweep signature and here separates with a gap of 0.46.

H12 separates completed hard sweeps by age, 0.90 to 2.83 for old against
5.79 to 13.32 for recent, a gap of 2.95, because recombination erodes the
swept haplotype block.

H2/H1 is deliberately NOT used to detect softness, although that is its
usual role. At an old hard sweep the block has decayed far enough that H2/H1
has recovered to 0.46-0.99, overlapping the 0.27-0.62 measured at soft
sweeps. Diversity separates partial from completed sweeps only partially,
0.24-0.85 against a hard-old maximum of 0.38, so it is applied after H12 has
already removed the old sweeps.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- classification thresholds, each in a measured gap ---
TAJD_SOFT_MIN = -0.20       # soft min +0.205, everything else max -0.252
H12_OLD_MAX = 3.10          # old max 2.834, next class min 3.458
PI_PARTIAL_MIN = 0.22       # partial min 0.241, hard-recent max 0.272

# How far around the detected peak to read the features, matching the window
# the calibration used.
FEATURE_RADIUS_BP = 60_000

# --- estimator calibration ---
# Trough width times local recombination rate equals this multiple of s,
# measured at 0.536, 0.571, 0.564 and 0.561 across the four classes.
WIDTH_RATE_OVER_S = 0.56

COMPLETE_CLASSES = ("hard_recent_complete", "hard_old_complete", "soft_standing")
# Completion time is estimated only for single-origin completed sweeps.
# Diversity recovery dates a sweep by how far below background the centre
# still sits, which assumes the sweep drove it to zero. A soft sweep never
# did -- its diversity ratio at the centre is 0.74 -- so the same estimator
# reads a soft sweep as roughly eighteen times older than it is.
TIMED_CLASSES = ("hard_recent_complete", "hard_old_complete")
TIME_BIAS = {"hard_old_complete": 1.17, "hard_recent_complete": 2.02}


def background_levels(obs: Dict[str, np.ndarray], regions: Sequence[Dict],
                      rate_bins: np.ndarray, exclude_bp: int = 1_200_000
                      ) -> Dict[int, Dict[str, float]]:
    """Median level of each statistic per rate bin, away from called regions."""
    masked = np.zeros(obs["pi"].size, dtype=bool)
    for r in regions:
        same = obs["chrom"] == r["chrom"]
        masked |= same & (np.abs(obs["mid"] - r["position"]) <= exclude_bp)
    out: Dict[int, Dict[str, float]] = {}
    for b in np.unique(rate_bins):
        sel = (rate_bins == b) & ~masked
        if sel.sum() < 30:
            sel = (rate_bins == b)
        lev = {}
        for stat in ("pi", "tajima_d", "h12", "h2_h1", "nsl_frac"):
            v = obs[stat].astype(float)[sel]
            v = v[np.isfinite(v)]
            lev[stat] = float(np.median(v)) if v.size else float("nan")
        out[int(b)] = lev
    return out


def features(obs: Dict[str, np.ndarray], region: Dict,
             bg: Dict[str, float]) -> Dict[str, float]:
    """Statistics around the detected peak, against background.

    Read in a narrow band around the peak rather than as an extreme over the
    whole region. Taking the extreme finds whichever window happens to sit at
    the edge of the footprint: the minimum H2/H1 over 1.5 Mb read 0.193 at
    genuine soft sweeps where the value at the sweep is 0.44, which inverted
    the classification.
    """
    peak = int(region["peak_index"])
    same = obs["chrom"] == region["chrom"]
    near = same & (np.abs(obs["mid"] - obs["mid"][peak]) <= FEATURE_RADIUS_BP)

    def at_peak(stat):
        v = obs[stat].astype(float)
        sel = near & np.isfinite(v)
        if not sel.any():
            return float(v[peak]) if np.isfinite(v[peak]) else float("nan")
        return float(np.median(v[sel]))

    f: Dict[str, float] = {}
    pi_at = at_peak("pi")
    f["pi_ratio"] = pi_at / bg["pi"] if bg["pi"] else float("nan")
    f["tajd_delta"] = at_peak("tajima_d") - bg["tajima_d"]
    h12_at = at_peak("h12")
    f["h12_ratio"] = h12_at / bg["h12"] if bg["h12"] else float("nan")
    h21_at = at_peak("h2_h1")
    f["h2h1_ratio"] = h21_at / bg["h2_h1"] if bg["h2_h1"] else float("nan")
    f["nsl_delta"] = at_peak("nsl_frac") - bg["nsl_frac"]

    # The deepest diversity in the whole region, used for dating rather than
    # for classification.
    v = obs["pi"].astype(float)
    reg = np.array(region["window_indices"], dtype=int)
    vv = v[reg][np.isfinite(v[reg])]
    f["pi_min_ratio"] = (float(vv.min()) / bg["pi"]) if (vv.size and bg["pi"]) else float("nan")
    return f


def classify(f: Dict[str, float]) -> str:
    tajd = f.get("tajd_delta", float("nan"))
    h12r = f.get("h12_ratio", float("nan"))
    pir = f.get("pi_ratio", float("nan"))

    # Softness first, on the one statistic that separates it cleanly.
    if np.isfinite(tajd) and tajd >= TAJD_SOFT_MIN:
        return "soft_standing"
    # Then age among completed hard sweeps: an eroded haplotype block.
    if np.isfinite(h12r) and h12r < H12_OLD_MAX:
        return "hard_old_complete"
    # Then whether the sweep finished, from residual diversity.
    if np.isfinite(pir) and pir >= PI_PARTIAL_MIN:
        return "partial"
    return "hard_recent_complete"


def estimate_s(obs: Dict[str, np.ndarray], region: Dict, bg_pi: float,
               local_rate: float, search_bp: int = 2_500_000) -> float:
    """From the width of the diversity trough: s ~ (w * r) / 0.56."""
    same = obs["chrom"] == region["chrom"]
    near = same & (np.abs(obs["mid"] - region["position"]) <= search_bp)
    v = obs["pi"].astype(float)
    sel = near & np.isfinite(v) & (v < 0.5 * bg_pi)
    if not sel.any() or not np.isfinite(local_rate) or local_rate <= 0:
        return float("nan")
    mids = obs["mid"][sel]
    width = float(mids.max() - mids.min())
    if width <= 0:
        width = float(np.median(np.diff(np.sort(obs["mid"][same]))))
    s = width * local_rate / WIDTH_RATE_OVER_S
    return float(np.clip(s, 1e-4, 0.5))


def estimate_time(f: Dict[str, float], klass: str, n_current: float
                  ) -> Optional[float]:
    """Completion time from diversity recovery, with the measured bias removed.

    Returns None for classes where the estimator does not apply: a sweep
    still segregating has no completion time, and a soft sweep never drove
    diversity to zero so the recovery clock has no zero to start from.
    """
    if klass not in TIMED_CLASSES:
        return None
    pir = f.get("pi_min_ratio", float("nan"))
    if not np.isfinite(pir) or pir <= 0:
        return None
    t = (2.0 * n_current * pir) / TIME_BIAS.get(klass, 1.0)
    return float(np.clip(t, 1.0, 20_000.0))
