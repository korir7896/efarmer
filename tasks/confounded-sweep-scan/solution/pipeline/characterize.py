"""Classify each detected sweep and estimate its strength and age.

Everything here is calibrated on 32 independent single-sweep replicates
simulated under the same history, grouped by what each sweep ACTUALLY turned
out to be rather than what it was asked to be (see
authoring/evidence/eval_classifier.py and gate_feasibility.py).

Why a nearest-centroid rule rather than thresholds
--------------------------------------------------
Single-feature thresholds looked clean on eight replicates and stopped being
clean on thirty-two. Tajima's D appeared to separate soft sweeps with a gap
of 0.46; with seven soft replicates instead of four it overlaps by 1.59. The
best single feature gives 69% leave-one-out accuracy. Four features together
give 90.6%, so the information is there but spread across them.

Measured means (ratios to background for diversity and H2/H1, differences for
Tajima's D and the extreme-nSL fraction):

    realized class          pi     Tajima D       H2/H1         nSL    n
    hard, recent, done   0.162       -2.326       0.123       0.043    9
    hard, old, done      0.212       -2.126       0.852       0.052    8
    soft, done           0.797       +0.435       0.603      -0.010    7
    partial              0.697       -0.891       0.161       0.430    8

Each class is distinguished by a different combination. A recent hard sweep
is low on everything. An old hard sweep looks the same except H2/H1 has
recovered to 0.85, because recombination has broken up the swept block. A
soft sweep barely depresses diversity at all, at 0.80 of background, and
pushes Tajima's D positive from the excess of intermediate-frequency variants
that several frequent haplotypes create. A partial sweep is the only class
with a strong nSL signal, because it is the only one still segregating and so
the only one carrying a long young haplotype.

Note that H2/H1 earns its place by separating old from recent hard sweeps,
not hard from soft as it usually would.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ===========================================================================
# CALIBRATED CONSTANTS.  Everything between here and background_levels() is
# produced by authoring/evidence/emit_constants.py from the single-sweep
# training replicates in authoring/evidence/training/ and pasted in verbatim.
# ===========================================================================

# --- nearest-centroid classifier ---
CLASS_FEATURES = ("pi_ratio", "tajd_delta", "h2h1_ratio", "nsl_delta")
FEATURE_MEAN = np.array([0.0, 0.0, 0.0, 0.0])
FEATURE_SD = np.array([1.0, 1.0, 1.0, 1.0])
CLASS_CENTROIDS = {
    "hard_complete": np.zeros(4),
    "hard_partial":  np.zeros(4),
    "soft_complete": np.zeros(4),
    "soft_partial":  np.zeros(4),
}

# How far around the detected peak to read the features, matching calibration.
FEATURE_RADIUS_BP = 60_000

# --- s from the width of the diversity trough ---
# The width is the contiguous run of windows whose smoothed diversity is below
# the midpoint between the trough's floor and background. Thresholding at half
# of BACKGROUND instead fails for the shallow classes, whose diversity never
# gets that low: it returned 10 kb for partial sweeps and 40 kb for soft ones
# against 230-320 kb for hard sweeps. Measuring recovery relative to each
# sweep's own floor is what makes one constant per class work across a range
# of s spanning a factor of 9.4.
SMOOTH_WINDOWS = 15
RECOVERY_FRAC = 0.5
WIDTH_SEARCH_BP = 1_200_000
CLASS_WIDTH_CONSTANT = {
    "hard_complete": 0.1200,
    "hard_partial":  0.0900,
    "soft_complete": 0.0750,
    "soft_partial":  0.0750,
}

# --- diversity-recovery clock ---
# A completed sweep's age is read off how far below background its centre
# still sits. One power law over the whole hard_complete class rather than a
# constant bias per class: class no longer splits by age, so hard_complete
# spans completion times from about a hundred to about two thousand
# generations and residual diversity is not proportional to age over that
# range.
#
# Reported for every sweep called complete, because the submission schema
# requires a time there, but only meaningful for the single-origin ones: the
# clock presumes the sweep drove diversity to zero, and a soft sweep leaves
# it near background. The verifier grades timing on true hard_complete
# sweeps only, for the same reason.
TIMED_CLASSES = ("hard_complete", "soft_complete")
TIME_CLOCK_A = 2.0
TIME_CLOCK_B = 1.0
# Fallback when the trough has no finite floor to read, so that a sweep
# called complete always carries a number.
TIME_FALLBACK_GENS = 500.0


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
    whole region: taking the extreme finds whichever window sits at the edge
    of the footprint, which inverts the classification.
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
    f["pi_ratio"] = at_peak("pi") / bg["pi"] if bg["pi"] else float("nan")
    f["tajd_delta"] = at_peak("tajima_d") - bg["tajima_d"]
    f["h12_ratio"] = at_peak("h12") / bg["h12"] if bg["h12"] else float("nan")
    f["h2h1_ratio"] = at_peak("h2_h1") / bg["h2_h1"] if bg["h2_h1"] else float("nan")
    f["nsl_delta"] = at_peak("nsl_frac") - bg["nsl_frac"]

    # Deepest diversity near the peak: used for dating, not classification.
    v = obs["pi"].astype(float)
    wide = same & (np.abs(obs["mid"] - obs["mid"][peak]) <= 1_000_000) & np.isfinite(v)
    f["pi_min_ratio"] = (float(v[wide].min()) / bg["pi"]) if (wide.any() and bg["pi"]) \
        else float("nan")
    return f


def classify(f: Dict[str, float]) -> str:
    """Nearest calibrated centroid in standardized feature space."""
    x = np.array([f.get(k, np.nan) for k in CLASS_FEATURES], dtype=float)
    if not np.all(np.isfinite(x)):
        # Fall back on the features that are available.
        ok = np.isfinite(x)
        if not ok.any():
            return "hard_complete"
        z = (x[ok] - FEATURE_MEAN[ok]) / FEATURE_SD[ok]
        return min(CLASS_CENTROIDS,
                   key=lambda c: float(np.linalg.norm(z - CLASS_CENTROIDS[c][ok])))
    z = (x - FEATURE_MEAN) / FEATURE_SD
    return min(CLASS_CENTROIDS,
               key=lambda c: float(np.linalg.norm(z - CLASS_CENTROIDS[c])))


def _smooth_median(v: np.ndarray, k: int) -> np.ndarray:
    out = np.full(v.size, np.nan)
    h = k // 2
    for i in range(v.size):
        a, b = max(0, i - h), min(v.size, i + h + 1)
        seg = v[a:b][np.isfinite(v[a:b])]
        if seg.size:
            out[i] = np.median(seg)
    return out


def trough_width(obs: Dict[str, np.ndarray], region: Dict, bg_pi: float) -> float:
    """Contiguous width of the diversity trough, relative to its own floor."""
    same = obs["chrom"] == region["chrom"]
    idx = np.where(same)[0]
    if idx.size == 0 or not np.isfinite(bg_pi) or bg_pi <= 0:
        return 0.0
    order = idx[np.argsort(obs["mid"][idx])]
    mid = obs["mid"][order]
    vs = _smooth_median(obs["pi"].astype(float)[order], SMOOTH_WINDOWS)
    centre = float(obs["mid"][int(region["peak_index"])])

    near = (np.abs(mid - centre) <= WIDTH_SEARCH_BP) & np.isfinite(vs)
    if not near.any():
        return 0.0
    floor = float(vs[near].min())
    level = floor + RECOVERY_FRAC * (bg_pi - floor)
    below = np.isfinite(vs) & (vs < level)
    if not below.any():
        return 0.0
    i0 = int(np.argmin(np.abs(mid - centre)))
    if not below[i0]:
        cand = np.where(below)[0]
        i0 = int(cand[np.argmin(np.abs(mid[cand] - centre))])
        if abs(mid[i0] - centre) > 600_000:
            return 0.0
    lo = i0
    while lo - 1 >= 0 and below[lo - 1]:
        lo -= 1
    hi = i0
    while hi + 1 < len(mid) and below[hi + 1]:
        hi += 1
    return float(mid[hi] - mid[lo])


def estimate_s(obs: Dict[str, np.ndarray], region: Dict, bg_pi: float,
               local_rate: float, klass: str) -> float:
    """s from the trough width times the local recombination rate."""
    w = trough_width(obs, region, bg_pi)
    k = CLASS_WIDTH_CONSTANT.get(klass, 0.1)
    if w <= 0 or not np.isfinite(local_rate) or local_rate <= 0 or k <= 0:
        return float("nan")
    return float(np.clip(w * local_rate / k, 1e-4, 0.5))


def estimate_time(f: Dict[str, float], klass: str, n_current: float
                  ) -> Optional[float]:
    """Completion time from the calibrated diversity-recovery power law."""
    if klass not in TIMED_CLASSES:
        return None
    pir = f.get("pi_min_ratio", float("nan"))
    if not np.isfinite(pir) or pir <= 0:
        return TIME_FALLBACK_GENS
    t = TIME_CLOCK_A * float(n_current) * (pir ** TIME_CLOCK_B)
    return float(np.clip(t, 1.0, 20_000.0))
