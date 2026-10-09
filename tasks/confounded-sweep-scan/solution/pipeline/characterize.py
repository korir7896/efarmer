"""Classify each detected sweep and estimate its strength and age.

The decision rules and the two estimators are calibrated on independent
single-sweep replicates simulated under the same history (see
authoring/evidence/analyze_training.py for the measurements).  Each threshold
sits between well-separated measured values rather than being guessed.

Measured signal at the sweep, relative to that replicate's own background:

    class                 pi ratio   Tajima D    H12 ratio   H2/H1 ratio
    hard, recent, done       0.033      -3.03        11.7          0.02
    soft, recent, done       ~0.05      -3.0          ~11           0.78-1.05
    hard, old, done          0.098      -2.82         3.54          0.24
    partial                  0.233      -2.36         9.99          0.04

H2/H1 separates hard from soft by a factor of about forty: a hard sweep
collapses the region onto one haplotype, so the second-most-common haplotype
is rare, while a soft sweep leaves several frequent haplotypes and H2/H1 stays
near its neutral value.  Diversity separates partial from completed sweeps,
since a sweep still segregating never removed the variation on the
non-selected background.  H12 separates old from recent completed sweeps,
because recombination erodes the swept haplotype block.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- classification thresholds (see the table above) ---
H2H1_SOFT_MIN = 0.55        # between 0.24 (old hard) and 0.78 (softest soft)
PI_PARTIAL_MIN = 0.15       # between 0.098 (old hard) and 0.233 (partial)
H12_OLD_MAX = 6.0           # between 3.54 (old) and ~10 (recent)

# --- estimator calibration ---
# Trough width times local recombination rate equals this multiple of s,
# measured as 0.536, 0.571, 0.564 and 0.561 across the four classes.
WIDTH_RATE_OVER_S = 0.56
# t_hat = 2*N_cur*pi_min/pi_bg overestimates, by a factor that depends on how
# much flanking recovery the window averages in. Measured ratios of estimate
# to truth: 1.17 for old sweeps, 2.02 for recent hard, 2.46 for recent soft.
TIME_BIAS = {"hard_old_complete": 1.17,
             "hard_recent_complete": 2.02,
             "soft_standing": 2.46}

COMPLETE_CLASSES = ("hard_recent_complete", "hard_old_complete", "soft_standing")


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
    """Extreme value of each statistic within the region, against background."""
    idx = np.array(region["window_indices"], dtype=int)
    f: Dict[str, float] = {}

    def grab(stat, direction):
        v = obs[stat].astype(float)[idx]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan")
        return float(v.min() if direction == "low" else v.max())

    pi_x = grab("pi", "low")
    f["pi_ratio"] = pi_x / bg["pi"] if bg["pi"] else float("nan")
    f["pi_min"] = pi_x
    f["tajd_delta"] = grab("tajima_d", "low") - bg["tajima_d"]
    h12_x = grab("h12", "high")
    f["h12_ratio"] = h12_x / bg["h12"] if bg["h12"] else float("nan")
    h21_x = grab("h2_h1", "low")
    f["h2h1_ratio"] = h21_x / bg["h2_h1"] if bg["h2_h1"] else float("nan")
    f["nsl_frac"] = grab("nsl_frac", "high")
    return f


def classify(f: Dict[str, float]) -> str:
    h21 = f.get("h2h1_ratio", float("nan"))
    pir = f.get("pi_ratio", float("nan"))
    h12r = f.get("h12_ratio", float("nan"))

    # Softness first: a soft sweep can have a shallow diversity dip and would
    # otherwise be mistaken for a partial sweep.
    if np.isfinite(h21) and h21 >= H2H1_SOFT_MIN:
        return "soft_standing"
    if np.isfinite(pir) and pir >= PI_PARTIAL_MIN:
        return "partial"
    if np.isfinite(h12r) and h12r < H12_OLD_MAX:
        return "hard_old_complete"
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
    """Completion time from diversity recovery, with the measured bias removed."""
    if klass not in COMPLETE_CLASSES:
        return None
    pir = f.get("pi_ratio", float("nan"))
    if not np.isfinite(pir) or pir <= 0:
        return None
    raw = 2.0 * n_current * pir
    t = raw / TIME_BIAS.get(klass, 1.0)
    return float(np.clip(t, 1.0, 20_000.0))
