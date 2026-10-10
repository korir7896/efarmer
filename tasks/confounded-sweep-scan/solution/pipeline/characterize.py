"""Classify each detected sweep and estimate its strength and age.

Everything here is calibrated on 51 independent single-sweep replicates
simulated under the same history at several strengths per class
(authoring/evidence/run_training.sh), grouped by what each sweep ACTUALLY
turned out to be rather than what it was asked to be. The feature sets and
the fits are chosen by held-out score in authoring/evidence/fit_classifier.py
and authoring/evidence/fit_estimators.py, not assumed.

Scores are leave-one-CONFIGURATION-out throughout. The shipped sweeps sit at
the very (s, age) configurations the calibration replicates were simulated
at, so holding out one REPLICATE at a time measures almost nothing: a rule
that had memorised those few pairs would score perfectly and generalise not
at all.

Measured means at the sweep, against each replicate's own background
(diversity, H12 and H2/H1 as ratios, Tajima's D and the extreme-nSL fraction
as differences):

    realized class     diversity   Tajima D    H12   H2/H1     nSL    n
    hard_complete          0.119      -2.44  7.409   0.084   +0.01   16
    hard_partial           0.801      -0.63  6.596   0.104   +0.41    9
    soft_complete          0.794      +0.53  4.298   0.673   -0.02   16
    soft_partial           0.811      -0.03  4.949   0.549   +0.20   10

Neither axis of the class scheme has a statistic that carries it alone.
H2/H1, the textbook hard-versus-soft discriminator, runs from 0.009 to 1.012
of background across the hard_complete replicates, swallowing the 0.025 to
0.861 the soft completed ones occupy: at an old hard sweep recombination has
broken up the swept block and H2/H1 has recovered. Diversity, the obvious
complete-versus-partial discriminator, runs from 0.011 to 0.719 across
hard_complete and from 0.383 to 1.167 across hard_partial, so it overlaps
too. The medians above separate; the distributions do not. What works is one
rule per axis, with softness fitted separately inside each completeness
group, where H2/H1 is usable again because every partial sweep is young.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ===========================================================================
# CALIBRATED CONSTANTS.  Everything between here and background_levels() is
# produced by authoring/evidence/emit_constants.py from the single-sweep
# training replicates in authoring/evidence/training/ and pasted in verbatim.
# ===========================================================================

# --- classifier: the two axes of the class scheme, one rule each ---
# A single nearest-centroid rule over all four classes at once scores 72.9%
# held out. The scheme is two independent binary properties, so two binary
# rules do better: 82.4%, measured leave-one-CONFIGURATION-out, which is the
# honest test here because the shipped sweeps sit at the very configurations
# the calibration replicates were simulated at and a rule that memorised
# those few (s, age) pairs would otherwise score perfectly.
#
# Softness is fitted separately inside each completeness group, because the
# axis is far easier among completed sweeps (90.6%) than among the ones still
# rising (73.7%): the frequent haplotypes a completed soft sweep leaves have
# had time to become frequent, while a partial sweep's are still a minority
# of the sample however many of them there are.
#
# What carries each axis:
#   completeness -- diversity and the number of segregating sites in the
#     window. A completed sweep removed variation; one still rising did not.
#   softness, completed -- Tajima's D, which several frequent haplotypes push
#     positive where one pushes it negative, with the extent of the H12
#     elevation controlling for the sweep's strength.
#   softness, partial -- H2/H1, the textbook hard-versus-soft discriminator,
#     which is usable here only because age is not in play: every partial
#     sweep is still rising. Pooled over completed sweeps of all ages it is
#     confounded, which is why it does not appear in the rule above.
CLASS_RULES = {
    "complete": dict(
        feats=["pi_ratio", "snp_ratio"],
        mean=[0.553882, 0.648399],
        sd=[0.3673, 0.27194],
        weights=[1.98594, 2.3216, -6.92335]),           # n=51, held out 0.961
    "soft_given_complete": dict(
        feats=["tajd_delta", "w_h12_2"],
        mean=[-1.09206, 1.08781e+06],
        sd=[1.60093, 1.02584e+06],
        weights=[0.414562, 3.17302, 0.715223]),         # n=32, held out 0.906
    "soft_given_partial": dict(
        feats=["h2h1_ratio"],
        mean=[0.378404],
        sd=[0.277096],
        weights=[0.142307, 0.894801]),                  # n=19, held out 0.737
}

# How far around the detected peak to read the features, matching calibration.
FEATURE_RADIUS_BP = 60_000

# --- s and age from the shape of the footprint ---
# Both estimators are log-linear fits in two measured features, chosen by
# held-out score in authoring/evidence/fit_estimators.py rather than assumed.
#
# The first version of this used one relation for s, trough width times the
# local recombination rate over a per-class constant, which is the textbook
# one: the swept region extends to roughly s/r. It put 74% of calibration
# replicates inside a factor of 2.5 but ranked them at only 0.32, because a
# soft sweep leaves diversity near background and has no trough whose width
# can be measured. Adding a second feature, and taking the swept extent from
# the H12 elevation rather than the diversity dip where diversity is flat,
# brings that to 98% and 0.83.
#
# Width features are scaled by the local recombination rate: what carries the
# strength is the GENETIC length of the footprint, so a sweep sitting in a
# cold spot leaves a physically wider one at the same s. The calibration
# genome is at the base rate throughout, so the scaling is 1 there and the
# coefficients are unchanged by it.
SMOOTH_WINDOWS = 15
RECOVERY_FRAC = 0.5
WIDTH_SEARCH_BP = 1_200_000
BASE_RATE = 1e-8

S_MODEL = {
    "hard_complete": dict(feats=["pi_ratio", "w_pi_50"],
                          coef=[-9.0909, -0.34032, 0.36376]),
    "hard_partial":  dict(feats=["pi_ratio", "w_h12_15"],
                          coef=[-10.386, 0.38234, 0.48695]),
    "soft_complete": dict(feats=["pi_ratio", "w_pi_75"],
                          coef=[-14.828, 0.34891, 0.88015]),
    "soft_partial":  dict(feats=["pi_ratio", "w_h12_2"],
                          coef=[-15.845, 0.084123, 0.88815]),
}
# Per-class median s, used only when a feature the model needs is missing --
# a footprint with no measurable extent at all.
S_FALLBACK = {"hard_complete": 0.026, "hard_partial": 0.022,
              "soft_complete": 0.028, "soft_partial": 0.025}

# --- diversity-recovery clock ---
# Age comes from H12 decay with the footprint extent controlling for
# strength, not from residual diversity alone. H12 at a completed sweep falls
# back towards background as recombination breaks up the swept block, which
# is a cleaner clock than the depth of the diversity trough: measured
# held-out across configurations, 15 of 16 completion times land inside the
# verifier's tolerance against 9 of 16 for the diversity-depth version.
#
# Reported for every sweep called complete, because the submission schema
# requires a number there, but only meaningful for the single-origin ones:
# the verifier grades timing on true hard_complete sweeps only.
TIMED_CLASSES = ("hard_complete", "soft_complete")
T_MODEL = dict(feats=["h12_ratio", "w_pi_75"], coef=[5.2993, -0.92298, 0.17419])
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
        for stat in ("pi", "tajima_d", "h12", "h2_h1", "nsl_frac",
                     "n_snps"):
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
    f["snp_ratio"] = (at_peak("n_snps") / bg["n_snps"]
                      if bg.get("n_snps") else float("nan"))

    # Deepest diversity near the peak: used for dating, not classification.
    v = obs["pi"].astype(float)
    wide = same & (np.abs(obs["mid"] - obs["mid"][peak]) <= 1_000_000) & np.isfinite(v)
    f["pi_min_ratio"] = (float(v[wide].min()) / bg["pi"]) if (wide.any() and bg["pi"]) \
        else float("nan")
    return f


def _logistic(rule: Dict, feat: Dict[str, float]) -> Optional[float]:
    """P(label) under a standardized logistic rule, or None if a feature is
    missing. A width of zero is a measurement, not a gap, so only a
    non-finite value counts as missing."""
    xs = []
    for k, mu, sd in zip(rule["feats"], rule["mean"], rule["sd"]):
        x = feat.get(k, float("nan"))
        if not np.isfinite(x):
            return None
        xs.append((float(x) - mu) / sd)
    w = rule["weights"]
    z = float(w[0]) + sum(float(b) * v for b, v in zip(w[1:], xs))
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


def classify(f: Dict[str, float]) -> str:
    """Completeness first, then softness with the rule for that group.

    Falls back on the commoner label for an axis whose features are missing,
    which is better than refusing to answer: a call with no class at all
    fails the schema, and the commoner label is right more often than not.
    """
    p_complete = _logistic(CLASS_RULES["complete"], f)
    complete = True if p_complete is None else p_complete >= 0.5
    rule = CLASS_RULES["soft_given_complete" if complete
                       else "soft_given_partial"]
    p_soft = _logistic(rule, f)
    soft = False if p_soft is None else p_soft >= 0.5
    return ("soft_" if soft else "hard_") + ("complete" if complete
                                             else "partial")


def _smooth_median(v: np.ndarray, k: int) -> np.ndarray:
    out = np.full(v.size, np.nan)
    h = k // 2
    for i in range(v.size):
        a, b = max(0, i - h), min(v.size, i + h + 1)
        seg = v[a:b][np.isfinite(v[a:b])]
        if seg.size:
            out[i] = np.median(seg)
    return out


def _run_around(mid: np.ndarray, flag: np.ndarray, centre: float) -> float:
    """Span of the contiguous run of flagged windows containing the centre."""
    if not flag.any():
        return 0.0
    i0 = int(np.argmin(np.abs(mid - centre)))
    if not flag[i0]:
        cand = np.where(flag)[0]
        i0 = int(cand[np.argmin(np.abs(mid[cand] - centre))])
        if abs(mid[i0] - centre) > 600_000:
            return 0.0
    lo = hi = i0
    while lo - 1 >= 0 and flag[lo - 1]:
        lo -= 1
    while hi + 1 < len(mid) and flag[hi + 1]:
        hi += 1
    return float(mid[hi] - mid[lo])


def contiguous_low_width(mid: np.ndarray, v: np.ndarray, bg: float,
                         centre: float, frac: float) -> float:
    """Width of the contiguous dip around the centre, relative to its floor.

    The level is floor + frac*(background - floor), not frac*background:
    thresholding on a fraction of BACKGROUND fails for the shallow classes,
    whose diversity never gets that low. It returned 10 kb for partial
    sweeps and 40 kb for soft ones against 230-320 kb for hard sweeps.
    """
    order = np.argsort(mid)
    m = mid[order]
    vs = _smooth_median(v[order].astype(float), SMOOTH_WINDOWS)
    near = (np.abs(m - centre) <= WIDTH_SEARCH_BP) & np.isfinite(vs)
    if not near.any() or not np.isfinite(bg):
        return 0.0
    floor = float(vs[near].min())
    level = floor + frac * (bg - floor)
    return _run_around(m, np.isfinite(vs) & (vs < level), centre)


def contiguous_high_width(mid: np.ndarray, v: np.ndarray, bg: float,
                          centre: float, mult: float) -> float:
    """Width of the contiguous run around the centre above mult*background."""
    order = np.argsort(mid)
    m = mid[order]
    vs = _smooth_median(v[order].astype(float), SMOOTH_WINDOWS)
    if not np.isfinite(bg) or bg <= 0:
        return 0.0
    return _run_around(m, np.isfinite(vs) & (vs > mult * bg), centre)


def shape_features(obs: Dict[str, np.ndarray], region: Dict,
                   bg: Dict[str, float], local_rate: float) -> Dict[str, float]:
    """Footprint extents the estimators are fitted on, in genetic units.

    Each width is multiplied by the local recombination rate over the base
    rate, because what a sweep's strength sets is the genetic length of its
    footprint; the same sweep inside a cold spot covers more base pairs.
    """
    same = obs["chrom"] == region["chrom"]
    idx = np.where(same)[0]
    out = {k: 0.0 for k in ("w_pi_50", "w_pi_75", "w_h12_15", "w_h12_2")}
    if idx.size == 0:
        return out
    mid = obs["mid"][idx]
    centre = float(obs["mid"][int(region["peak_index"])])
    scale = (float(local_rate) / BASE_RATE
             if np.isfinite(local_rate) and local_rate > 0 else 1.0)
    pi = obs["pi"][idx]
    h12 = obs["h12"][idx]
    out["w_pi_50"] = contiguous_low_width(mid, pi, bg.get("pi", np.nan),
                                          centre, 0.50) * scale
    out["w_pi_75"] = contiguous_low_width(mid, pi, bg.get("pi", np.nan),
                                          centre, 0.75) * scale
    out["w_h12_15"] = contiguous_high_width(mid, h12, bg.get("h12", np.nan),
                                            centre, 1.5) * scale
    out["w_h12_2"] = contiguous_high_width(mid, h12, bg.get("h12", np.nan),
                                           centre, 2.0) * scale
    return out


def _loglinear(model: Dict, feat: Dict[str, float]) -> Optional[float]:
    """exp(a) * prod(x_i ** b_i), or None if a feature is missing."""
    coef = model["coef"]
    total = float(coef[0])
    for k, b in zip(model["feats"], coef[1:]):
        x = feat.get(k, float("nan"))
        if not np.isfinite(x) or x <= 0:
            return None
        total += float(b) * math.log(float(x))
    return math.exp(total)


def estimate_s(feat: Dict[str, float], klass: str) -> float:
    """Selection coefficient from the class's fitted footprint model."""
    f = feat
    model = S_MODEL.get(klass)
    val = _loglinear(model, f) if model else None
    if val is None:
        val = S_FALLBACK.get(klass, 0.025)
    return float(np.clip(val, 1e-4, 0.5))


def estimate_time(feat: Dict[str, float], klass: str) -> Optional[float]:
    """Completion time from H12 decay, with the footprint extent controlling
    for strength. None for the classes that have no completion time."""
    if klass not in TIMED_CLASSES:
        return None
    val = _loglinear(T_MODEL, feat)
    if val is None:
        val = TIME_FALLBACK_GENS
    return float(np.clip(val, 1.0, 20_000.0))
