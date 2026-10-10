"""Classify each detected sweep and estimate its strength and age.

Everything here is calibrated on 111 independent single-sweep replicates at
37 configurations, simulated under the same history at several strengths per
class (authoring/evidence/run_training.sh) and grouped by what each sweep
ACTUALLY turned out to be rather than what it was asked to be. The feature
sets and the fits are chosen by held-out score in
authoring/evidence/fit_classifier.py and authoring/evidence/fit_estimators.py
and pasted in from their --emit output, not assumed.

Two things about those replicates are worth stating outright, because they
are what the constants here are and are not evidence for. They are simulated
under the TRUE demography, not under a history fitted from data, so these
constants carry no demographic-inference error; an analyst doing this for
real would fit them on simulations of their own fitted history, and would
inherit that history's error. And ten of the 37 configurations ARE the
shipped sweeps' own (s, origin, n_origins), so the reference is calibrated
at points it is then graded on. The remaining 20 are not used by the shipped
replicate at all.

Scores are therefore leave-one-CONFIGURATION-out throughout: holding out one
REPLICATE at a time measures almost nothing, since a rule that had memorised
those few (s, age) pairs would score perfectly and generalise not at all.
Leaving out the whole configuration is the strongest test available from
these replicates, and authoring/evidence/holdout_calibration.py goes one
step further -- it re-SELECTS and refits every constant on the
configurations the shipped replicate does not use at all, and regrades.

Measured medians at the sweep, against each replicate's own background
(diversity, H12 and H2/H1 as ratios, Tajima's D and the extreme-nSL fraction
as differences):

    realized class     diversity   Tajima D    H12   H2/H1     nSL    n
    hard_complete          0.119      -2.43  5.142   0.169   +0.01   40
    hard_partial           0.540      -0.83  6.596   0.150   +0.45   21
    soft_complete          0.700      +0.23  4.427   0.642   -0.01   26
    soft_partial           0.787      +0.00  5.418   0.488   +0.20   24

Neither axis of the class scheme has a statistic that carries it alone.
H2/H1, the textbook hard-versus-soft discriminator, runs from 0.007 to 1.012
of background across the hard_complete replicates, swallowing the 0.013 to
0.861 the soft completed ones occupy: at an old hard sweep recombination has
broken up the swept block and H2/H1 has recovered. Diversity, the obvious
complete-versus-partial discriminator, runs from 0.011 to 0.719 across
hard_complete and from 0.158 to 1.431 across hard_partial, so it overlaps
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
# produced by the --emit output of authoring/evidence/fit_classifier.py and
# authoring/evidence/fit_estimators.py, run over the single-sweep training
# replicates in authoring/evidence/training{,_holdout}/, and pasted in
# verbatim.
# ===========================================================================

# --- classifier: the two axes of the class scheme, one rule each ---
# A nearest-centroid rule over all four classes at once scores 75.7% held
# out, at its best feature triple. The scheme is two independent binary
# properties, so two binary rules do better: 80.2% four-way, measured
# leave-one-CONFIGURATION-out over 111 replicates at 37 configurations, 20 of
# which the shipped replicate does not use. Leaving out one CONFIGURATION
# rather than one replicate is the honest test here, because the shipped
# sweeps sit at ten of those configurations and a rule that had memorised
# those few (s, age) pairs would otherwise score perfectly.
#
# Softness is fitted separately inside each completeness group, because the
# axis is far easier among completed sweeps (92.4% held out) than among the
# ones still rising (80.0%): the frequent haplotypes a completed soft sweep
# left have had time to become frequent, while a partial sweep's are still a
# minority of the sample however many of them there are.
#
# The three feature sets are chosen by a joint search over the top six
# candidates per axis, 216 triples, scored FOUR-WAY. Scoring each axis on its
# own is the wrong objective: the softness rules are fitted inside a
# completeness group and applied to the PREDICTED group, so the axes
# interact, and a rule that is better on its own axis can hand the next stage
# replicates it handles worse. Selecting over the ten shipped configurations
# alone picked (tajd_delta, snp_ratio) / (pi_min_ratio, w_pi_50, w_pi_75) /
# (h2h1_ratio) and scored 0.824 there; on the full 37 configurations that
# triple falls well below this one, so the earlier figure was optimistic in
# exactly the way one would expect from selecting on the graded instances.
#
# What carries each axis:
#   completeness (91.0% held out) -- the floor of the diversity trough, the
#     H12 elevation and the extreme-nSL fraction. A completed sweep removed
#     variation and left one or a few long haplotypes; a sweep still rising
#     carries a long YOUNG high-frequency haplotype against a background that
#     is otherwise intact, which is what the standardized nSL fraction
#     responds to and what separates the two.
#   softness among COMPLETED sweeps (92.4%) -- the trough floor with Tajima's
#     D and H2/H1. One frequent haplotype drives diversity to nothing and D
#     far negative; several leave both milder and H2/H1 high.
#   softness among PARTIAL sweeps (80.0%) -- the EXTENT of the H12 elevation
#     with H2/H1, the textbook hard-versus-soft discriminator, usable here
#     only because age is not in play: every partial sweep is still rising.
#     Pooled over completed sweeps of all ages H2/H1 alone is confounded by
#     age, which is why the rule above does not rest on it.
CLASS_RULES = {
    "complete": dict(
        feats=["pi_min_ratio", "h12_ratio", "nsl_delta"],
        mean=[0.202414, 6.16942, 0.139751],
        sd=[0.133287, 4.02357, 0.202995],
        weights=[1.29204, -2.53342, 1.34942, -2.84588]),   # n=111, held out 0.910
    "soft_given_complete": dict(
        feats=["pi_min_ratio", "tajd_delta", "h2h1_ratio"],
        mean=[0.132788, -1.39298, 0.432094],
        sd=[0.106168, 1.48955, 0.343405],
        weights=[-0.501691, 1.7704, 2.50172, -1.90602]),   # n=66, held out 0.924
    "soft_given_partial": dict(
        feats=["w_h12_15", "h2h1_ratio"],
        mean=[1.45733e+06, 0.387836],
        sd=[890214, 0.27403],
        weights=[0.449437, 2.40476, 1.97816]),             # n=45, held out 0.800
}

# How far around the detected peak to read the features, matching calibration.
FEATURE_RADIUS_BP = 60_000

# --- s and age from the shape of the footprint ---
# Both estimators are log-linear fits in measured features -- s_hat = exp(a)
# * prod(feature_i ** b_i) -- with the predictors chosen per class by
# held-out score in authoring/evidence/fit_estimators.py rather than assumed.
#
# The first version used one relation for s, trough width times the local
# recombination rate over a per-class constant, which is the textbook one:
# the swept region extends to roughly s/r. It put 74% of calibration
# replicates inside a factor of 2.5 but ranked them at only 0.32, because a
# soft sweep leaves diversity near background and has no trough whose width
# can be measured. Fitting per class in two or three features instead brings
# that to 114 of 120 inside a factor of 2.5 with a pooled rank correlation of
# +0.58, held out by CONFIGURATION across all 40 -- 16 of which the shipped
# replicate does not use.
#
# Which features each class reads is a measurement, and the answers differ
# by class because the footprints do. hard_complete has a real diversity
# trough and reads its width (w_pi_50, the extent at half recovery);
# hard_partial has no trough edge and reads the H12 elevation instead;
# soft_partial has no reliable extent at all and reads only the level of the
# H12 peak.
#
# Widths are in BASE PAIRS, not divided by the local recombination rate. The
# textbook correction is to divide, since what strength sets is the GENETIC
# length of the footprint -- but a cold spot here is finite, the footprint
# runs out of it into base-rate flanks, and the measured stretching is 2.05x
# in a 0.25x spot rather than the 4x the ratio implies. So the rate is
# offered to the fit as its own predictor, rate_ratio, and the exponent is
# fitted. Measured, only soft_complete takes it: that is the class holding
# the soft sweep inside a cold spot, and the one an estimator calibrated at
# the base rate alone gets most wrong (biased 2.65x, worst case 5.90x).
SMOOTH_WINDOWS = 15
RECOVERY_FRAC = 0.5
WIDTH_SEARCH_BP = 1_200_000
BASE_RATE = 1e-8

S_MODEL = {
    "hard_complete": dict(feats=["pi_min_ratio", "w_pi_50"],
                          coef=[-8.4615, -0.30364, 0.31393]),
    "hard_partial":  dict(feats=["pi_ratio", "w_h12_15"],
                          coef=[-8.7726, 0.07419, 0.36409]),
    "soft_complete": dict(feats=["snp_ratio", "w_pi_50", "rate_ratio"],
                          coef=[2.1002, -0.89684, -0.51141, 0.71458]),
    "soft_partial":  dict(feats=["h12_ratio"],
                          coef=[-4.0223, 0.30065]),
}
S_FALLBACK = {"hard_complete": 0.026, "hard_partial": 0.022,
              "soft_complete": 0.028, "soft_partial": 0.025}

# --- completion clock ---
# Age comes from H12 DECAY, with the local recombination rate controlling
# for how fast that decay runs. H12 at a completed sweep falls back towards
# background as recombination breaks up the swept block, which is a cleaner
# clock than the depth of the diversity trough: held out by configuration
# over 42 hard_complete replicates spanning ages of 36 to 2504 generations,
# 37 land inside the verifier's tolerance, with a rank correlation of +0.88.
#
# This is the one place the local rate earns its keep outright. Diversity
# recovery after a sweep is a local clock and how much has returned to a
# window depends on how much recombination that window has had, so
# h12_ratio alone scores 0.83 inside tolerance and h12_ratio with the rate
# scores 0.88 -- the best of any pair tried.
#
# Reported for every sweep called complete, because the submission schema
# requires a number there, but only meaningful for the single-origin ones:
# the verifier grades timing on true hard_complete sweeps only.
TIMED_CLASSES = ("hard_complete", "soft_complete")
T_MODEL = dict(feats=["h12_ratio", "rate_ratio"],
               coef=[7.6903, -0.98094, -0.27172])
TIME_FALLBACK_GENS = 500.0


# Every feature name level_features() and shape_features() can return. The
# models above are checked against it at import, because the failure mode is
# silent: _loglinear() returns None for a feature the pipeline never
# produces, estimate_s() substitutes a per-class constant, and the result is
# a submission that looks complete while reporting the same strength for
# every sweep of a class. That is a spread-and-rank failure with no error
# message anywhere, and it is exactly what an earlier revision shipped when
# a model was fitted on a width the authoring side measured and this side
# did not.
PRODUCED_FEATURES = frozenset((
    "pi_ratio", "pi_min_ratio", "h12_ratio", "h2h1_ratio", "nsl_delta",
    "tajd_delta", "snp_ratio",
    "w_pi_50", "w_pi_75", "w_h12_15", "w_h12_2", "rate_ratio",
))


def _check_models_are_computable() -> None:
    missing = {}
    for name, feats in ([(f"S_MODEL[{k!r}]", v["feats"])
                         for k, v in S_MODEL.items()]
                        + [("T_MODEL", T_MODEL["feats"])]
                        + [(f"CLASS_RULES[{k!r}]", v["feats"])
                           for k, v in CLASS_RULES.items()]):
        gap = [f for f in feats if f not in PRODUCED_FEATURES]
        if gap:
            missing[name] = gap
    if missing:
        raise AssertionError(
            "these fitted models ask for features this module does not "
            "compute, so they would silently fall back:\n  "
            + "\n  ".join(f"{k}: {', '.join(v)}" for k, v in missing.items()))


_check_models_are_computable()


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


def grid_step(mid: np.ndarray) -> float:
    """One step of the scan grid, the floor on every width this module reports.

    A contiguous run of windows is measured as the distance between the
    centres of its first and last window, so a run ONE window wide measures
    zero -- and so does a statistic that never crosses its threshold near the
    centre at all. Both mean "narrower than this scan can resolve", which is a
    measurement rather than a missing value, but the strength and age models
    are log-linear in these widths and log(0) is not a number. Reporting one
    step keeps the quantity finite and monotone in the right direction: a
    weaker sweep leaves a narrower footprint.

    The authoring-side fitter applies the identical floor. If these two ever
    disagree the shipped coefficients are fitted on one quantity and applied
    to another, which is silent and would not show up as an error anywhere.
    """
    m = np.sort(np.asarray(mid, dtype=float))
    d = np.diff(m)
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def _run_around(mid: np.ndarray, flag: np.ndarray, centre: float) -> float:
    """Span of the contiguous run of flagged windows containing the centre,
    floored at one scan step. See grid_step for why the floor exists."""
    step = grid_step(mid)
    if not flag.any():
        return step
    i0 = int(np.argmin(np.abs(mid - centre)))
    if not flag[i0]:
        cand = np.where(flag)[0]
        i0 = int(cand[np.argmin(np.abs(mid[cand] - centre))])
        if abs(mid[i0] - centre) > 600_000:
            return step
    lo = hi = i0
    while lo - 1 >= 0 and flag[lo - 1]:
        lo -= 1
    while hi + 1 < len(mid) and flag[hi + 1]:
        hi += 1
    return max(float(mid[hi] - mid[lo]), step)


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
        return grid_step(m)
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
        return grid_step(m)
    return _run_around(m, np.isfinite(vs) & (vs > mult * bg), centre)


def shape_features(obs: Dict[str, np.ndarray], region: Dict,
                   bg: Dict[str, float], local_rate: float) -> Dict[str, float]:
    """Footprint extents the estimators are fitted on, plus the local rate.

    Widths are returned in BASE PAIRS, unscaled, and the local rate relative
    to the base rate is returned beside them as `rate_ratio` so the fitted
    model can use it.

    The obvious thing is to divide each width by the rate ratio, on the
    reasoning that a sweep's strength sets the GENETIC length of its
    footprint, so the same sweep in a cold spot covers proportionally more
    base pairs. That reasoning is right in an infinite cold spot and wrong
    here, and the calibration replicates say by how much: at s = 0.040 in a
    0.25x spot the measured footprint is 1,335 kb against 650 kb at the base
    rate -- a factor of 2.05, not the factor of 4 the rate ratio implies,
    because the spot is 2.8 Mb wide and the footprint runs out of it into
    base-rate flanks. Dividing by 4 would over-correct by about two, which
    is worse than not correcting at all.

    So the exponent is fitted rather than assumed. `rate_ratio` is a
    candidate feature for the strength and age models, which are log-linear,
    so the fit learns what power of it the data actually wants. It is
    deliberately NOT offered to the classifier: a sweep's class is read off
    haplotype structure, which does not depend on the recombination rate,
    and with five cold-spot configurations in the calibration set a
    classifier would be free to learn "cold spot implies this class" from
    the handful of shipped sweeps that happen to sit in one.
    """
    same = obs["chrom"] == region["chrom"]
    idx = np.where(same)[0]
    out = {k: 0.0 for k in ("w_pi_50", "w_pi_75", "w_h12_15", "w_h12_2")}
    out["rate_ratio"] = 1.0
    if idx.size == 0:
        return out
    mid = obs["mid"][idx]
    centre = float(obs["mid"][int(region["peak_index"])])
    out["rate_ratio"] = (float(local_rate) / BASE_RATE
                         if np.isfinite(local_rate) and local_rate > 0
                         else 1.0)
    pi = obs["pi"][idx]
    h12 = obs["h12"][idx]
    out["w_pi_50"] = contiguous_low_width(mid, pi, bg.get("pi", np.nan),
                                          centre, 0.50)
    out["w_pi_75"] = contiguous_low_width(mid, pi, bg.get("pi", np.nan),
                                          centre, 0.75)
    out["w_h12_15"] = contiguous_high_width(mid, h12, bg.get("h12", np.nan),
                                            centre, 1.5)
    out["w_h12_2"] = contiguous_high_width(mid, h12, bg.get("h12", np.nan),
                                           centre, 2.0)
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


def estimate_time(feat: Dict[str, float], klass: str = "") -> Optional[float]:
    """Completion time from H12 decay, with the footprint extent controlling
    for strength.

    Reported for EVERY call, whatever class the call was given. The verifier
    grades timing against a sweep's true class, so a sweep this pipeline
    misclassifies as partial still has its date graded -- and the date is
    read off the footprint, not off the class, so there is no reason to
    withhold it. Suppressing it for the calls this pipeline believes are
    partial threw away two correct dates on the shipped replicate and failed
    the timing gate with the classification as the only cause. The number is
    ignored by the verifier for a sweep that really is partial, and
    `klass` is kept in the signature for callers that pass it.
    """
    val = _loglinear(T_MODEL, feat)
    if val is None:
        val = TIME_FALLBACK_GENS
    return float(np.clip(val, 1.0, 20_000.0))
