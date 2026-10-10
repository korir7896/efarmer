"""Frozen configuration of the shipped replicate.

Every number the agent must recover is defined here and nowhere else.  The
verifier's sealed truth file is generated from this config plus the realized
trajectories recorded by the forward simulation; no solver, reference or
otherwise, contributes to it.

The sweep zoo
-------------
The four classes are the two binary properties of a sweep that a genome scan
can be asked to recover, crossed:

  * COMPLETE if the beneficial allele passed 0.98 (forward.COMPLETION_FREQ),
    PARTIAL if it is still below that today;
  * SOFT if two or more independent origins of the allele survive to the
    present, HARD if exactly one does.

A sweep's AGE is deliberately not part of its class. An earlier version of
this task split the completed hard sweeps into "recent" and "old" at a fixed
boundary, which put age inside the class label: a submission could then
satisfy the timing gate with one constant per class without dating anything,
and the review of that version said so. Age is now graded only as a
continuous quantity, over the hard_complete sweeps, whose completion times
here span a factor of about sixteen.

Two spreads do the work of making the graded quantities discriminating:

  * s spans a factor of 9.4, from 0.0085 to 0.080, and is assigned across
    classes so that neither s nor age can be read off the other;
  * the four hard_complete completion times are far enough apart that the
    tolerance windows of any three of them have no point in common, so no
    single number can satisfy a gate that requires three correct dates.
    gate_feasibility.py checks that against the realized truth rather than
    against these targets.

Timescales are also chosen so that no one statistic sees everything, which
is what makes the scan itself hard:

  * N_cur = 9000, so haplotype statistics (H12, nSL, iHS) keep power for
    sweeps younger than roughly 0.1*N = 900 generations and lose it beyond;
  * site-frequency statistics keep power to roughly 0.25*N = 2250;
  * the two oldest hard sweeps completed 1100-2100 generations ago, inside
    the SFS window and outside the haplotype window;
  * partial sweeps never complete, so statistics keyed on fixed differences
    or on a completed spectrum miss them while nSL/iHS see them;
  * soft sweeps carry several frequent haplotypes, so H12 sees them while
    H1-style and SFS-based statistics are weak.

The bottleneck sits at 3000-3400 generations ago, i.e. 0.17 coalescent units
in the current size.  It is recent and severe enough to skew the genome-wide
site-frequency spectrum and extend linkage disequilibrium everywhere, which
is what makes an uncalibrated scan produce false positives.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

CHROM_LEN = 12_000_000
N_CHROM = 4
BASE_REC_RATE = 1e-8
MUT_RATE = 2.0e-8

N_ANCESTRAL = 12_000
N_BOTTLENECK = 800
N_CURRENT = 9_000
T_BOTTLENECK_START = 3_400      # generations before present
T_BOTTLENECK_END = 3_000
T_FORWARD = 2_600               # forward phase length; < T_BOTTLENECK_END

N_SAMPLE_DIPLOIDS = 50

MASTER_SEED = 20_251_009

# ---------------------------------------------------------------------------
# sweep zoo
# ---------------------------------------------------------------------------
# (sweep_id, chrom, approx_offset_fraction, s, origin_gens_ago, n_origins,
#  scheduled_class)
#
# n_origins is the number of INDEPENDENT introductions of the beneficial
# allele at that position. One is a hard sweep. Many is a soft sweep from
# recurrent mutation: the allele sits on that many different haplotype
# backgrounds, of which some survive, so the swept region ends up carrying
# several frequent haplotypes instead of one. See forward.py for why putting
# the allele on a single node with many descendants does not do this.
#
# Origin times come from the measurements in
# authoring/evidence/tune_trajectories.py (grid v3, 8 replicates per
# configuration at N_CURRENT on a short genome), reproduced in ACCEPT below.
#
# Positions are jittered off these fractions by the seeded RNG in sweeps.py so
# that no selected site sits on a round number -- a round coordinate would
# identify every sweep without any analysis.
#
# Minimum spacing between selected sites and minimum distance from cold spots
# and chromosome ends are asserted in sweeps.build_plans().

SWEEP_SPEC: List[Tuple[str, int, float, float, int, int, str]] = [
    # --- hard, complete: four widely spread completion times. s is NOT
    #     monotone in age, so neither can be inferred from the other.
    ("sw01", 0, 0.14,  0.0800,  280,   1, "hard_complete"),   # ~130 gens ago
    ("sw02", 1, 0.46,  0.0300,  710,   1, "hard_complete"),   # ~330
    ("sw03", 2, 0.655, 0.0085, 2300,   1, "hard_complete"),   # ~1140
    ("sw04", 3, 0.71,  0.0250, 2500,   1, "hard_complete"),   # ~2020
    # --- soft, complete: many origins, recent enough that several survive ---
    ("sw05", 0, 0.62,  0.0180,  650, 100, "soft_complete"),
    ("sw06", 1, 0.84,  0.0450,  380, 100, "soft_complete"),
    # --- hard, partial: one origin, still segregating ---
    ("sw07", 2, 0.18,  0.0140,  530,   1, "hard_partial"),
    ("sw08", 0, 0.86,  0.0350,  230,   1, "hard_partial"),
    # --- soft, partial: the weakest cell. Twelve origins rather than a
    #     hundred, so the surviving beneficial haplotypes are few enough to
    #     leave a haplotype signal while still being more than one.
    ("sw09", 1, 0.26,  0.0110,  560, 150, "soft_partial"),
    ("sw10", 3, 0.22,  0.0600,  130,  12, "soft_partial"),
]

MIN_SWEEP_SPACING = 2_000_000
MIN_DIST_FROM_COLDSPOT = 1_200_000
MIN_DIST_FROM_CHROM_END = 1_200_000
POSITION_JITTER = 180_000

# ---------------------------------------------------------------------------
# replicate acceptance criteria (realized, not scheduled)
# ---------------------------------------------------------------------------
# A replicate is usable only if every sweep's REALIZED outcome falls in its
# own window. This is what lets the generator record realized trajectories as
# truth instead of rejection-sampling to hit exact targets, and it is per
# sweep rather than per class because the four hard_complete sweeps are aimed
# at four different ages.
#
# Measured over 8 independent replicates per configuration at N_CURRENT
# (authoring/evidence/tune_trajectories.py, grid v3); completion is the
# generation the allele passed 0.98, which is the graded quantity:
#
#   s=0.0800 origin  280   8/8 complete   completed  103-143 generations ago
#   s=0.0300 origin  710   8/8 complete   completed  263-351
#   s=0.0085 origin 2300   8/8 complete   completed  874-1278
#   s=0.0250 origin 2500   8/8 complete   completed 1937-2065
#   s=0.0180 origin  650 x100   8/8 complete, completed 118-214,
#                                 5-9 surviving origins, soft in 8/8
#   s=0.0450 origin  380 x100   8/8 complete, completed 155-287,
#                                 1-9 surviving origins, soft in 6/8
#   s=0.0140 origin  450        0/8 complete, final frequency 0.118-0.745
#   s=0.0140 origin  520        0/8 complete, final frequency 0.112-0.833
#   s=0.0350 origin  210        0/8 complete, final frequency 0.075-0.824
#   s=0.0350 origin  230        0/8 complete, final frequency 0.535-0.855
#   s=0.0600 origin  140 x12    0/8 complete, final frequency 0.850-0.924,
#                                 5-6 surviving origins, soft in 8/8
#
#   s=0.0110 origin  740 x12    calibration replicates reached 0.860, 0.931
#                                 and 0.986: too far along, and the third
#                                 counts as complete.
#
# Those numbers are from ONE-SWEEP runs on a one-megabase genome, and the
# shipped genome is forty-eight megabases carrying ten sweeps. For the
# completed sweeps the two agree closely; for the partial ones they do not,
# which is why generate.py has --forward-only and why the windows above are
# set from three probe runs of the real configuration
# (authoring/evidence/context_probe.md):
#
#   sw01 109   sw02 397   sw03 1179  sw04 1946  generations ago
#   sw05 129 (6 origins)  sw06 295 (4 origins)
#   sw07 freq 0.702       sw08 freq 0.479       sw10 freq 0.754 (3 origins)
#
# Every one of those is inside its window. sw09 was the single failure, and
# for one reason: at s=0.011 only about 2s of the founding lineages escape
# drift, so twelve origins left ONE surviving lineage and the sweep realized
# as hard_partial. The number of origins a soft sweep needs therefore scales
# as 1/(2s): three surviving lineages at s=0.011 needs about 3/(2*0.011) =
# 136 introductions, so sw09 carries 150. What the haplotype statistics see
# is the number of SURVIVING origins, which stays at three to five, so the
# calibration replicates remain comparable.
# sw10's origin is pulled back from the measured 140 to 130 because 140 put
# the median at 0.896, close enough to completion to risk rejection.
#
# Two things that measurement settled. A weak sweep's completion time
# scatters far more than a strong one's: s=0.0085 completed anywhere in a
# 404-generation band from a fixed origin, where s=0.0800 held inside 40. So
# the youngest slot, the one with the tightest absolute tolerance in the
# timing gate, went to the strongest sweep and the weak coefficients went to
# the older slots where the tolerance is hundreds of generations wide. And
# s=0.0085 passed 0.98 in 8 of 8 while reaching exact fixation in only 3: the
# last few ancestral haplotypes take a long time to go, which is why
# completion rather than fixation is both the stated convention and the
# estimable one.
#
# The windows below bracket those measurements, tightened at the edges where
# a looser window would let two hard_complete sweeps drift close enough
# together for one number to date both.
# A rejected sweep sends the forward driver back to that sweep's origin and
# replays everything after it, so a narrow window on an EARLY sweep is
# expensive: sw03's origin is 2300 generations back, and rejecting it throws
# away almost the whole forward phase. The windows are therefore as wide as
# the timing gate's non-triviality allows, which
# gate_feasibility.design_guarantee() checks rather than assumes -- it takes
# the windows, not one realized replicate, and verifies that no constant can
# land inside three of the four hard_complete tolerance ranges for ANY draw
# the generator would accept.
ACCEPT: Dict[str, Dict] = {
    "sw01": dict(class_="hard_complete", must_complete=True,
                 done_min=90,   done_max=210),
    "sw02": dict(class_="hard_complete", must_complete=True,
                 done_min=240,  done_max=420),
    "sw03": dict(class_="hard_complete", must_complete=True,
                 done_min=820,  done_max=1400),
    "sw04": dict(class_="hard_complete", must_complete=True,
                 done_min=1850, done_max=2200),
    "sw05": dict(class_="soft_complete", must_complete=True,
                 done_min=100,  done_max=270),
    "sw06": dict(class_="soft_complete", must_complete=True,
                 done_min=130,  done_max=330),
    "sw07": dict(class_="hard_partial", must_complete=False,
                 freq_min=0.42, freq_max=0.93),
    "sw08": dict(class_="hard_partial", must_complete=False,
                 freq_min=0.42, freq_max=0.93),
    "sw09": dict(class_="soft_partial", must_complete=False,
                 freq_min=0.45, freq_max=0.95),
    "sw10": dict(class_="soft_partial", must_complete=False,
                 freq_min=0.45, freq_max=0.95),
}
# `class` is a keyword, so the dicts above carry `class_`; expose it under the
# name the rest of the code reads.
for _sid, _rule in ACCEPT.items():
    _rule["class"] = _rule.pop("class_")
del _sid, _rule

assert sorted(ACCEPT) == sorted(spec[0] for spec in SWEEP_SPEC)
assert all(ACCEPT[spec[0]]["class"] == spec[6] for spec in SWEEP_SPEC)
_S = sorted(spec[3] for spec in SWEEP_SPEC)
assert len(set(_S)) == len(_S), "duplicate selection coefficients"
assert _S[-1] / _S[0] > 9.0, f"s spans only {_S[-1] / _S[0]:.2f}"
del _S


def genome_config():
    from genome import GenomeConfig, default_cold_spots
    return GenomeConfig(
        n_chrom=N_CHROM, chrom_len=CHROM_LEN,
        base_rec_rate=BASE_REC_RATE, mut_rate=MUT_RATE,
        cold_spots=default_cold_spots(CHROM_LEN),
    )


def demography():
    from recap import Demography
    return Demography(
        n_cur=N_CURRENT, n_bot=N_BOTTLENECK, n_anc=N_ANCESTRAL,
        t_expansion=T_BOTTLENECK_END, t_bottleneck=T_BOTTLENECK_START,
    )
