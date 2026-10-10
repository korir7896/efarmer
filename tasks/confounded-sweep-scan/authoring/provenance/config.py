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
continuous quantity, over the hard_complete sweeps.

Fifteen sweeps rather than ten, and what the extra five buy
-----------------------------------------------------------
The previous zoo carried ten sweeps, all of them strong enough that a scan
for low diversity plus long shared haplotypes found nine, and the
characterisation gates then had three or four calls of slack. Two
independent agent transcripts cleared ten of the twelve gates with that
much room, which is the definition of a task that is not measuring what it
claims to. What is different here:

  * s spans a factor of 12.0, from 0.0075 to 0.0900, and is assigned so
    that s is NON-MONOTONE in age -- the youngest completed sweep is the
    strongest in the zoo, the second youngest is the weakest, and the oldest
    is the second strongest. Neither quantity can be read off the other even
    approximately, so a rank correlation on s cannot be earned by dating.
  * FIVE hard_complete sweeps are dateable instead of four, at ages spaced
    so that the verifier's tolerance windows never overlap three at a time.
    No single constant can date more than two of the five.
  * THREE sweeps sit inside a wide, moderate recombination cold spot
    (SWEEPS_IN_COLDSPOT). Their footprints are stretched in base pairs and
    unchanged in Morgans, so reading selection strength off physical
    footprint width -- the textbook shortcut, and what both agent
    transcripts did -- overestimates s there roughly threefold. Scaling
    width by the local rate recovers it.
  * TWO pairs sit 800 and 840 kb apart (CLOSE_PAIRS): above the verifier's
    250 kb dedupe radius, so both have to be reported, but close enough
    that a scan on megabase windows merges them into one call and loses a
    sweep to the recall gate.
  * Three DEEP NARROW cold spots carry no sweep at all and exist only to
    produce sweep-like statistics where there is nothing to find. The
    genome is 70 Mb rather than 48, so there is half again as much room for
    a miscalibrated scan to spend its one allowed false positive.
  * The sample is 40 diploids rather than 50. Haplotype statistics are the
    ones that separate soft from hard, and they are the ones that lose most
    from a smaller sample.

Timescales are chosen so that no one statistic sees everything, which is
what makes the scan itself hard:

  * N_cur = 9000, so haplotype statistics (H12, nSL, iHS) keep power for
    sweeps younger than roughly 0.1*N = 900 generations and lose it beyond;
  * site-frequency statistics keep power to roughly 0.25*N = 2250;
  * the two oldest hard sweeps completed about 1550 and 2500 generations
    ago, one inside and one beyond even that window;
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

CHROM_LEN = 14_000_000
N_CHROM = 5
BASE_REC_RATE = 1e-8
MUT_RATE = 2.0e-8

N_ANCESTRAL = 12_000
N_BOTTLENECK = 800
N_CURRENT = 9_000
T_BOTTLENECK_START = 3_400      # generations before present
T_BOTTLENECK_END = 3_000
T_FORWARD = 2_900               # forward phase length; < T_BOTTLENECK_END

N_SAMPLE_DIPLOIDS = 40

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
    # --- hard, complete: FIVE widely spread completion times -------------
    # Ages are spaced so that the verifier's tolerance windows never overlap
    # three at a time, which is what stops a single constant from satisfying
    # a gate requiring four correct dates. With a tolerance of
    # max(40%, 120 generations) that needs each age to be more than three
    # times the one two slots below it, so five is the most that fits between
    # 120 generations and the end of the forward phase. Origins come from the
    # probe runs recorded in ACCEPT below, not from theory.
    #
    # s is deliberately NON-MONOTONE in age: the youngest sweep is the
    # strongest in the zoo, the second youngest is the weakest, and the
    # oldest is the second strongest. Nothing about a sweep's strength can be
    # read off its age or the reverse.
    ("sw01", 0, 0.120,  0.0900,  274,   1, "hard_complete"),   # ~130 gens ago
    ("sw02", 1, 0.160,  0.0075, 1569,   1, "hard_complete"),   # ~320
    ("sw03", 1, 0.6571, 0.0400, 1083,   1, "hard_complete"),   # ~760
    ("sw04", 2, 0.420,  0.0130, 2492,   1, "hard_complete"),   # ~1550
    ("sw05", 2, 0.7357, 0.0550, 2722,   1, "hard_complete"),   # ~2500, COLD
    # --- soft, complete: many origins, recent enough that several survive --
    ("sw06", 1, 0.600,  0.0220,  700, 300, "soft_complete"),
    ("sw07", 3, 0.460,  0.0500,  420,  80, "soft_complete"),
    ("sw08", 4, 0.3429, 0.0095, 1500, 150, "soft_complete"),   # COLD, weakest
    # --- hard, partial: one origin, still segregating ---------------------
    ("sw09", 1, 0.8857, 0.0160,  500,   1, "hard_partial"),
    ("sw10", 0, 0.7143, 0.0380,  210,   1, "hard_partial"),    # COLD
    ("sw11", 3, 0.520,  0.0085,  850,   1, "hard_partial"),
    ("sw12", 4, 0.6786, 0.0650,  160,   1, "hard_partial"),
    # --- soft, partial: the weakest cell. Origin counts scale as 1/(2s),
    #     because a founding lineage escapes drift with probability ~2s, so
    #     the SURVIVING origin count stays in the two-to-six range that the
    #     haplotype statistics can actually resolve.
    ("sw13", 0, 0.8929, 0.0120,  430, 150, "soft_partial"),
    ("sw14", 4, 0.8786, 0.0580,  130,  14, "soft_partial"),
    ("sw15", 3, 0.800,  0.0300,  230,  40, "soft_partial"),
]

# Three sweeps sit INSIDE a wide moderate cold spot, one per completeness
# and softness combination that can carry one: sw05 (hard, complete, the
# oldest), sw08 (soft, complete, the weakest) and sw10 (hard, partial).
# Their footprints are stretched in base pairs and unchanged in Morgans, so
# reading s off physical width overestimates it roughly threefold there.
# Asserted against the map in sweeps.build_plans(), not just declared.
SWEEPS_IN_COLDSPOT = ("sw05", "sw08", "sw10")

# Two pairs sit 800 and 840 kb apart, which is above the verifier's 250 kb
# dedupe radius -- so they must be reported as two calls -- but close enough
# that a scan on megabase windows merges them and loses one.
CLOSE_PAIRS = (("sw06", "sw03"), ("sw07", "sw11"))

MIN_SWEEP_SPACING = 600_000
MIN_DIST_FROM_COLDSPOT = 900_000   # decoy spots only
MIN_DIST_FROM_CHROM_END = 1_200_000
POSITION_JITTER = 60_000

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
# Every window below brackets three single-sweep probe runs of the exact
# configuration above (authoring/evidence/make_training.py with --s, --origin
# and --n-origins), and the measured values are quoted beside each one. The
# windows are as wide as the design guarantee allows, because a rejected
# sweep sends the forward driver back to that sweep's origin and replays
# everything after it: sw05 originates 2722 generations back, so rejecting it
# throws away almost the whole forward phase.
#
# Three things the probes settled, each of which changed the config:
#
#   * A soft sweep DECAYS TO HARD the longer ago it completed, because the
#     surviving beneficial backgrounds keep drifting after fixation. sw06 at
#     130 origins realized hard in two draws of three. The fix is more
#     origins, not a younger sweep: 300 origins gave 11 and 17 survivors.
#   * sw13 at origin 600 COMPLETED in two draws of three, so it could not be
#     a soft_partial sweep at all; origin 430 gives frequencies 0.24 to 0.60
#     with three to five surviving origins.
#   * sw12 at origin 120 left frequencies as low as 0.116, which is barely
#     detectable, and at origin 190 it completed. Origin 160 sits between.
#
# sw11 is deliberately left close to completion -- the probes gave 0.762,
# 0.909 and 0.938 -- because a partial sweep at 94% frequency is the hardest
# case the complete-versus-partial axis has, and the zoo should contain one.
ACCEPT: Dict[str, Dict] = {
    # hard_complete: completion-age windows. Spaced so that NO constant can
    # land inside three of the five tolerance ranges for any draw the
    # generator would accept -- checked over the windows themselves by
    # gate_feasibility.design_guarantee(), not over one realized replicate.
    "sw01": dict(class_="hard_complete", must_complete=True,
                 done_min=100,  done_max=180),    # probed 111, 131, 148
    "sw02": dict(class_="hard_complete", must_complete=True,
                 done_min=260,  done_max=430),    # probed 283, 357, 404
    "sw03": dict(class_="hard_complete", must_complete=True,
                 done_min=700,  done_max=880),    # probed 749, 802, 823
    "sw04": dict(class_="hard_complete", must_complete=True,
                 done_min=1420, done_max=1900),   # probed 1487, 1704, 1795
    "sw05": dict(class_="hard_complete", must_complete=True,
                 done_min=2400, done_max=2620),   # probed 2468, 2494, 2532
    # soft_complete: must stay SOFT, which is the binding constraint.
    "sw06": dict(class_="soft_complete", must_complete=True,
                 done_min=200,  done_max=560),    # probed 350 (11 origins)
    "sw07": dict(class_="soft_complete", must_complete=True,
                 done_min=160,  done_max=330),    # probed 218, 220, 240
    "sw08": dict(class_="soft_complete", must_complete=True,
                 done_min=440,  done_max=700),    # probed 550, 582
    # hard_partial: still segregating today.
    "sw09": dict(class_="hard_partial", must_complete=False,
                 freq_min=0.30, freq_max=0.90),   # probed 0.361, 0.542, 0.712
    "sw10": dict(class_="hard_partial", must_complete=False,
                 freq_min=0.22, freq_max=0.93),   # probed 0.267, 0.664, 0.846
    "sw11": dict(class_="hard_partial", must_complete=False,
                 freq_min=0.60, freq_max=0.96),   # probed 0.762, 0.909, 0.938
    "sw12": dict(class_="hard_partial", must_complete=False,
                 freq_min=0.55, freq_max=0.96),   # probed 0.752, 0.894
    # soft_partial: still segregating AND at least two surviving origins.
    "sw13": dict(class_="soft_partial", must_complete=False,
                 freq_min=0.20, freq_max=0.90),   # probed 0.243, 0.340, 0.598
    "sw14": dict(class_="soft_partial", must_complete=False,
                 freq_min=0.15, freq_max=0.93),   # probed 0.199, 0.551, 0.769
    "sw15": dict(class_="soft_partial", must_complete=False,
                 freq_min=0.45, freq_max=0.95),   # probed 0.683, 0.864, 0.890
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
assert _S[-1] / _S[0] > 11.5, f"s spans only {_S[-1] / _S[0]:.2f}"
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
