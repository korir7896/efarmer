"""Frozen configuration of the shipped replicate.

Every number the agent must recover is defined here and nowhere else.  The
verifier's sealed truth file is generated from this config plus the realized
trajectories recorded by the forward simulation; no solver, reference or
otherwise, contributes to it.

Timescales are chosen so that each sweep class defeats a different statistic:

  * N_cur = 9000, so haplotype-based statistics (H12, nSL, iHS) retain power
    for sweeps younger than roughly 0.1*N = 900 generations and lose it
    beyond that;
  * site-frequency-based statistics retain power to roughly 0.25*N = 2250
    generations;
  * the "old" sweeps fix 1200-1700 generations ago, inside the SFS window and
    outside the haplotype window;
  * the "recent" sweeps fix 150-500 generations ago, inside both;
  * partial sweeps never fix, so statistics keyed on fixed differences or on a
    completed frequency spectrum miss them while nSL/iHS see them;
  * soft sweeps start from many copies, so H12 sees them but H1-style and
    SFS-based statistics are weak.

The bottleneck sits at 3000-3400 generations ago, i.e. 0.17 coalescent units
in the current size.  It is recent and severe enough to skew the genome-wide
site-frequency spectrum and extend linkage disequilibrium everywhere, which is
what makes an uncalibrated scan produce false positives.
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
# (sweep_id, chrom, approx_offset_fraction, s, origin_gens_ago, ndescendants,
#  intended_class)
#
# Positions are jittered off these fractions by the seeded RNG in sweeps.py so
# that no selected site sits on a round number -- a round coordinate would
# identify every sweep without any analysis.
#
# Minimum spacing between selected sites and minimum distance from cold spots
# and chromosome ends are asserted in sweeps.build_plans().

SWEEP_SPEC: List[Tuple[str, int, float, float, int, int, str]] = [
    # --- hard, recent, complete: visible to everything ---
    ("sw01", 0, 0.62, 0.035,  750,  1, "hard_recent_complete"),
    ("sw02", 1, 0.26, 0.030,  820,  1, "hard_recent_complete"),
    ("sw03", 3, 0.71, 0.040,  680,  1, "hard_recent_complete"),
    # --- hard, old, complete: SFS only, haplotype signal decayed ---
    ("sw04", 2, 0.18, 0.028, 2150,  1, "hard_old_complete"),
    ("sw05", 3, 0.22, 0.024, 2300,  1, "hard_old_complete"),
    # --- soft, from standing variation: H12 yes, SFS/H1 weak ---
    ("sw06", 0, 0.86, 0.028,  900, 24, "soft_standing"),
    ("sw07", 1, 0.78, 0.032,  780, 18, "soft_standing"),
    ("sw08", 2, 0.86, 0.026, 1050, 32, "soft_standing"),
    # --- partial / incomplete: never fixes ---
    ("sw09", 1, 0.46, 0.018,  520,  1, "partial"),
    ("sw10", 2, 0.64, 0.020,  480,  1, "partial"),
]

MIN_SWEEP_SPACING = 2_000_000
MIN_DIST_FROM_COLDSPOT = 1_200_000
MIN_DIST_FROM_CHROM_END = 1_200_000
POSITION_JITTER = 180_000

# ---------------------------------------------------------------------------
# replicate acceptance criteria (realized, not intended)
# ---------------------------------------------------------------------------
# A replicate is usable only if every sweep's REALIZED outcome matches its
# intended class under these rules.  This is what lets the generator record
# realized trajectories as truth instead of rejection-sampling to hit exact
# targets.

ACCEPT = {
    "hard_recent_complete": dict(must_fix=True, fix_ago_min=120, fix_ago_max=700),
    "hard_old_complete":    dict(must_fix=True, fix_ago_min=1100, fix_ago_max=1900),
    "soft_standing":        dict(must_fix=True, fix_ago_min=120, fix_ago_max=900),
    "partial":              dict(must_fix=False, freq_min=0.40, freq_max=0.88),
}


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
