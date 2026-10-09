"""Sweep plan construction and replicate acceptance.

Positions are jittered off the configured fractions with a seeded RNG and
forced to non-round values: a selected site sitting on an exact multiple of
100 kb would identify every sweep with no analysis at all.

Geometric constraints (distance from cold spots, from chromosome ends, and
between sweeps) are asserted AFTER jitter, and the jitter is re-drawn until
they hold, so the shipped layout can never silently violate them.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

import config as C
from forward import SweepPlan, SweepRealization
from genome import GenomeConfig


def _violations(pos: int, chrom: int, placed: List[Tuple[int, int]],
                cfg: GenomeConfig) -> List[str]:
    bad = []
    local = pos - cfg.chrom_offset(chrom)
    if local < C.MIN_DIST_FROM_CHROM_END or local > cfg.chrom_len - C.MIN_DIST_FROM_CHROM_END:
        bad.append(f"within {C.MIN_DIST_FROM_CHROM_END} of a chromosome end (local={local})")
    for spot in cfg.cold_spots:
        if spot.chrom != chrom:
            continue
        d = 0 if spot.start <= local < spot.end else min(abs(local - spot.start),
                                                         abs(local - spot.end))
        if d < C.MIN_DIST_FROM_COLDSPOT:
            bad.append(f"{d} from cold spot [{spot.start},{spot.end})")
    for pchrom, ppos in placed:
        if pchrom == chrom and abs(ppos - pos) < C.MIN_SWEEP_SPACING:
            bad.append(f"{abs(ppos - pos)} from another sweep")
    return bad


def build_plans(cfg: GenomeConfig, seed: int) -> List[SweepPlan]:
    rng = np.random.default_rng(seed)
    plans: List[SweepPlan] = []
    placed: List[Tuple[int, int]] = []

    for sid, chrom, frac, s, origin, n_origins, klass in C.SWEEP_SPEC:
        base = cfg.chrom_offset(chrom) + int(frac * cfg.chrom_len)
        for attempt in range(500):
            jitter = int(rng.integers(-C.POSITION_JITTER, C.POSITION_JITTER + 1))
            # force a non-round coordinate
            pos = base + jitter
            if pos % 1000 == 0:
                pos += int(rng.integers(1, 999))
            bad = _violations(pos, chrom, placed, cfg)
            if not bad:
                break
        else:
            raise RuntimeError(f"{sid}: could not place after 500 draws: {bad}")

        placed.append((chrom, pos))
        # h=1.0, NOT 0.5.  fwdpy11's Multiplicative(2.0) assigns genotype
        # fitnesses 1, 1 + h*s, 1 + 2s.  With h=0.5 that is 1, 1+s/2, 1+2s --
        # a partially recessive model whose per-allele advantage is not s, so
        # the s recorded as ground truth would be a number no correct
        # estimator could return.  h=1.0 gives 1, 1+s, 1+2s: additive, with
        # per-allele advantage exactly s.  Verified empirically by reading
        # realized fitness against genotype.
        plans.append(SweepPlan(sweep_id=sid, position=pos, s=s, h=1.0,
                               origin_gens_ago=origin, n_origins=n_origins,
                               intended_class=klass))
    return plans


def realized_class(r: SweepRealization) -> str:
    """Classify a sweep from its REALIZED outcome alone.

    This is the definition the verifier grades against, and it is a function
    of the simulation's recorded trajectory -- never of a solver's output.

    Softness is defined as two or more surviving independent origins of the
    beneficial allele, which is the standard definition of a multiple-origin
    soft sweep and is exactly what the haplotype statistics respond to.
    """
    if not r.completed:
        return "partial"
    if r.n_origins_surviving >= 2:
        return "soft_standing"
    boundary = C.ACCEPT["hard_recent_complete"]["fix_ago_max"]
    return ("hard_recent_complete" if r.completion_gens_ago <= boundary
            else "hard_old_complete")


def check_acceptance(reals: Dict[str, SweepRealization]) -> Tuple[bool, List[str]]:
    """Does this replicate realize every intended class?"""
    problems: List[str] = []
    for sid, r in sorted(reals.items()):
        rule = C.ACCEPT[r.intended_class]
        if rule["must_fix"]:
            if not r.completed:
                problems.append(
                    f"{sid} ({r.intended_class}): did not complete, freq={r.final_freq:.3f}")
                continue
            if not (rule["fix_ago_min"] <= r.completion_gens_ago <= rule["fix_ago_max"]):
                problems.append(
                    f"{sid} ({r.intended_class}): completed {r.completion_gens_ago} "
                    f"gens ago, want [{rule['fix_ago_min']},{rule['fix_ago_max']}]")
            if r.intended_class == "soft_standing" and r.n_origins_surviving < 2:
                problems.append(
                    f"{sid}: only {r.n_origins_surviving} surviving origin(s), "
                    f"so the sweep is not soft")
            if r.intended_class != "soft_standing" and r.n_origins_surviving > 1:
                problems.append(
                    f"{sid}: {r.n_origins_surviving} surviving origins on a sweep "
                    f"meant to be hard")
        else:
            if r.completed:
                problems.append(f"{sid} ({r.intended_class}): completed, wanted partial")
                continue
            if not (rule["freq_min"] <= r.final_freq <= rule["freq_max"]):
                problems.append(
                    f"{sid} ({r.intended_class}): freq {r.final_freq:.3f}, "
                    f"want [{rule['freq_min']},{rule['freq_max']}]")
        if realized_class(r) != r.intended_class:
            problems.append(f"{sid}: realized class {realized_class(r)} != intended {r.intended_class}")
    return (len(problems) == 0), problems
