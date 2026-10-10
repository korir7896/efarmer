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

    Two independent binary properties, crossed:

      * COMPLETE if the beneficial allele passed forward.COMPLETION_FREQ
        (0.98), PARTIAL if it is still below that at the present;
      * SOFT if two or more independent origins of the allele are still
        represented in the population, HARD if exactly one is. That is the
        standard definition of a multiple-origin soft sweep and is exactly
        what the haplotype statistics respond to.

    A sweep's AGE is deliberately not part of its class. An earlier version
    split the completed hard sweeps into "recent" and "old" at a fixed
    boundary, which meant class carried age: a submission could satisfy the
    timing gate with one constant per class without dating anything. Age is
    now graded only as a continuous quantity.
    """
    hard = r.n_origins_surviving < 2
    done = r.completed
    return ("hard_complete" if (hard and done) else
            "hard_partial" if (hard and not done) else
            "soft_complete" if done else "soft_partial")


def check_acceptance(reals: Dict[str, SweepRealization]) -> Tuple[bool, List[str]]:
    """Does this replicate realize what each sweep was scheduled to be?

    Acceptance is per sweep, not per class, because the four hard_complete
    sweeps are scheduled to finish at four widely separated times and each
    has its own window. The windows are what keep the graded completion
    times far enough apart that no single number can satisfy the timing gate
    (see authoring/evidence/gate_feasibility.py, which checks exactly that
    against the realized truth).
    """
    problems: List[str] = []
    for sid, r in sorted(reals.items()):
        rule = C.ACCEPT[sid]
        want = rule["class"]
        if rule["must_complete"]:
            if not r.completed:
                problems.append(
                    f"{sid} ({want}): did not complete, freq={r.final_freq:.3f}")
                continue
            if not (rule["done_min"] <= r.completion_gens_ago <= rule["done_max"]):
                problems.append(
                    f"{sid} ({want}): completed {r.completion_gens_ago} "
                    f"gens ago, want [{rule['done_min']},{rule['done_max']}]")
        else:
            if r.completed:
                problems.append(f"{sid} ({want}): completed, wanted partial")
                continue
            if not (rule["freq_min"] <= r.final_freq <= rule["freq_max"]):
                problems.append(
                    f"{sid} ({want}): freq {r.final_freq:.3f}, "
                    f"want [{rule['freq_min']},{rule['freq_max']}]")
        if want.startswith("soft") and r.n_origins_surviving < 2:
            problems.append(
                f"{sid}: only {r.n_origins_surviving} surviving origin(s), "
                f"so the sweep is not soft")
        if want.startswith("hard") and r.n_origins_surviving > 1:
            problems.append(
                f"{sid}: {r.n_origins_surviving} surviving origins on a sweep "
                f"meant to be hard")
        if realized_class(r) != want:
            problems.append(
                f"{sid}: realized class {realized_class(r)} != scheduled {want}")
    return (len(problems) == 0), problems


def failing_sweeps(reals: Dict[str, SweepRealization]) -> Tuple[bool, List[str]]:
    """Acceptance as the forward driver wants it: (ok, failing sweep ids).

    Lets the driver backtrack to the one sweep that did not work out instead
    of throwing away a whole replicate.
    """
    ok, problems = check_acceptance(reals)
    if ok:
        return True, []
    bad = []
    for sid in sorted(reals):
        if any(p.startswith(sid) for p in problems):
            bad.append(sid)
    return False, bad or sorted(reals)
