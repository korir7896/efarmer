"""Recapitation, neutral mutation overlay and VCF export.

The forward phase leaves a tree sequence whose trees have many roots at the
start of the forward epoch.  msprime supplies the deeper history exactly:
the tail of the expanded epoch, the bottleneck and the ancestral epoch.

Chromosome independence is reproduced in the coalescent part by a one-base
interval of recombination rate ln(2) at each chromosome boundary.  Under
msprime's discrete genome at most one breakpoint can fall in a one-base
interval, so the probability of a breakpoint there is 1 - exp(-ln 2) = 0.5
exactly, matching fwdpy11's BinomialPoint(0.5) in the forward phase.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import msprime
import tskit

from genome import GenomeConfig

BOUNDARY_RATE = math.log(2.0)


@dataclass
class Demography:
    """Bottleneck-and-expansion history, in generations before present."""
    n_cur: int
    n_bot: int
    n_anc: int
    t_expansion: int      # gens ago the bottleneck ended / expansion happened
    t_bottleneck: int     # gens ago the bottleneck began

    def __post_init__(self):
        if not (0 < self.t_expansion < self.t_bottleneck):
            raise ValueError("require 0 < t_expansion < t_bottleneck")

    @property
    def bottleneck_duration(self) -> int:
        return self.t_bottleneck - self.t_expansion

    def to_msprime(self, population_name: str) -> msprime.Demography:
        d = msprime.Demography()
        d.add_population(name=population_name, initial_size=self.n_cur)
        d.add_population_parameters_change(
            time=self.t_expansion, initial_size=self.n_bot, population=population_name)
        d.add_population_parameters_change(
            time=self.t_bottleneck, initial_size=self.n_anc, population=population_name)
        return d

    def to_dict(self):
        return {
            "N_ancestral": self.n_anc,
            "N_bottleneck": self.n_bot,
            "N_current": self.n_cur,
            "T_bottleneck_start_gens_ago": self.t_bottleneck,
            "T_bottleneck_end_gens_ago": self.t_expansion,
            "bottleneck_duration_gens": self.bottleneck_duration,
        }


def recombination_map(cfg: GenomeConfig) -> msprime.RateMap:
    """msprime RateMap mirroring the forward-phase genetic map.

    The final base of every chromosome except the last carries rate ln(2),
    giving exactly 50% recombination between chromosomes.
    """
    pos: List[float] = [0.0]
    rate: List[float] = []
    for s, e, r in cfg.segments():
        seg_end = float(e)
        c_end = cfg.chrom_offset(s // cfg.chrom_len + 1)
        is_last_seg_of_chrom = (e == c_end)
        last_chrom = (e == cfg.total_len)
        if is_last_seg_of_chrom and not last_chrom:
            # split off the final base to carry the boundary rate
            if seg_end - 1.0 > pos[-1]:
                pos.append(seg_end - 1.0)
                rate.append(r)
            pos.append(seg_end)
            rate.append(BOUNDARY_RATE)
        else:
            pos.append(seg_end)
            rate.append(r)
    return msprime.RateMap(position=np.array(pos), rate=np.array(rate))


def recapitate(ts: tskit.TreeSequence, cfg: GenomeConfig, demog: Demography,
               seed: int) -> tskit.TreeSequence:
    pop_name = ts.population(0).metadata.get("name", "pop0")
    d = demog.to_msprime(pop_name)
    rts = msprime.sim_ancestry(
        initial_state=ts,
        demography=d,
        recombination_rate=recombination_map(cfg),
        random_seed=seed,
    )
    return rts


def overlay_mutations(ts: tskit.TreeSequence, cfg: GenomeConfig,
                      seed: int) -> tskit.TreeSequence:
    """Add neutral mutations under a Jukes-Cantor model on a discrete genome.

    `keep=False` DISCARDS the forward-phase selected mutations.  This is
    deliberate and is the single most important leak control in the generator.
    fwdpy11 writes selected variants with a `0`/`1` allele alphabet (against
    ACGT for everything msprime produces), at non-integer positions, and with
    metadata that states the selection coefficient outright -- any one of
    which identifies every sweep with a one-line grep.

    Nothing about the sweep signal is lost.  A sweep is a feature of the
    genealogy, not of the beneficial mutation itself: the distorted coalescent
    tree around the selected site, and therefore the diversity, site-frequency
    and haplotype signatures in the linked neutral variation, are all
    unchanged.  Dropping the causal site only means the agent must localize
    sweeps from linked variation, which is what a real genome scan does.
    """
    return msprime.sim_mutations(
        ts,
        rate=cfg.mut_rate,
        model=msprime.JC69(),
        discrete_genome=True,
        keep=False,
        random_seed=seed,
    )
