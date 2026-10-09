"""Neutral coalescent replicates under the same demography and genetic map.

Used for two things:
  * calibrating what the scan statistics do when there is no selection at all
    (including inside the recombination cold spots, which is where an
    uncalibrated scan invents sweeps);
  * supplying the reference solution with its own null distribution.

No selection anywhere: any peak in a replicate from here is a false positive
by construction.
"""
from __future__ import annotations

from typing import Dict, Optional

import allel
import msprime
import numpy as np
import tskit

from genome import GenomeConfig
from recap import Demography, recombination_map
from sweepstats import ChromData


def simulate_null(cfg: GenomeConfig, demog: Demography, n_diploids: int,
                  seed: int) -> tskit.TreeSequence:
    d = demog.to_msprime("pop")
    ts = msprime.sim_ancestry(
        samples={"pop": n_diploids},
        demography=d,
        recombination_rate=recombination_map(cfg),
        random_seed=seed,
    )
    return msprime.sim_mutations(ts, rate=cfg.mut_rate, model=msprime.JC69(),
                                 discrete_genome=True, random_seed=seed + 1)


def ts_to_chromdata(ts: tskit.TreeSequence, cfg: GenomeConfig
                    ) -> Dict[str, ChromData]:
    """Split a whole-genome tree sequence into per-chromosome haplotype arrays.

    Applies exactly the filters the VCF exporter applies -- biallelic,
    segregating, integer position -- so statistics computed here are
    comparable with statistics computed on the agent's file.
    """
    pos_all = []
    gen_all = []
    for var in ts.variants():
        p = var.site.position
        if p != int(p):
            continue
        g = var.genotypes
        u = np.unique(g)
        if u.size != 2 or len(var.alleles) > 2:
            continue
        pos_all.append(int(p))
        gen_all.append(g.astype(np.int8))
    if not pos_all:
        raise RuntimeError("no usable variants")
    pos_all = np.asarray(pos_all, dtype=np.int64)
    G = np.vstack(gen_all)

    out: Dict[str, ChromData] = {}
    for c in range(cfg.n_chrom):
        lo, hi = cfg.chrom_bounds(c)
        m = (pos_all >= lo) & (pos_all < hi)
        if not m.any():
            continue
        sub = G[m]
        hap = allel.HaplotypeArray(sub)
        n_hap = sub.shape[1]
        der = sub.sum(axis=1)
        ac = np.stack([n_hap - der, der], axis=1)
        out[f"chr{c + 1}"] = ChromData(
            name=f"chr{c + 1}",
            pos=(pos_all[m] - lo + 1).astype(np.int64),  # 1-based, like the VCF
            hap=hap, ac=ac,
        )
    return out
