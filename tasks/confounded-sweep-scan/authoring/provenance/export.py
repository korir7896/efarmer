"""Sample selection and agent-facing VCF export.

Leak controls applied here, in addition to dropping the selected mutations in
recap.overlay_mutations:

  * only present-day sample nodes are retained; the preserved first
    generation used for recapitation is simplified away;
  * fwdpy11 individual metadata (which carries each individual's fitness `w`
    and genetic value `g`, and so labels sweep carriers outright) is never
    written -- sample names are generated here;
  * only biallelic, segregating sites are written, in ascending position
    order, with no INFO field other than the ancestral allele;
  * every record is formatted identically, so no field distinguishes a site
    near a sweep from any other site.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tskit

from genome import GenomeConfig

GT = np.array(["0|0", "0|1", "1|0", "1|1"], dtype=object)


def present_day_individuals(ts: tskit.TreeSequence) -> np.ndarray:
    """Individuals all of whose nodes are sample nodes at time 0."""
    nt = ts.tables.nodes.time
    flags = ts.tables.nodes.flags
    out = []
    for ind in ts.individuals():
        nodes = np.asarray(ind.nodes)
        if nodes.size == 0:
            continue
        if np.all(nt[nodes] == 0.0) and np.all(flags[nodes] & tskit.NODE_IS_SAMPLE != 0):
            out.append(ind.id)
    return np.array(out, dtype=np.int64)


def subsample(ts: tskit.TreeSequence, n_diploids: int, seed: int
              ) -> Tuple[tskit.TreeSequence, np.ndarray]:
    """Draw `n_diploids` present-day diploids and simplify to them."""
    rng = np.random.default_rng(seed)
    pool = present_day_individuals(ts)
    if pool.size < n_diploids:
        raise ValueError(f"only {pool.size} present-day diploids, need {n_diploids}")
    chosen = np.sort(rng.choice(pool, size=n_diploids, replace=False))
    nodes = np.concatenate([np.asarray(ts.individual(i).nodes) for i in chosen])
    sts = ts.simplify(samples=nodes, filter_individuals=True,
                      filter_sites=True, keep_unary=False)
    return sts, chosen


def scrub_metadata(ts: tskit.TreeSequence) -> tskit.TreeSequence:
    """Blank all individual/node/population metadata inherited from fwdpy11."""
    t = ts.dump_tables()
    t.individuals.metadata_schema = tskit.MetadataSchema.null()
    t.individuals.packset_metadata([b""] * t.individuals.num_rows)
    t.nodes.metadata_schema = tskit.MetadataSchema.null()
    t.nodes.packset_metadata([b""] * t.nodes.num_rows)
    t.populations.metadata_schema = tskit.MetadataSchema.null()
    t.populations.packset_metadata([b""] * t.populations.num_rows)
    t.mutations.metadata_schema = tskit.MetadataSchema.null()
    t.mutations.packset_metadata([b""] * t.mutations.num_rows)
    t.sites.metadata_schema = tskit.MetadataSchema.null()
    t.sites.packset_metadata([b""] * t.sites.num_rows)
    return t.tree_sequence()


@dataclass
class ExportStats:
    n_sites_total: int
    n_written: int
    n_dropped_monomorphic: int
    n_dropped_multiallelic: int
    n_dropped_noninteger: int


def write_vcf(ts: tskit.TreeSequence, cfg: GenomeConfig, path: str,
              sample_prefix: str = "IND") -> ExportStats:
    """Write a phased, biallelic, segregating-sites-only VCF (gzip if .gz)."""
    n_dip = ts.num_samples // 2
    names = [f"{sample_prefix}{i + 1:03d}" for i in range(n_dip)]

    opener = gzip.open if path.endswith(".gz") else open
    mode = "wt"

    n_total = n_mono = n_multi = n_nonint = n_out = 0

    with opener(path, mode) as fh:
        fh.write("##fileformat=VCFv4.2\n")
        for c in range(cfg.n_chrom):
            fh.write(f"##contig=<ID=chr{c + 1},length={cfg.chrom_len}>\n")
        fh.write('##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral allele">\n')
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                 + "\t".join(names) + "\n")

        buf: List[str] = []
        for var in ts.variants():
            n_total += 1
            pos = var.site.position
            if pos != int(pos):
                n_nonint += 1
                continue
            g = var.genotypes
            if np.any(g < 0):
                continue
            alleles = var.alleles
            present = np.unique(g)
            if present.size < 2:
                n_mono += 1
                continue
            if present.size > 2 or len(alleles) > 2:
                n_multi += 1
                continue

            anc = alleles[0]
            der = alleles[1]
            if anc is None or der is None or len(anc) != 1 or len(der) != 1:
                n_multi += 1
                continue

            p = int(pos)
            c = p // cfg.chrom_len
            local = p - c * cfg.chrom_len + 1   # VCF is 1-based

            # diploid genotype index: 2*hap0 + hap1 maps into GT
            gg = g.reshape(n_dip, 2)
            idx = (gg[:, 0] << 1) | gg[:, 1]
            buf.append(
                f"chr{c + 1}\t{local}\t.\t{anc}\t{der}\t.\tPASS\tAA={anc}\tGT\t"
                + "\t".join(GT[idx]) + "\n"
            )
            n_out += 1
            if len(buf) >= 2000:
                fh.writelines(buf)
                buf.clear()
        if buf:
            fh.writelines(buf)

    return ExportStats(n_total, n_out, n_mono, n_multi, n_nonint)
