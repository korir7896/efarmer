"""Genome architecture: chromosomes, recombination map with cold spots.

Shared by the generator (authoring) and exported to the agent as a BED-like
rate map. Nothing here encodes selected-site locations.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Tuple


@dataclass(frozen=True)
class ColdSpot:
    """A low-recombination window, given as (chrom_index, start, end, factor).

    `decoy` says which job the spot does, and the two jobs want opposite
    geometry:

      * A DECOY is narrow and deep. It carries no sweep, and its only purpose
        is to produce a sweep-like signal -- heavy tails in every statistic,
        because fewer independent genealogies fall in a window -- where there
        is nothing to find. Deep and narrow maximises that per base pair.

      * A SWEEP-BEARING spot is wide and moderate. It holds a selected site,
        and its purpose is to stretch that sweep's footprint in BASE PAIRS
        while leaving it unchanged in Morgans. A method that reads selection
        strength off physical footprint width, which is the textbook
        shortcut, overestimates s there by roughly the inverse of `factor`;
        one that scales by the local rate does not. The spot has to be WIDE
        for this to bite: the footprint only stretches while it is still
        inside the spot, so a 300 kb spot barely moves it however deep it is,
        and the spots that hold sweeps here are 2.6 to 3.0 Mb.

    Nothing in the exported map says which is which -- both are just rate
    segments -- so the distinction is a property of the design, not a hint.
    """
    chrom: int
    start: int          # relative to chromosome start
    end: int
    factor: float       # local rate = base_rate * factor
    decoy: bool = True


@dataclass
class GenomeConfig:
    n_chrom: int = 4
    chrom_len: int = 12_000_000
    base_rec_rate: float = 1e-8          # per bp per generation
    mut_rate: float = 1.2e-8             # neutral overlay, per bp per generation
    cold_spots: List[ColdSpot] = field(default_factory=list)

    @property
    def total_len(self) -> int:
        return self.n_chrom * self.chrom_len

    def chrom_offset(self, c: int) -> int:
        return c * self.chrom_len

    def chrom_bounds(self, c: int) -> Tuple[int, int]:
        o = self.chrom_offset(c)
        return o, o + self.chrom_len

    def to_dict(self):
        d = asdict(self)
        return d

    def segments(self) -> List[Tuple[int, int, float]]:
        """Tile the whole genome as (start, end, rate) in global coordinates.

        Segments never straddle a chromosome boundary.
        """
        segs: List[Tuple[int, int, float]] = []
        for c in range(self.n_chrom):
            o = self.chrom_offset(c)
            cs = sorted([x for x in self.cold_spots if x.chrom == c],
                        key=lambda x: x.start)
            cur = 0
            for spot in cs:
                if spot.start > cur:
                    segs.append((o + cur, o + spot.start, self.base_rec_rate))
                segs.append((o + spot.start, o + spot.end,
                             self.base_rec_rate * spot.factor))
                cur = spot.end
            if cur < self.chrom_len:
                segs.append((o + cur, o + self.chrom_len, self.base_rec_rate))
        return segs

    def rate_at(self, pos: int) -> float:
        for s, e, r in self.segments():
            if s <= pos < e:
                return r
        raise ValueError(f"position {pos} outside genome")


def default_cold_spots(chrom_len: int) -> List[ColdSpot]:
    """Six cold spots of two kinds, on four of the five chromosomes.

    Chromosome 4 (index 3) deliberately has none and chromosomes 1 and 3
    (indices 0 and 2) have two, so the agent can neither assume one per
    chromosome nor stop looking after finding one. Widths differ by a factor
    of ten so no single hard-coded width finds them.

    Three are deep narrow decoys carrying no sweep; three are wide moderate
    spots that each hold one. See ColdSpot for why the two kinds have
    opposite geometry.
    """
    return [
        # deep, narrow, no sweep inside
        ColdSpot(chrom=0, start=3_920_000, end=4_370_000, factor=0.10),
        ColdSpot(chrom=1, start=6_160_000, end=6_460_000, factor=0.08),
        ColdSpot(chrom=2, start=2_800_000, end=3_500_000, factor=0.12),
        # wide, moderate, one selected site inside each
        ColdSpot(chrom=0, start=8_600_000, end=11_400_000, factor=0.25,
                 decoy=False),
        ColdSpot(chrom=2, start=8_800_000, end=11_800_000, factor=0.22,
                 decoy=False),
        ColdSpot(chrom=4, start=3_500_000, end=6_100_000, factor=0.28,
                 decoy=False),
    ]


def write_rate_map_bed(cfg: GenomeConfig, path: str) -> None:
    """Agent-facing recombination map. Plain BED4: chrom start end rate."""
    with open(path, "w") as fh:
        fh.write("#chrom\tstart\tend\trec_rate_per_bp_per_gen\n")
        for s, e, r in cfg.segments():
            c = s // cfg.chrom_len
            fh.write(f"chr{c + 1}\t{s - cfg.chrom_offset(c)}\t{e - cfg.chrom_offset(c)}\t{r:.6g}\n")
