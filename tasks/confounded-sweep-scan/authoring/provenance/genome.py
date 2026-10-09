"""Genome architecture: chromosomes, recombination map with cold spots.

Shared by the generator (authoring) and exported to the agent as a BED-like
rate map. Nothing here encodes selected-site locations.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Tuple


@dataclass(frozen=True)
class ColdSpot:
    """A low-recombination window, given as (chrom_index, start, end, factor)."""
    chrom: int
    start: int          # relative to chromosome start
    end: int
    factor: float       # local rate = base_rate * factor


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
    """Three cold spots, on three different chromosomes, away from the edges.

    Chromosome 3 deliberately has none, so the agent cannot assume one per
    chromosome. Widths differ so a single hard-coded width does not find them.
    """
    return [
        ColdSpot(chrom=0, start=int(0.30 * chrom_len), end=int(0.30 * chrom_len) + 450_000, factor=0.1),
        ColdSpot(chrom=1, start=int(0.62 * chrom_len), end=int(0.62 * chrom_len) + 300_000, factor=0.08),
        ColdSpot(chrom=2, start=int(0.45 * chrom_len), end=int(0.45 * chrom_len) + 600_000, factor=0.12),
    ]


def write_rate_map_bed(cfg: GenomeConfig, path: str) -> None:
    """Agent-facing recombination map. Plain BED4: chrom start end rate."""
    with open(path, "w") as fh:
        fh.write("#chrom\tstart\tend\trec_rate_per_bp_per_gen\n")
        for s, e, r in cfg.segments():
            c = s // cfg.chrom_len
            fh.write(f"chr{c + 1}\t{s - cfg.chrom_offset(c)}\t{e - cfg.chrom_offset(c)}\t{r:.6g}\n")
