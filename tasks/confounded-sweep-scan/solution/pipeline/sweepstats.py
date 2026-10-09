"""Scan statistics over a phased VCF.

Deliberately kept to numpy + scikit-allel so the same module can run inside
the task environment and inside the reference solution with pinned, ordinary
dependencies.

Nothing here knows where the sweeps are.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import allel
import numpy as np


@dataclass
class ChromData:
    name: str
    pos: np.ndarray             # 1-based positions within the chromosome
    hap: allel.HaplotypeArray   # (n_variants, n_haplotypes)
    ac: np.ndarray              # (n_variants, 2) allele counts, col 1 = derived

    @property
    def n_hap(self) -> int:
        return self.hap.shape[1]

    @property
    def daf(self) -> np.ndarray:
        return self.ac[:, 1] / self.n_hap


def load_vcf(path: str) -> Dict[str, ChromData]:
    """Load a phased biallelic VCF into per-chromosome haplotype arrays.

    REF is the ancestral allele by construction (the generator writes AA=REF),
    so allele 1 is the derived allele and no extra polarization is needed.
    """
    cb = allel.read_vcf(path, fields=["variants/CHROM", "variants/POS",
                                      "calldata/GT"])
    chroms = cb["variants/CHROM"]
    pos = cb["variants/POS"]
    gt = allel.GenotypeArray(cb["calldata/GT"])
    out: Dict[str, ChromData] = {}
    for name in sorted(set(chroms.tolist()), key=lambda x: (len(x), x)):
        m = chroms == name
        g = gt[m]
        h = g.to_haplotypes()
        ac = g.count_alleles(max_allele=1)
        out[name] = ChromData(name=name, pos=pos[m].astype(np.int64),
                              hap=h, ac=np.asarray(ac))
    return out


# ---------------------------------------------------------------------------
# window helpers
# ---------------------------------------------------------------------------

def bp_windows(chrom_len: int, size: int, step: int) -> np.ndarray:
    starts = np.arange(1, max(2, chrom_len - size + 2), step)
    return np.stack([starts, starts + size - 1], axis=1)


def window_mids(win: np.ndarray) -> np.ndarray:
    return win.mean(axis=1)


# ---------------------------------------------------------------------------
# site-frequency statistics
# ---------------------------------------------------------------------------

def windowed_sfs_stats(cd: ChromData, chrom_len: int, size: int, step: int
                       ) -> Dict[str, np.ndarray]:
    win = bp_windows(chrom_len, size, step)
    pi, _, nb, _ = allel.windowed_diversity(cd.pos, cd.ac, windows=win)
    theta_w, _, _, _ = allel.windowed_watterson_theta(cd.pos, cd.ac, windows=win)
    with np.errstate(invalid="ignore", divide="ignore"):
        tajd, _, _ = allel.windowed_tajima_d(cd.pos, cd.ac, windows=win,
                                             min_sites=8)
    return {"mid": window_mids(win), "win": win, "pi": np.asarray(pi, float),
            "theta_w": np.asarray(theta_w, float),
            "tajima_d": np.asarray(tajd, float), "n_sites": np.asarray(nb, float)}


# ---------------------------------------------------------------------------
# haplotype statistics
# ---------------------------------------------------------------------------

def garud_scan(cd: ChromData, win_snps: int, step_snps: int
               ) -> Dict[str, np.ndarray]:
    """H1/H12/H123/H2H1 in sliding SNP windows, plus window midpoints in bp."""
    n = cd.hap.shape[0]
    if n < win_snps:
        empty = np.array([])
        return {k: empty for k in ("mid", "h1", "h12", "h123", "h2_h1")}
    starts = np.arange(0, n - win_snps + 1, step_snps)
    h1 = np.empty(starts.size); h12 = np.empty(starts.size)
    h123 = np.empty(starts.size); h21 = np.empty(starts.size)
    mid = np.empty(starts.size)
    H = cd.hap
    for i, s0 in enumerate(starts):
        sub = H[s0:s0 + win_snps]
        a, b, c, d = allel.garud_h(sub)
        h1[i], h12[i], h123[i], h21[i] = a, b, c, d
        mid[i] = 0.5 * (cd.pos[s0] + cd.pos[s0 + win_snps - 1])
    return {"mid": mid, "h1": h1, "h12": h12, "h123": h123, "h2_h1": h21}


def garud_scan_bp(cd: ChromData, chrom_len: int, size: int, step: int,
                  min_snps: int = 20) -> Dict[str, np.ndarray]:
    """H1/H12/H123/H2H1 in fixed PHYSICAL windows.

    SNP-count windows are the classical choice, but SNP density collapses
    inside a sweep -- by a factor of ~25 in this model -- so a fixed-count
    window centred on a sweep spans megabases and averages the haplotype
    signal away.  Physical windows keep the footprint fixed; windows with too
    few SNPs to resolve haplotypes are returned as NaN rather than as a
    spuriously high homozygosity.
    """
    win = bp_windows(chrom_len, size, step)
    mids = window_mids(win)
    left = np.searchsorted(cd.pos, win[:, 0], side="left")
    right = np.searchsorted(cd.pos, win[:, 1], side="right")
    h1 = np.full(mids.size, np.nan); h12 = np.full(mids.size, np.nan)
    h123 = np.full(mids.size, np.nan); h21 = np.full(mids.size, np.nan)
    nsnp = np.zeros(mids.size)
    H = cd.hap
    for i in range(mids.size):
        a, b = int(left[i]), int(right[i])
        nsnp[i] = b - a
        if b - a < min_snps:
            continue
        x, y, z, w = allel.garud_h(H[a:b])
        h1[i], h12[i], h123[i], h21[i] = x, y, z, w
    return {"mid": mids, "win": win, "h1": h1, "h12": h12, "h123": h123,
            "h2_h1": h21, "n_snps": nsnp}


def nsl_scan(cd: ChromData, n_bins: int = 20, min_per_bin: int = 25
             ) -> Dict[str, np.ndarray]:
    """Standardized nSL per variant (positive = long derived haplotypes)."""
    raw = allel.nsl(cd.hap, use_threads=False)
    aac = cd.ac[:, 1]
    with np.errstate(invalid="ignore"):
        std = allel.standardize_by_allele_count(raw, aac, n_bins=n_bins,
                                                diagnostics=False)
    std = np.asarray(std[0] if isinstance(std, tuple) else std, dtype=float)
    return {"pos": cd.pos, "raw": np.asarray(raw, float), "std": std,
            "daf": cd.daf}


def windowed_max_abs(pos: np.ndarray, score: np.ndarray, chrom_len: int,
                     size: int, step: int, kind: str = "max") -> Dict[str, np.ndarray]:
    """Summarize a per-variant score into bp windows."""
    win = bp_windows(chrom_len, size, step)
    mids = window_mids(win)
    out = np.full(mids.size, np.nan)
    ok = np.isfinite(score)
    p = pos[ok]; v = score[ok]
    left = np.searchsorted(p, win[:, 0], side="left")
    right = np.searchsorted(p, win[:, 1], side="right")
    for i in range(mids.size):
        seg = v[left[i]:right[i]]
        if seg.size == 0:
            continue
        if kind == "max":
            out[i] = seg.max()
        elif kind == "frac_extreme":
            out[i] = float(np.mean(np.abs(seg) > 2.0))
        else:
            out[i] = seg.mean()
    return {"mid": mids, "win": win, "value": out}
