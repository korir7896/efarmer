"""Scan for sweeps against a null simulated under the inferred history.

The whole point of the exercise is that a threshold taken from the genome
itself is wrong twice over.  The bottleneck depresses diversity and skews the
spectrum everywhere, so a constant-size null calls sweeps all over the
genome.  And recombination is not uniform: a cold spot has fewer independent
genealogies per window, so every statistic has heavier tails there.  Measured
on neutral replicates of this genome, a genome-wide 99.9th-percentile cut
fires 7 to 21 times more often inside cold spots than outside.

So the null here is simulated under the fitted history AND with the real
recombination map, and each window is compared against null windows of
comparable local recombination rate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import allel
import msprime
import numpy as np
import tskit

from . import sweepstats as SS

BOUNDARY_RATE = math.log(2.0)

# Statistics used for DETECTION, with the direction that indicates a sweep.
#
# Both directions of Tajima's D are included, which is not a mistake. A hard
# sweep drives it down; a soft sweep drives it UP, because several frequent
# haplotypes leave an excess of intermediate-frequency variants. Measured at
# the sweep, soft replicates sit at +0.44 against background while every
# other class is negative. In a Fisher combination the two directions do not
# double-count: a window extreme in one tail has a p near one in the other
# and contributes nothing from it, so the pair behaves as a two-sided test.
#
# With only four one-sided statistics, soft sweeps contributed to exactly one
# of them: diversity at a soft sweep is 0.80 of background, barely
# depressed; nSL reads zero because the sweep has finished; and Tajima's D
# pointed the wrong way. Two of the three soft sweeps in the shipped
# replicate were missed. H2/H1 is reduced at soft sweeps as well, to about
# 0.6 of background against a null first percentile of 0.48, so it
# contributes too.
DETECT_STATS = (("pi", "low"), ("tajima_d", "low"), ("tajima_d", "high"),
                ("h12", "high"), ("h2_h1", "low"), ("nsl_frac", "high"))


# ---------------------------------------------------------------------------
# recombination map
# ---------------------------------------------------------------------------

@dataclass
class RateMap:
    n_chrom: int
    chrom_len: int
    starts: Dict[int, np.ndarray] = field(default_factory=dict)
    ends: Dict[int, np.ndarray] = field(default_factory=dict)
    rates: Dict[int, np.ndarray] = field(default_factory=dict)

    def mean_rate(self, chrom: int, lo: int, hi: int) -> float:
        s, e, r = self.starts[chrom], self.ends[chrom], self.rates[chrom]
        left = np.maximum(s, lo); right = np.minimum(e, hi)
        w = np.clip(right - left, 0, None)
        tot = w.sum()
        return float((w * r).sum() / tot) if tot else float("nan")

    def median_rate(self) -> float:
        allr, allw = [], []
        for c in range(1, self.n_chrom + 1):
            allr.append(self.rates[c]); allw.append(self.ends[c] - self.starts[c])
        r = np.concatenate(allr); w = np.concatenate(allw)
        order = np.argsort(r)
        r, w = r[order], w[order]
        cw = np.cumsum(w)
        return float(r[np.searchsorted(cw, cw[-1] / 2.0)])

    def to_msprime(self) -> msprime.RateMap:
        """One map over the concatenated genome, with ln(2) at each boundary.

        Under msprime's discrete genome at most one breakpoint falls in a
        one-base interval, so a rate of ln(2) there gives a breakpoint with
        probability exactly 0.5: independent chromosomes.
        """
        pos = [0.0]; rate = []
        for c in range(1, self.n_chrom + 1):
            off = (c - 1) * self.chrom_len
            s, e, r = self.starts[c], self.ends[c], self.rates[c]
            for i in range(len(s)):
                seg_end = off + float(e[i])
                last_of_chrom = (e[i] == self.chrom_len)
                if last_of_chrom and c < self.n_chrom:
                    if seg_end - 1.0 > pos[-1]:
                        pos.append(seg_end - 1.0); rate.append(float(r[i]))
                    pos.append(seg_end); rate.append(BOUNDARY_RATE)
                else:
                    pos.append(seg_end); rate.append(float(r[i]))
        return msprime.RateMap(position=np.array(pos), rate=np.array(rate))


def load_rate_map(path: str) -> RateMap:
    rows: Dict[int, List[Tuple[int, int, float]]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split()
            c = f[0]
            c = int(c[3:]) if c.lower().startswith("chr") else int(c)
            rows.setdefault(c, []).append((int(f[1]), int(f[2]), float(f[3])))
    n_chrom = max(rows)
    chrom_len = max(e for v in rows.values() for _, e, _ in v)
    rm = RateMap(n_chrom=n_chrom, chrom_len=chrom_len)
    for c, v in rows.items():
        v.sort()
        rm.starts[c] = np.array([a for a, _, _ in v], dtype=np.int64)
        rm.ends[c] = np.array([b for _, b, _ in v], dtype=np.int64)
        rm.rates[c] = np.array([r for _, _, r in v], dtype=float)
    return rm


# ---------------------------------------------------------------------------
# windowed scan
# ---------------------------------------------------------------------------

def scan_chrom(cd, chrom_len: int, win: int, step: int) -> Dict[str, np.ndarray]:
    sfs = SS.windowed_sfs_stats(cd, chrom_len, win, step)
    gar = SS.garud_scan_bp(cd, chrom_len, win, step, min_snps=20)
    ns = SS.nsl_scan(cd)
    nsw = SS.windowed_max_abs(ns["pos"], ns["std"], chrom_len, win, step,
                              kind="frac_extreme")
    return {"mid": sfs["mid"], "win": sfs["win"], "pi": sfs["pi"],
            "tajima_d": sfs["tajima_d"], "h12": gar["h12"],
            "h2_h1": gar["h2_h1"], "nsl_frac": nsw["value"],
            "n_snps": gar["n_snps"]}


def scan_genome(chroms: Dict[str, object], chrom_len: int, rm: RateMap,
                win: int, step: int) -> Dict[str, np.ndarray]:
    cols: Dict[str, List[np.ndarray]] = {}
    for name, cd in sorted(chroms.items(), key=lambda kv: int(kv[0][3:])):
        c = int(name[3:])
        sc = scan_chrom(cd, chrom_len, win, step)
        sc["chrom"] = np.full(sc["mid"].size, c)
        sc["rate"] = np.array([rm.mean_rate(c, int(a), int(b))
                               for a, b in sc["win"]])
        for k, v in sc.items():
            if k == "win":
                continue
            cols.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in cols.items()}


def rate_bin(rates: np.ndarray, median: float) -> np.ndarray:
    """Three bins: strongly suppressed, mildly suppressed, ordinary."""
    rel = rates / median
    b = np.full(rel.size, 2, dtype=int)
    b[rel < 0.70] = 1
    b[rel < 0.30] = 0
    return b


# ---------------------------------------------------------------------------
# null
# ---------------------------------------------------------------------------

def simulate_null_genome(hist, rm: RateMap, mu: float, n_dip: int, seed: int):
    ts = msprime.sim_ancestry(samples={"pop": n_dip}, demography=hist.to_msprime(),
                              recombination_rate=rm.to_msprime(),
                              random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=mu, model=msprime.JC69(),
                               discrete_genome=True, random_seed=seed + 1)
    return ts


def ts_to_chroms(ts: tskit.TreeSequence, n_chrom: int, chrom_len: int):
    pos, gen = [], []
    for var in ts.variants():
        p = var.site.position
        if p != int(p):
            continue
        g = var.genotypes
        u = np.unique(g)
        if u.size != 2 or len(var.alleles) > 2:
            continue
        pos.append(int(p)); gen.append(g.astype(np.int8))
    pos = np.asarray(pos, dtype=np.int64)
    G = np.vstack(gen)
    out = {}
    for c in range(n_chrom):
        lo, hi = c * chrom_len, (c + 1) * chrom_len
        m = (pos >= lo) & (pos < hi)
        if not m.any():
            continue
        sub = G[m]
        der = sub.sum(axis=1)
        ac = np.stack([sub.shape[1] - der, der], axis=1)
        out[f"chr{c + 1}"] = SS.ChromData(name=f"chr{c + 1}",
                                          pos=(pos[m] - lo + 1).astype(np.int64),
                                          hap=allel.HaplotypeArray(sub), ac=ac)
    return out


def build_null(hist, rm: RateMap, mu: float, n_dip: int, win: int, step: int,
               n_reps: int, seed: int = 9_100_001, log=print):
    reps = []
    for i in range(n_reps):
        ts = simulate_null_genome(hist, rm, mu, n_dip, seed + 977 * i)
        chroms = ts_to_chroms(ts, rm.n_chrom, rm.chrom_len)
        sc = scan_genome(chroms, rm.chrom_len, rm, win, step)
        # replicate id, so clusters can be formed within a replicate rather
        # than across the concatenation of all of them
        sc["rep"] = np.full(sc["pi"].size, i)
        reps.append(sc)
        log(f"    null replicate {i + 1}/{n_reps}")
    return {k: np.concatenate([r[k] for r in reps]) for k in reps[0]}


# ---------------------------------------------------------------------------
# scoring and peak calling
# ---------------------------------------------------------------------------

def _empirical_p(obs_vals: np.ndarray, null_vals: np.ndarray,
                 direction: str) -> np.ndarray:
    """Rank-based one-sided p-value of each observation against the null.

    Rank-based rather than a z-score. These statistics are strongly skewed,
    so (x - mean) / sd is not on a calibrated scale: taking the largest z
    across four of them and thresholding for a fixed false-positive budget
    put the cut at z = 10, which only the sharpest sweeps clear. Both old
    sweeps in the shipped replicate were missed that way despite having
    diversity at 0.21 of background and Tajima's D at -2.1.
    """
    nv = null_vals[np.isfinite(null_vals)]
    if nv.size < 20:
        return np.full(obs_vals.size, np.nan)
    srt = np.sort(nv)
    p = np.full(obs_vals.size, np.nan)
    ok = np.isfinite(obs_vals)
    if direction == "low":
        # how many null values are <= each observation
        k = np.searchsorted(srt, obs_vals[ok], side="right")
    else:
        k = nv.size - np.searchsorted(srt, obs_vals[ok], side="left")
    p[ok] = (1.0 + k) / (1.0 + nv.size)
    return p


def sweep_scores(obs: Dict[str, np.ndarray], null: Dict[str, np.ndarray],
                 median_rate: float) -> np.ndarray:
    """Per-window sweep evidence, combining statistics by Fisher's method.

    Each statistic contributes a rate-matched rank-based p-value and the
    combination is -2 sum log p. Combining beats taking the strongest single
    signal: a sweep class that only one statistic responds to still gets a
    high score, while a window that is mildly unusual on three statistics is
    no longer dismissed. A statistic pointing the wrong way contributes a
    p near one and so adds nothing, rather than dragging the score down --
    which matters for soft sweeps, where Tajima's D moves opposite to the
    direction a sweep scan expects.

    The threshold for this score is taken from the null itself
    (null_score_threshold), so the false-positive budget stays exact
    regardless of how the statistics are distributed or correlated.
    """
    ob = rate_bin(obs["rate"], median_rate)
    nb = rate_bin(null["rate"], median_rate)
    total = np.zeros(obs["pi"].size)
    counted = np.zeros(obs["pi"].size)
    for stat, direction in DETECT_STATS:
        o = obs[stat].astype(float)
        p = np.full(o.size, np.nan)
        for b in np.unique(ob):
            sel = ob == b
            nvals = null[stat].astype(float)[nb == b]
            if nvals[np.isfinite(nvals)].size < 200:
                nvals = null[stat].astype(float)
            p[sel] = _empirical_p(o[sel], nvals, direction)
        good = np.isfinite(p) & (p > 0)
        total[good] += -2.0 * np.log(p[good])
        counted[good] += 1
    # windows where nothing could be evaluated get no evidence
    total[counted == 0] = -np.inf
    return total


def null_score_threshold(null: Dict[str, np.ndarray], median_rate: float,
                         n_reps: int, expected_false: float = 0.3) -> float:
    """Threshold whose expected number of null windows per genome is small.

    Taken from the null itself rather than from a distributional assumption,
    so the false-positive budget holds whatever shape the combined score has
    and however correlated the statistics are.
    """
    self_scores = sweep_scores(null, null, median_rate)
    finite = self_scores[np.isfinite(self_scores)]
    if finite.size == 0:
        return float("inf")
    per_genome = finite.size / max(1, n_reps)
    q = 1.0 - (expected_false / per_genome)
    q = min(max(q, 0.5), 1.0 - 1e-9)
    return float(np.quantile(finite, q))


# ---------------------------------------------------------------------------
# cluster-level inference
# ---------------------------------------------------------------------------
#
# Thresholding single windows is the wrong test for this signal. A sweep
# footprint here spans one to two megabases, which is a hundred to two
# hundred windows at a 10 kb step, each moderately elevated. Asking any one
# of them to clear a cut set for 0.3 expected false windows per genome puts
# that cut at the 99.993rd percentile of the null, and measured on the
# shipped replicate the three soft sweeps scored 49.3, 55.4 and 55.9 against
# a threshold of 56.44 -- real signal, two of them missing by a single point,
# against a genome-wide median of 12.7.
#
# So clusters are formed at a mild per-window threshold and judged by their
# total mass, the summed excess over that threshold. A broad, consistently
# elevated footprint accumulates far more mass than the short runs that noise
# produces. The mass cut is calibrated by applying the identical procedure to
# each null replicate, so the false-positive budget stays exact and is now
# a budget on CLUSTERS, which is what a call actually is.


def cluster_runs(chrom: np.ndarray, mid: np.ndarray, score: np.ndarray,
                 form_thresh: float, max_gap_windows: int = 10
                 ) -> List[np.ndarray]:
    """Contiguous runs of windows above the cluster-forming threshold."""
    runs: List[np.ndarray] = []
    for c in np.unique(chrom):
        sel = np.where(chrom == c)[0]
        order = sel[np.argsort(mid[sel])]
        if order.size == 0:
            continue
        steps = np.diff(np.sort(mid[order]))
        step = float(np.median(steps)) if steps.size else 1.0
        hot = [i for i in order if np.isfinite(score[i]) and score[i] >= form_thresh]
        if not hot:
            continue
        cur = [hot[0]]
        for i in hot[1:]:
            if (mid[i] - mid[cur[-1]]) <= (max_gap_windows + 1) * step:
                cur.append(i)
            else:
                runs.append(np.array(cur)); cur = [i]
        runs.append(np.array(cur))
    return runs


def cluster_mass(score: np.ndarray, idx: np.ndarray, form_thresh: float) -> float:
    v = score[idx]
    v = v[np.isfinite(v)]
    return float(np.sum(v - form_thresh)) if v.size else 0.0


def calibrate_clusters(null: Dict[str, np.ndarray], null_score: np.ndarray,
                       form_thresh: float, n_reps: int,
                       expected_false: float = 0.3,
                       max_gap_windows: int = 10) -> float:
    """Cluster-mass cut whose expected count per genome is `expected_false`."""
    masses: List[float] = []
    reps = null.get("rep")
    rep_ids = np.unique(reps) if reps is not None else np.array([0])
    for r in rep_ids:
        m = (reps == r) if reps is not None else np.ones(null_score.size, bool)
        idx_map = np.where(m)[0]
        runs = cluster_runs(null["chrom"][m], null["mid"][m], null_score[m],
                            form_thresh, max_gap_windows)
        for run in runs:
            masses.append(cluster_mass(null_score, idx_map[run], form_thresh))
    if not masses:
        return 0.0
    masses = np.sort(np.array(masses))[::-1]
    k = int(round(expected_false * len(rep_ids)))
    if k <= 0:
        return float(masses[0] * 1.000001)
    if k >= masses.size:
        return 0.0
    return float(masses[k - 1])


def call_clusters(obs: Dict[str, np.ndarray], score: np.ndarray,
                  form_thresh: float, mass_thresh: float,
                  max_gap_windows: int = 10, min_separation: int = 1_000_000
                  ) -> List[Dict[str, object]]:
    """Clusters whose mass clears the calibrated cut, one call each."""
    regions: List[Dict[str, object]] = []
    for run in cluster_runs(obs["chrom"], obs["mid"], score, form_thresh,
                            max_gap_windows):
        if cluster_mass(score, run, form_thresh) < mass_thresh:
            continue
        regions.append(_summarize(obs, score, list(run),
                                  int(obs["chrom"][run[0]])))
    regions.sort(key=lambda r: (r["chrom"], r["position"]))
    merged: List[Dict[str, object]] = []
    for r in regions:
        if merged and merged[-1]["chrom"] == r["chrom"] and \
                r["position"] - merged[-1]["position"] < min_separation:
            if r["peak_score"] > merged[-1]["peak_score"]:
                merged[-1] = r
            continue
        merged.append(r)
    return merged


def call_regions(obs: Dict[str, np.ndarray], score: np.ndarray, thresh: float,
                 max_gap_windows: int = 10, min_separation: int = 1_000_000
                 ) -> List[Dict[str, object]]:
    """Cluster windows over threshold into regions, one call per region.

    The gap and separation are generous because a sweep footprint here is one
    to two megabases and its evidence is not monotonic across it: with a
    30 kb gap tolerance and a 400 kb merge radius, single sweeps were split
    into two calls 500 kb apart, which cost both a false positive and, when
    the stronger half sat off-centre, the match itself. True sweeps are at
    least 2 Mb apart, so merging within 1 Mb cannot join two of them.
    """
    regions: List[Dict[str, object]] = []
    for c in np.unique(obs["chrom"]):
        sel = np.where(obs["chrom"] == c)[0]
        order = sel[np.argsort(obs["mid"][sel])]
        hot = [i for i in order if np.isfinite(score[i]) and score[i] >= thresh]
        if not hot:
            continue
        cluster = [hot[0]]
        mids = obs["mid"]
        step = float(np.median(np.diff(np.sort(mids[order])))) if order.size > 1 else 1.0
        for i in hot[1:]:
            if (mids[i] - mids[cluster[-1]]) <= (max_gap_windows + 1) * step:
                cluster.append(i)
            else:
                regions.append(_summarize(obs, score, cluster, int(c)))
                cluster = [i]
        regions.append(_summarize(obs, score, cluster, int(c)))

    regions.sort(key=lambda r: (r["chrom"], r["position"]))
    merged: List[Dict[str, object]] = []
    for r in regions:
        if merged and merged[-1]["chrom"] == r["chrom"] and \
                r["position"] - merged[-1]["position"] < min_separation:
            if r["peak_score"] > merged[-1]["peak_score"]:
                merged[-1] = r
            continue
        merged.append(r)
    return merged


def _summarize(obs, score, idx: Sequence[int], chrom: int) -> Dict[str, object]:
    idx = list(idx)
    best = max(idx, key=lambda i: score[i])
    return {
        "chrom": chrom,
        "position": int(round(float(obs["mid"][best]))),
        "peak_score": float(score[best]),
        "window_indices": idx,
        "peak_index": int(best),
        "span": (float(obs["mid"][min(idx)]), float(obs["mid"][max(idx)])),
    }
