"""Infer the bottleneck-and-expansion history from the site frequency spectrum.

The history has to be inferred before any sweep can be called, because the
threshold a scan statistic must clear depends on it: a bottleneck lowers
diversity and skews the spectrum genome-wide, which is exactly what a sweep
does locally.  A scan calibrated against a constant-size null reports the
bottleneck as sweeps.

Method: simulated method of moments.  Propose a history, simulate it with
msprime, compare the simulated binned spectrum and mean diversity with the
observed ones, and search for the history that matches.  Every proposal uses
the same random seed (common random numbers), which makes the objective a
deterministic function of the parameters instead of a noisy one, so a plain
local search converges.

Recombination is ignored in the fitting simulations: the spectrum and mean
diversity are single-site quantities, so a uniform rate is enough and is much
cheaper than carrying the real map.  The real map matters for the null used
in detection, which is built separately in detect.py.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import allel
import msprime
import numpy as np

# Bin edges over derived allele count, as fractions of the number of
# haplotypes. Log-spaced: the low-frequency end carries most of the
# information about recent size change, so it gets the finest bins.
BIN_FRACTIONS = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.11,
                          0.15, 0.20, 0.27, 0.36, 0.48, 0.64, 0.82, 1.0])


@dataclass
class History:
    n_ancestral: float
    n_bottleneck: float
    n_current: float
    t_start: float          # generations ago the bottleneck began
    t_end: float            # generations ago it ended (the expansion)

    def as_dict(self):
        return {
            "N_ancestral": float(self.n_ancestral),
            "N_bottleneck": float(self.n_bottleneck),
            "N_current": float(self.n_current),
            "T_bottleneck_start_gens_ago": float(self.t_start),
            "T_bottleneck_end_gens_ago": float(self.t_end),
        }

    def to_msprime(self) -> msprime.Demography:
        d = msprime.Demography()
        d.add_population(name="pop", initial_size=self.n_current)
        d.add_population_parameters_change(time=self.t_end,
                                           initial_size=self.n_bottleneck,
                                           population="pop")
        d.add_population_parameters_change(time=self.t_start,
                                           initial_size=self.n_ancestral,
                                           population="pop")
        return d


def binned_sfs(ac_derived: np.ndarray, n_hap: int) -> np.ndarray:
    """Normalized spectrum of derived allele counts, in BIN_FRACTIONS bins."""
    edges = np.unique(np.clip((BIN_FRACTIONS * n_hap).round().astype(int), 0, n_hap))
    counts, _ = np.histogram(ac_derived, bins=edges)
    tot = counts.sum()
    return counts / tot if tot else counts.astype(float)


def observed_summaries(chroms: Dict[str, "object"], chrom_len: int,
                       win: int = 50_000, step: int = 50_000,
                       drop_low_pi_frac: float = 0.15) -> Dict[str, object]:
    """Spectrum and mean diversity over windows that are unlikely to be swept.

    Sweeps depress diversity and skew the spectrum locally.  Fitting a neutral
    history to the whole genome therefore absorbs some of the selection into
    the demography.  Dropping the lowest-diversity windows removes most of
    that; the fit is then refined once sweeps have actually been located.
    """
    from . import sweepstats as SS

    pis: List[np.ndarray] = []
    per_window_ac: List[List[np.ndarray]] = []
    n_hap = None
    for name, cd in sorted(chroms.items()):
        n_hap = cd.n_hap
        st = SS.windowed_sfs_stats(cd, chrom_len, win, step)
        pis.append(st["pi"])
        lo = np.searchsorted(cd.pos, st["win"][:, 0], side="left")
        hi = np.searchsorted(cd.pos, st["win"][:, 1], side="right")
        per_window_ac.append([cd.ac[a:b, 1] for a, b in zip(lo, hi)])

    pi_all = np.concatenate(pis)
    finite = np.isfinite(pi_all)
    if drop_low_pi_frac > 0 and finite.sum() > 20:
        cut = np.nanquantile(pi_all[finite], drop_low_pi_frac)
    else:
        cut = -np.inf

    keep_ac: List[np.ndarray] = []
    keep_pi: List[float] = []
    k = 0
    for block in per_window_ac:
        for ac in block:
            p = pi_all[k]; k += 1
            if not np.isfinite(p) or p <= cut:
                continue
            keep_pi.append(p)
            if ac.size:
                keep_ac.append(ac)
    ac_cat = np.concatenate(keep_ac) if keep_ac else np.array([], dtype=int)
    return {"sfs": binned_sfs(ac_cat, n_hap),
            "pi": float(np.mean(keep_pi)) if keep_pi else float("nan"),
            "n_hap": int(n_hap),
            "n_snps": int(ac_cat.size)}


def bin_expected(afs: np.ndarray, n_hap: int) -> np.ndarray:
    """Bin an expected (branch-length) spectrum with the same edges as the data."""
    edges = np.unique(np.clip((BIN_FRACTIONS * n_hap).round().astype(int), 0, n_hap))
    inner = afs[1:n_hap]                       # drop the 0 and n classes
    counts = np.zeros(len(edges) - 1)
    idx = np.arange(1, n_hap)
    for b in range(len(edges) - 1):
        m = (idx >= max(edges[b], 1)) & (idx < edges[b + 1])
        counts[b] = float(inner[m].sum()) if m.any() else 0.0
    tot = counts.sum()
    return counts / tot if tot else counts


def simulate_summaries(hist: History, mu: float, rate: float, n_dip: int,
                       length: int, seed: int, n_trees: int = 1500
                       ) -> Dict[str, object]:
    """Expected spectrum and diversity for a proposed history.

    Averaged over many INDEPENDENT coalescent trees, using tskit's
    branch-length statistics rather than simulated mutations.

    Both choices remove noise that was otherwise swamping the signal. The
    expected site spectrum is the mutation rate times the expected branch
    length subtending each allele count, so placing mutations only adds
    sampling noise to a quantity whose expectation is already available. And
    the spectrum is a single-site statistic, so it does not depend on
    recombination: simulating one long recombining sequence just produces
    many correlated trees, where the same compute spent on independent
    unlinked trees converges far faster.

    With one 4 Mb recombining sequence and simulated mutations the objective
    was rugged enough to be useless -- on a neutral replicate the true
    history scored 0.0223 while a history with the bottleneck more than twice
    as old scored 0.0125. `length` and `rate` are accepted and ignored, so
    callers do not have to change.
    """
    reps = msprime.sim_ancestry(samples={"pop": n_dip},
                                demography=hist.to_msprime(),
                                sequence_length=1, recombination_rate=0,
                                num_replicates=n_trees, random_seed=seed)
    afs_sum = None
    pi_sum = 0.0
    n_hap = 2 * n_dip
    for ts in reps:
        a = ts.allele_frequency_spectrum(mode="branch", polarised=True,
                                         span_normalise=True)
        afs_sum = a if afs_sum is None else afs_sum + a
        pi_sum += ts.diversity(mode="branch")
    afs = np.asarray(afs_sum, float) / n_trees
    pi = (pi_sum / n_trees) * mu
    return {"sfs": bin_expected(afs, n_hap), "pi": float(pi), "n_hap": n_hap}


def distance(obs: Dict[str, object], sim: Dict[str, object],
             pi_weight: float = 8.0) -> float:
    o = np.asarray(obs["sfs"], float); s = np.asarray(sim["sfs"], float)
    m = o + s > 0
    chi = float(np.sum((s[m] - o[m]) ** 2 / (o[m] + 1e-6)))
    if not (np.isfinite(obs["pi"]) and np.isfinite(sim["pi"])) \
            or obs["pi"] <= 0 or sim["pi"] <= 0:
        return float("inf")
    lp = float(np.log(sim["pi"] / obs["pi"])) ** 2
    return chi + pi_weight * lp


def _propose(rng: np.random.Generator, n: int) -> List[History]:
    """Random histories, drawn log-uniformly over generous ranges."""
    out = []
    for _ in range(n):
        n_anc = float(np.exp(rng.uniform(np.log(2_000), np.log(60_000))))
        n_cur = float(np.exp(rng.uniform(np.log(2_000), np.log(60_000))))
        n_bot = float(np.exp(rng.uniform(np.log(80), np.log(10_000))))
        n_bot = min(n_bot, 0.9 * min(n_anc, n_cur))
        t_start = float(np.exp(rng.uniform(np.log(600), np.log(12_000))))
        t_end = t_start * float(rng.uniform(0.15, 0.95))
        out.append(History(n_anc, n_bot, n_cur, t_start, t_end))
    return out


def _jitter(h: History, rng: np.random.Generator, scale: float) -> History:
    def lg(v, lo, hi):
        return float(np.clip(v * np.exp(rng.normal(0, scale)), lo, hi))
    n_anc = lg(h.n_ancestral, 1_000, 80_000)
    n_cur = lg(h.n_current, 1_000, 80_000)
    n_bot = lg(h.n_bottleneck, 50, 20_000)
    n_bot = min(n_bot, 0.9 * min(n_anc, n_cur))
    t_start = lg(h.t_start, 300, 20_000)
    t_end = float(np.clip(h.t_end * np.exp(rng.normal(0, scale)),
                          0.05 * t_start, 0.97 * t_start))
    return History(n_anc, n_bot, n_cur, t_start, t_end)


def fit_demography(obs: Dict[str, object], mu: float, rate: float,
                   n_dip: int, seed: int = 20_240_101,
                   n_coarse: int = 240, n_refine_rounds: int = 7,
                   n_refine: int = 36, length: int = 20_000_000,
                   log=print) -> Tuple[History, float]:
    rng = np.random.default_rng(seed)
    sim_seed = seed + 7           # common random numbers across all proposals

    def score(h: History) -> float:
        try:
            sim = simulate_summaries(h, mu, rate, n_dip, length, sim_seed)
        except Exception:
            return float("inf")
        return distance(obs, sim)

    cands = _propose(rng, n_coarse)
    scored = [(score(h), h) for h in cands]
    scored.sort(key=lambda x: x[0])
    log(f"    coarse search: best distance {scored[0][0]:.4f}")

    best_d, best_h = scored[0]
    pool = [h for _, h in scored[:4]]
    scale = 0.45
    for r in range(n_refine_rounds):
        trials = []
        per = max(1, n_refine // max(1, len(pool)))
        for h in pool:
            trials += [_jitter(h, rng, scale) for _ in range(per)]
        ranked = sorted(((score(h), h) for h in trials), key=lambda x: x[0])
        if ranked and ranked[0][0] < best_d:
            best_d, best_h = ranked[0]
        # Keep a few distinct good candidates rather than collapsing onto one:
        # the objective has a ridge (a short severe bottleneck resembles a
        # longer milder one), and a single-point pool slides along it.
        pool = [best_h] + [h for _, h in ranked[:3]]
        scale *= 0.68
        log(f"    refine round {r + 1}: best distance {best_d:.4f}  "
            f"N_anc={best_h.n_ancestral:.0f} N_bot={best_h.n_bottleneck:.0f} "
            f"N_cur={best_h.n_current:.0f} T=[{best_h.t_end:.0f},{best_h.t_start:.0f}]")

    # Final polish: one parameter at a time, which the random jitter above
    # does poorly once the scale is small.
    for _ in range(2):
        for field in ("n_ancestral", "n_bottleneck", "n_current", "t_start", "t_end"):
            for factor in (0.82, 0.91, 1.10, 1.22):
                cand = History(**asdict(best_h))
                setattr(cand, field, getattr(cand, field) * factor)
                if cand.t_end >= cand.t_start or cand.n_bottleneck <= 0:
                    continue
                if cand.n_bottleneck >= min(cand.n_ancestral, cand.n_current):
                    continue
                d = score(cand)
                if d < best_d:
                    best_d, best_h = d, cand
    log(f"    after polish: distance {best_d:.4f}  "
        f"N_anc={best_h.n_ancestral:.0f} N_bot={best_h.n_bottleneck:.0f} "
        f"N_cur={best_h.n_current:.0f} T=[{best_h.t_end:.0f},{best_h.t_start:.0f}]")
    return best_h, best_d
