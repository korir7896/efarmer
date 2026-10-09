"""Forward simulation of the sweep zoo with fwdpy11, with trajectory recording.

Design
------
The forward phase covers only the recent epoch in which selection acts
(constant size N_cur).  All older history -- the expansion, the bottleneck and
the ancestral epoch -- is supplied afterwards by msprime recapitation, which
is exact and far cheaper than forward-simulating it.  Every sweep in the zoo
is younger than the start of the forward phase.

Selected mutations never arise de novo: the selected mutation rate is zero and
`sregions` is empty.  Every selected variant is introduced by hand at a
scheduled generation and position, so ground truth is the generator's own
schedule plus the realized trajectory it records, never a solver's output.

Hard versus soft
----------------
A sweep is introduced as `n_origins` INDEPENDENT singletons at the same
position, each landing on whatever haplotype background it finds.  One origin
is a hard sweep.  Many origins is a soft sweep from recurrent mutation, and
it is soft in the way that matters: the beneficial allele sits on several
distinct backgrounds, so the swept region carries several frequent haplotypes
rather than one.

Introducing the allele on a single node with many descendants, which is the
obvious alternative, does NOT produce this.  All those carriers descend from
one ancestor and share its haplotype over roughly 1/(r*T) base pairs, with
T ~ 2*N*f the ancestor's age.  At f = 24/18000 that is about 4 Mb, wider than
the sweep footprint, so the result is indistinguishable from a hard sweep --
measured H2/H1 of 0.034 against 0.022 for hard sweeps.  fwdpy11 also refuses
descendant counts above about 100, since no node has exactly that many.

`fwdpy11.evolvets` forbids `add_mutation` from inside a recorder, so the run
is segmented: evolve to a scheduled origin, deep-copy the population, add the
mutations, evolve a short establishment window, and either accept or restore
the copy and retry.  This conditions each sweep on non-loss, which is the
standard way to simulate a sweep that is known to have happened.

Ground truth is the REALIZED outcome, not the intended one.  Replicate-level
acceptance is applied in sweeps.py.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import fwdpy11

from genome import GenomeConfig


# --------------------------------------------------------------------------
# plan / realization records
# --------------------------------------------------------------------------

@dataclass
class SweepPlan:
    """One scheduled selected site."""
    sweep_id: str
    position: int             # global bp coordinate
    s: float                  # per-allele selective advantage (additive)
    h: float                  # fwdpy11 dominance; 1.0 gives 1, 1+s, 1+2s
    origin_gens_ago: int      # when to introduce, in generations before present
    n_origins: int            # independent introductions: 1 hard, many soft
    intended_class: str


@dataclass
class SweepRealization:
    """What actually happened at a scheduled site."""
    sweep_id: str
    position: int
    s: float
    h: float
    n_origins_introduced: int
    intended_class: str
    n_origins_surviving: int = 0
    origin_gen_forward: int = -1
    origin_gens_ago: int = -1
    establishment_attempts: int = 0
    established: bool = False
    fixed: bool = False
    fixation_gen_forward: int = -1
    fixation_gens_ago: int = -1
    final_count: int = 0
    final_freq: float = 0.0
    max_freq: float = 0.0
    lost: bool = False
    trajectory: Dict[int, int] = field(default_factory=dict)

    def to_dict(self, include_trajectory: bool = True):
        d = asdict(self)
        if include_trajectory:
            d["trajectory"] = {str(k): int(v) for k, v in sorted(self.trajectory.items())}
        else:
            d.pop("trajectory")
        return d


# --------------------------------------------------------------------------
# trajectory recorder
# --------------------------------------------------------------------------

class TrajectoryRecorder:
    """Tracks each sweep's total frequency and its number of living origins.

    A sweep is a SET of mutation keys.  Mutations are identified by key tuple,
    not table index, because simplification remaps indices.  With no neutral
    mutations in the forward phase `pop.mutations` holds only the selected
    variants, so the per-generation scan is cheap.
    """

    def __init__(self, two_n: int, total_gens: int, traj_thin: int = 1):
        self.two_n = two_n
        self.total_gens = total_gens
        self.traj_thin = traj_thin
        self.tracked: Dict[str, List[tuple]] = {}
        self.real: Dict[str, SweepRealization] = {}

    def track(self, sweep_id: str, keys: Sequence[tuple], real: SweepRealization):
        self.tracked[sweep_id] = list(keys)
        self.real[sweep_id] = real

    @staticmethod
    def _count_for_key(pop, key) -> Tuple[int, bool]:
        for i, m in enumerate(pop.mutations):
            if m.key == key:
                c = int(pop.mcounts[i]) if i < len(pop.mcounts) else 0
                if c > 0:
                    return c, False
                break
        # NOTE: find_fixation_by_key returns -1 (not None) when absent.
        if pop.find_fixation_by_key(key) >= 0:
            return -1, True
        return 0, False

    def _sweep_state(self, pop, keys) -> Tuple[int, bool, int]:
        """(total allele count, whether fixed, number of surviving origins)."""
        total = 0
        surviving = 0
        fixed = False
        for k in keys:
            c, is_fixed = self._count_for_key(pop, k)
            if is_fixed:
                fixed = True
                surviving += 1
            elif c > 0:
                total += c
                surviving += 1
        if fixed or total >= self.two_n:
            return self.two_n, True, surviving
        return total, False, surviving

    def snapshot(self, pop) -> Dict[str, Tuple[int, bool, int]]:
        return {sid: self._sweep_state(pop, ks) for sid, ks in self.tracked.items()}

    def __call__(self, pop, sampler) -> None:
        gen = pop.generation
        for sid, keys in self.tracked.items():
            r = self.real[sid]
            total, fixed, surviving = self._sweep_state(pop, keys)
            if surviving:
                r.n_origins_surviving = surviving
            if fixed and not r.fixed:
                r.fixed = True
                r.established = True
                r.fixation_gen_forward = gen
                r.fixation_gens_ago = self.total_gens - gen
            if total == 0 and r.established and not r.fixed:
                r.lost = True
            r.final_count = self.two_n if r.fixed else total
            r.final_freq = r.final_count / self.two_n
            r.max_freq = max(r.max_freq, r.final_freq)
            if self.traj_thin == 1 or gen % self.traj_thin == 0 or gen == self.total_gens:
                r.trajectory[gen] = r.final_count


# --------------------------------------------------------------------------
# model assembly
# --------------------------------------------------------------------------

def est_threshold(self_s: float) -> int:
    """Allele count at which a beneficial variant has escaped drift.

    Under the diffusion approximation a beneficial allele at count k fixes
    with probability about 1 - exp(-2ks), so k = 2/s gives 2ks = 4 and a
    post-establishment loss probability near 2%.
    """
    return max(20, int(math.ceil(2.0 / self_s)))


def est_window(self_s: float) -> int:
    """Generations a logistic rise needs to reach est_threshold, with slack."""
    return max(60, int(math.ceil(2.0 * math.log(est_threshold(self_s)) / self_s)))


def constant_size_yaml(n: int) -> str:
    return f"""
time_units: generations
demes:
  - name: pop
    epochs:
      - start_size: {n}
"""


def build_recregions(cfg: GenomeConfig):
    """Poisson crossover intervals per rate segment; 50% recombination between
    chromosomes."""
    regions = []
    for s, e, rate in cfg.segments():
        mean = rate * (e - s)
        if mean > 0:
            regions.append(fwdpy11.PoissonInterval(int(s), int(e), mean, discrete=True))
    for c in range(1, cfg.n_chrom):
        regions.append(fwdpy11.BinomialPoint(cfg.chrom_offset(c), 0.5, discrete=True))
    return regions


def make_params(cfg: GenomeConfig, n_cur: int, total_gens: int, simlen: int):
    demog = fwdpy11.ForwardDemesGraph.from_demes(
        constant_size_yaml(n_cur), burnin=total_gens, burnin_is_exact=True
    )
    return fwdpy11.ModelParams(
        nregions=[], sregions=[],
        recregions=build_recregions(cfg),
        rates=(0.0, 0.0, None),
        gvalue=fwdpy11.Multiplicative(2.0),
        demography=demog,
        simlen=simlen,
    )


# --------------------------------------------------------------------------
# staged driver
# --------------------------------------------------------------------------

def run_forward(cfg: GenomeConfig, n_cur: int, total_gens: int,
                plans: Sequence[SweepPlan], seed: int,
                simplification_interval: int = 100,
                max_attempts: int = 4000,
                chunk: int = 25,
                min_soft_origins: int = 2,
                traj_thin: int = 1,
                verbose: bool = True):
    """Run the forward phase in stages, conditioning each sweep on non-loss.

    Returns (tskit TreeSequence, TrajectoryRecorder, seconds).
    """
    two_n = 2 * n_cur
    pop = fwdpy11.DiploidPopulation(n_cur, float(cfg.total_len))
    rng = fwdpy11.GSLrng(seed)
    recorder = TrajectoryRecorder(two_n, total_gens, traj_thin)

    by_gen: Dict[int, List[SweepPlan]] = {}
    for p in plans:
        g = total_gens - p.origin_gens_ago
        if not (0 < g < total_gens):
            raise ValueError(f"{p.sweep_id}: origin generation {g} outside (0,{total_gens})")
        by_gen.setdefault(g, []).append(p)
    origins = sorted(by_gen)

    t0 = time.time()
    first_call = True

    def evolve(duration: int, pop_, rec_):
        nonlocal first_call
        if duration <= 0:
            return
        params = make_params(cfg, n_cur, total_gens, duration)
        kw = dict(recorder=rec_, track_mutation_counts=True,
                  suppress_table_indexing=True)
        if first_call:
            kw["preserve_first_generation"] = True
        fwdpy11.evolvets(rng, pop_, params, simplification_interval, **kw)
        first_call = False

    for gi, g in enumerate(origins):
        evolve(g - pop.generation, pop, recorder)

        next_stop = origins[gi + 1] if gi + 1 < len(origins) else total_gens
        stage_plans = by_gen[g]
        s_min = min(p.s for p in stage_plans)
        window = min(est_window(s_min), next_stop - g)

        checkpoint = copy.deepcopy(pop)
        tracked_before = {k: list(v) for k, v in recorder.tracked.items()}
        real_before = copy.deepcopy(recorder.real)

        accepted = False
        attempts = 0

        while attempts < max_attempts and not accepted:
            attempts += 1
            trial = copy.deepcopy(checkpoint)
            recorder.tracked = {k: list(v) for k, v in tracked_before.items()}
            recorder.real = copy.deepcopy(real_before)

            new_ids = []
            for p in stage_plans:
                keys = []
                for _ in range(p.n_origins):
                    idx = trial.add_mutation(
                        rng,
                        window=(float(p.position), float(p.position) + 1.0),
                        ndescendants=1,
                        data=fwdpy11.NewMutationData(effect_size=p.s, dominance=p.h),
                    )
                    if idx is not None:
                        keys.append(trial.mutations[idx].key)
                if not keys:
                    continue
                r = SweepRealization(
                    sweep_id=p.sweep_id, position=p.position, s=p.s, h=p.h,
                    n_origins_introduced=len(keys),
                    intended_class=p.intended_class,
                    origin_gen_forward=g, origin_gens_ago=total_gens - g,
                )
                recorder.track(p.sweep_id, keys, r)
                new_ids.append(p.sweep_id)
            if len(new_ids) != len(stage_plans):
                continue

            # Evolve the establishment window in chunks, abandoning the attempt
            # as soon as a new sweep is gone.  Most failures die within the
            # first few 1/s generations, so paying for the whole window on
            # every failure wastes almost all of the compute.
            remaining = window
            alive = True
            while remaining > 0:
                step = min(chunk, remaining)
                evolve(step, trial, recorder)
                remaining -= step
                snap = recorder.snapshot(trial)
                if any(snap[sid][0] == 0 and not snap[sid][1] for sid in new_ids):
                    alive = False
                    break
                if all(snap[sid][1] or snap[sid][0] >= est_threshold(recorder.real[sid].s)
                       for sid in new_ids):
                    break
            if not alive:
                continue

            snap = recorder.snapshot(trial)
            accepted = True
            for sid in new_ids:
                total, fixed, surviving = snap[sid]
                plan = next(p for p in stage_plans if p.sweep_id == sid)
                if not fixed and total < est_threshold(plan.s):
                    accepted = False
                    break
                # A soft sweep that has already collapsed to one origin by the
                # end of the establishment window will never be soft, so retry
                # now rather than discovering it at the end of the replicate.
                if plan.n_origins > 1 and surviving < min_soft_origins:
                    accepted = False
                    break

            if accepted:
                for sid in new_ids:
                    recorder.real[sid].establishment_attempts = attempts
                    recorder.real[sid].established = True
                pop = trial

        if not accepted:
            raise RuntimeError(
                f"sweeps {[p.sweep_id for p in stage_plans]} at generation {g} "
                f"failed to establish in {max_attempts} attempts")
        if verbose:
            ids = ",".join(p.sweep_id for p in stage_plans)
            print(f"  [stage {gi+1}/{len(origins)}] gen {g} established {ids} "
                  f"after {attempts} attempt(s)  [{time.time()-t0:.0f}s]", flush=True)

    evolve(total_gens - pop.generation, pop, recorder)

    elapsed = time.time() - t0
    ts = pop.dump_tables_to_tskit()
    return ts, recorder, elapsed
