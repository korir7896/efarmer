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
    completed: bool = False
    completion_gen_forward: int = -1
    completion_gens_ago: int = -1
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

# A sweep counts as COMPLETE at this frequency rather than at exact fixation.
# With many origins, exact fixation requires every ancestral haplotype to be
# lost, which can lag the sweep by hundreds of generations: a measured
# 200-origin sweep sat at 0.9999 and so was still formally segregating. No
# statistic distinguishes 0.9999 from 1.0, and what diversity recovery
# actually dates is when the sweep finished, so completion is both the
# robust definition and the estimable one.
COMPLETION_FREQ = 0.98


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
    def _key_index(pop) -> Dict[tuple, int]:
        """One key-to-row map per generation.

        Scanning the mutation table once per tracked key is O(keys x table).
        With three soft sweeps of a hundred origins each that is around
        ninety thousand comparisons every generation, which dominated the
        whole simulation. Building the map once makes each lookup O(1).
        """
        return {m.key: i for i, m in enumerate(pop.mutations)}

    def _count_for_key(self, pop, key, lookup=None) -> Tuple[int, bool]:
        lookup = self._key_index(pop) if lookup is None else lookup
        i = lookup.get(key)
        if i is not None:
            c = int(pop.mcounts[i]) if i < len(pop.mcounts) else 0
            if c > 0:
                return c, False
        # NOTE: find_fixation_by_key returns -1 (not None) when absent.
        if pop.find_fixation_by_key(key) >= 0:
            return -1, True
        return 0, False

    def _sweep_state(self, pop, keys, lookup=None) -> Tuple[int, bool, int]:
        """(total allele count, whether fixed, number of surviving origins)."""
        lookup = self._key_index(pop) if lookup is None else lookup
        total = 0
        surviving = 0
        fixed = False
        for k in keys:
            c, is_fixed = self._count_for_key(pop, k, lookup)
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
        lookup = self._key_index(pop)
        return {sid: self._sweep_state(pop, ks, lookup)
                for sid, ks in self.tracked.items()}

    def __call__(self, pop, sampler) -> None:
        gen = pop.generation
        lookup = self._key_index(pop)
        for sid, keys in self.tracked.items():
            r = self.real[sid]
            total, fixed, surviving = self._sweep_state(pop, keys, lookup)
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
            if r.final_freq >= COMPLETION_FREQ and not r.completed:
                r.completed = True
                r.completion_gen_forward = gen
                r.completion_gens_ago = self.total_gens - gen
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
# driver: checkpoints, deadlines and backtracking
# --------------------------------------------------------------------------
#
# Earlier versions tested each sweep's establishment inside the gap before the
# next sweep's origin.  That is wrong whenever sweeps are closely spaced, and
# in this zoo they are: six of the seven single-origin sweeps sit 10 to 170
# generations from their neighbour while needing 196 to 525 generations to
# reach an escape-from-drift threshold.  Clipping the window to the gap made
# the threshold unreachable for one sweep (4000 futile attempts) and, worse,
# made the others succeed only on atypically fast early trajectories -- 74 and
# 93 attempts to find a singleton that reached 50 copies in 40 generations.
# That silently biased the sweeps towards unrepresentative growth.
#
# Instead each sweep gets a DEADLINE, origin + est_window(s), independent of
# where other sweeps sit.  The simulation runs through the deadlines, and a
# sweep that has not escaped drift by its own deadline sends the run back to
# that sweep's checkpoint for a fresh draw.  Later sweeps are re-added on the
# next pass.  Failure is detected as early as possible -- a lost variant
# within one chunk -- so a retry costs about one establishment window rather
# than the whole remaining run.


def summarize_stage_failures(tally: Dict[str, List[dict]]) -> str:
    """Why a sweep could not be established, counted by cause.

    "failed to establish" is three different diagnoses wearing one name, and
    they call for opposite fixes. If the allele is being lost, the sweep is
    simply improbable and wants more origins or a stronger s. If it survives
    but is short of its count at the deadline, the deadline is too early for
    the s it was given. If it survives with enough copies but too few
    distinct origins, the softness requirement is what is unreachable, and
    the fix is more origins, not a longer window. If the mutation cannot be
    inserted at all, nothing about the plan is wrong and no number of retries
    will help -- that is a simulator-level refusal at that position.
    """
    if not tally:
        return "    (no stage-level failures recorded)"
    out = []
    for sid in sorted(tally):
        rows = tally[sid]
        causes: Dict[str, int] = {}
        for r in rows:
            causes[r["why"]] = causes.get(r["why"], 0) + 1
        out.append(f"    {sid}: {len(rows)} failure(s); "
                   + ", ".join(f"{k} x{v}" for k, v in sorted(causes.items())))
        short = [r for r in rows if r["why"] == "short at deadline"]
        if short:
            cnt = sorted(r["count"] for r in short)
            org = sorted(r["surviving_origins"] for r in short)
            blocked: Dict[str, int] = {}
            for r in short:
                blocked[r["blocked_by"]] = blocked.get(r["blocked_by"], 0) + 1
            out.append(
                f"      at deadline: count {cnt[0]}/{cnt[len(cnt)//2]}/{cnt[-1]} "
                f"(need {short[0]['need_count']}), surviving origins "
                f"{org[0]}/{org[len(org)//2]}/{org[-1]} "
                f"(need {short[0]['need_origins']}); blocked by "
                + ", ".join(f"{k} x{v}" for k, v in sorted(blocked.items())))
        lost = [r for r in rows if r["why"] == "allele lost"]
        if lost:
            g = sorted(r["gens_since_origin"] for r in lost)
            out.append(f"      lost after {g[0]}/{g[len(g)//2]}/{g[-1]} "
                       f"generations (min/median/max)")
        spans = [r for r in rows if r.get("tree_span_refusals")]
        if spans:
            tot = sum(r["tree_span_refusals"] for r in spans)
            out.append(f"      simulator refused {tot} insertion(s) at this "
                       f"position across {len(spans)} attempt(s)")
    return "\n".join(out)


def summarize_rejections(tally: Dict[str, List[dict]]) -> str:
    """What each sweep actually realized across every rejected attempt.

    When a replicate cannot be accepted, the question is always which windows
    are wrong and by how much. One sampled log line cannot answer that; the
    distribution over hundreds of rejections can, and it is the input to
    retuning origins in context.
    """
    if not tally:
        return "    (no acceptance failures recorded)"
    import statistics as _st
    lines = ["    realized values over rejected attempts:"]
    for sid in sorted(tally, key=lambda x: (len(x), x)):
        obs = tally[sid]
        done = [o["done"] for o in obs if o["done"] is not None]
        freq = [o["freq"] for o in obs]
        orig = [o["origins"] for o in obs]
        bits = [f"n={len(obs)}"]
        if done:
            bits.append(f"completed {min(done)}/{int(_st.median(done))}/"
                        f"{max(done)}")
        if freq:
            bits.append(f"freq {min(freq):.3f}/{_st.median(freq):.3f}/"
                        f"{max(freq):.3f}")
        if orig:
            bits.append(f"origins {min(orig)}/{int(_st.median(orig))}/"
                        f"{max(orig)}")
        lines.append(f"      {sid:<6} " + "  ".join(bits))
    return "\n".join(lines)


@dataclass
class _Stage:
    gen: int
    plan: SweepPlan
    deadline: int


def run_forward(cfg: GenomeConfig, n_cur: int, total_gens: int,
                plans: Sequence[SweepPlan], seed: int,
                simplification_interval: int = 100,
                max_attempts_per_stage: int = 800,
                stage_report_every: int = 50,
                max_total_restarts: int = 40_000,
                chunk: int = 25,
                min_soft_origins: int = 5,
                accept_fn=None,
                _dbg: bool = False,
                traj_thin: int = 1,
                verbose: bool = True):
    """Run the forward phase, conditioning every sweep on escaping drift.

    `accept_fn`, if given, is called on the realized sweeps once the run
    reaches the present.  It must return (ok, [failing sweep_id, ...]); the
    run then backtracks to the earliest failing sweep instead of discarding
    the whole replicate.

    Returns (tskit TreeSequence, TrajectoryRecorder, seconds).
    """
    two_n = 2 * n_cur
    rng = fwdpy11.GSLrng(seed)
    recorder = TrajectoryRecorder(two_n, total_gens, traj_thin)

    # one sweep per stage; duplicate origins nudged apart
    used = set()
    stages: List[_Stage] = []
    for p in sorted(plans, key=lambda x: (-x.origin_gens_ago, x.sweep_id)):
        g = total_gens - p.origin_gens_ago
        while g in used:
            g += 1
        if not (0 < g < total_gens):
            raise ValueError(f"{p.sweep_id}: origin generation {g} outside (0,{total_gens})")
        used.add(g)
        stages.append(_Stage(gen=g, plan=p,
                             deadline=min(total_gens, g + est_window(p.s))))
    stages.sort(key=lambda st: st.gen)
    n_stages = len(stages)
    stage_of = {st.plan.sweep_id: k for k, st in enumerate(stages)}

    t0 = time.time()
    first_call = True

    def evolve(pop_, duration: int):
        nonlocal first_call
        if duration <= 0:
            return
        params = make_params(cfg, n_cur, total_gens, duration)
        kw = dict(recorder=recorder, track_mutation_counts=True,
                  suppress_table_indexing=True)
        if first_call:
            kw["preserve_first_generation"] = True
        fwdpy11.evolvets(rng, pop_, params, simplification_interval, **kw)
        first_call = False

    def snapshot(pop_):
        return (copy.deepcopy(pop_),
                {k: list(v) for k, v in recorder.tracked.items()},
                copy.deepcopy(recorder.real))

    def restore(snap):
        pop_, tracked, real = snap
        recorder.tracked = {k: list(v) for k, v in tracked.items()}
        recorder.real = copy.deepcopy(real)
        return copy.deepcopy(pop_)

    # Half-widths tried, in order, when the simulator refuses to insert at
    # the planned position. See _resolve_position.
    SEARCH_HALF_WIDTHS = (0.5, 8.0, 64.0, 512.0, 4096.0)

    def _insert(pop_, st: _Stage, lo: float, hi: float):
        """One origin somewhere in [lo, hi), or None if the simulator refuses.

        Re-raises anything that is not the tree-span refusal: a different
        RuntimeError means something is actually wrong and must not be
        silently retried at another position.
        """
        try:
            return pop_.add_mutation(
                rng, window=(lo, hi), ndescendants=1,
                data=fwdpy11.NewMutationData(effect_size=st.plan.s,
                                             dominance=st.plan.h))
        except RuntimeError as e:
            if "tree span" not in str(e):
                raise
            return None

    def _resolve_position(pop_, st: _Stage):
        """A position at this locus that the simulator will actually accept.

        fwdpy11 raises "invalid tree span of <= 0.0" when the marginal tree
        covering the requested window gives it nothing to place a mutation
        on. Whether that happens is a property of the tree sequence at that
        instant, so it is deterministic within a restart: backtracking
        restores the same population and re-requests the same single base
        pair, which is refused again. Measured on this zoo, sw14 was refused
        all fourteen of its origins on every one of 800 consecutive attempts
        and killed a 40-minute run -- not because the sweep was improbable,
        but because that one base pair was unusable.

        Treating the refusal as a lost origin, which is what this did
        before, cannot recover from that: retrying the same base pair is
        guaranteed to fail the same way. So widen the window instead. A
        sweep's locus is an arbitrary point in a neutral genome and the
        acceptance windows are in generations and frequencies, not base
        pairs, so moving the site a few kilobases changes nothing that is
        graded: 4 kb at the base rate is 4e-05 Morgans, against the 250 kb
        the verifier allows a call to be out by.

        The position that is actually used is returned and recorded as the
        sweep's position, so the sealed truth says where the mutation IS
        rather than where it was asked for. In the normal case the planned
        base pair is accepted and the two are identical; this only ever
        moves a sweep that could not otherwise exist.
        """
        planned = float(st.plan.position)
        for half in SEARCH_HALF_WIDTHS:
            idx = _insert(pop_, st, planned - half + 0.5, planned + half + 0.5)
            if idx is not None:
                return idx, float(pop_.mutations[idx].pos), half
        return None, planned, None

    def add_stage(pop_, st: _Stage):
        idx, pos, half = _resolve_position(pop_, st)
        if idx is None:
            stage_tally.setdefault(st.plan.sweep_id, []).append(
                {"why": "no position at this locus was accepted",
                 "tree_span_refusals": len(SEARCH_HALF_WIDTHS),
                 "n_origins": st.plan.n_origins})
            return False
        keys = [pop_.mutations[idx].key]
        # Report only a real widening. fwdpy11 chooses the base pair itself
        # inside whatever window it is given, so even the narrowest window
        # can come back one base pair from the planned one; that is not a
        # refusal and saying so would bury the message that matters. It also
        # means the planned position and the mutation's position were never
        # quite the same thing, which is why the realized one is what gets
        # recorded.
        if half > SEARCH_HALF_WIDTHS[0] and verbose:
            print(f"  {st.plan.sweep_id}: the simulator refused every window "
                  f"up to {SEARCH_HALF_WIDTHS[SEARCH_HALF_WIDTHS.index(half) - 1]:.0f} bp "
                  f"around the planned position; using {pos:,.0f}, "
                  f"{abs(pos - float(st.plan.position)):,.0f} bp away",
                  flush=True)
        # Every remaining origin goes at the position the first one actually
        # took, not at the planned one. They have to share a single site to
        # be independent origins of the SAME allele, which is what makes the
        # sweep soft; letting each pick its own position inside a widened
        # window would scatter them across a few kilobases and the sweep
        # would be fourteen separate sweeps.
        span_fail = 0
        for _ in range(st.plan.n_origins - 1):
            k = _insert(pop_, st, pos, pos + 1.0)
            if k is None:
                span_fail += 1
                continue
            keys.append(pop_.mutations[k].key)
        if span_fail:
            stage_tally.setdefault(st.plan.sweep_id, []).append(
                {"why": "partial insertion", "tree_span_refusals": span_fail,
                 "inserted": len(keys), "n_origins": st.plan.n_origins})
        recorder.track(st.plan.sweep_id, keys, SweepRealization(
            sweep_id=st.plan.sweep_id, position=pos,
            s=st.plan.s, h=st.plan.h, n_origins_introduced=len(keys),
            intended_class=st.plan.intended_class,
            origin_gen_forward=st.gen, origin_gens_ago=total_gens - st.gen))
        return True

    # ---- initial state, evolved to the first origin ----
    pop = fwdpy11.DiploidPopulation(n_cur, float(cfg.total_len))
    evolve(pop, stages[0].gen)
    snaps: Dict[int, tuple] = {0: snapshot(pop)}

    attempts = [0] * n_stages
    start_stage = 0
    total_restarts = 0
    reject_tally: Dict[str, List[dict]] = {}
    # Why each stage-level (establishment) failure happened, as opposed to
    # reject_tally, which records why a whole replicate was not accepted.
    # Without this a stage that fails every attempt reports only its name,
    # and "failed to establish" covers three different causes: the mutation
    # could not be inserted at all, the allele was lost, or it survived but
    # missed the count or the surviving-origin requirement at its deadline.
    stage_tally: Dict[str, List[dict]] = {}

    def backtrack_to(k: int) -> None:
        """Rewind to stage k and forget the failures of everything after it.

        Resetting the downstream counters is the whole point. A stage's
        attempt count is evidence that THAT stage cannot be realized from the
        state it starts in; once an earlier stage is re-run, every later
        stage starts from a different population and its old failures say
        nothing. Without the reset the counters accumulate across restarts
        and the first merely-improbable sweep downstream of a fussy one
        exhausts its budget: measured on this zoo, 29 acceptance restarts at
        sw09 ran sw07's establishment counter to 800 and killed the run,
        although sw07 establishes in about one attempt in thirty and had
        succeeded on every one of those 29 passes.
        """
        nonlocal start_stage
        for j in range(k + 1, n_stages):
            attempts[j] = 0
        start_stage = k

    while True:
        pop = restore(snaps[start_stage])
        cur = start_stage
        confirmed: Dict[str, bool] = {
            st.plan.sweep_id: True for st in stages[:start_stage]}
        failed_at: Optional[int] = None

        while True:
            # 1. introduce every sweep scheduled at the current generation.
            #    This has to come BEFORE choosing the next target: the first
            #    stage's origin IS the generation the population starts at,
            #    so treating origins only as future events evolved straight
            #    past it and never added the mutation.
            while cur < n_stages and stages[cur].gen == pop.generation:
                snaps[cur] = snapshot(pop)
                if not add_stage(pop, stages[cur]):
                    failed_at = cur
                    break
                confirmed[stages[cur].plan.sweep_id] = False
                cur += 1
            if failed_at is not None:
                break

            # 2. judge the sweeps: lost now, or short at their own deadline.
            #
            # The LOSS check runs on every tracked sweep, confirmed or not.
            # Confirmation means a sweep cleared its establishment threshold,
            # not that it can no longer be lost, and on a backtrack every
            # stage before the restart point is marked confirmed and replayed.
            # Skipping those meant a sweep whose allele died during the replay
            # went unnoticed until replicate acceptance, hundreds of
            # generations later: that is how sw09, a weak sweep sitting just
            # before two sweeps that need many establishment attempts, kept
            # arriving at the present with a frequency of zero after having
            # been established in an earlier pass.
            snap_counts = recorder.snapshot(pop)
            for sid in list(recorder.tracked):
                st = stages[stage_of[sid]]
                total, done, surviving = snap_counts[sid]
                if total == 0 and not done:
                    stage_tally.setdefault(sid, []).append(
                        {"why": "allele lost",
                         "gens_since_origin": pop.generation - st.gen,
                         "deadline_in": st.deadline - pop.generation})
                    failed_at = stage_of[sid]
                    break
                if confirmed.get(sid):
                    continue
                if pop.generation >= st.deadline:
                    strong = done or total >= est_threshold(st.plan.s)
                    # How many origins must still be present at the
                    # establishment deadline for the sweep to count as soft.
                    # A flat five is right for a hundred introductions and
                    # wrong for twelve: the longer the establishment window,
                    # the more the founding lineages drift against each
                    # other, and a weak sweep's window is long. Measured on
                    # twelve introductions, five survivors after the 351
                    # generations s=0.025 needs is rare enough that the
                    # driver spent eighteen minutes on one replicate without
                    # finishing, where s=0.060, whose window is 118
                    # generations, kept five or six in 8 of 8. Two surviving
                    # origins is what makes a sweep soft by definition, so
                    # the requirement scales and never drops below two, and
                    # is capped at three: this check only exists to abandon a
                    # hopeless run early, and replicate acceptance is what
                    # actually decides whether a sweep counts as soft.
                    soft_target = max(2, min(3, st.plan.n_origins // 8))
                    soft_ok = (st.plan.n_origins == 1
                               or surviving >= soft_target)
                    if strong and soft_ok:
                        confirmed[sid] = True
                        recorder.real[sid].established = True
                    else:
                        stage_tally.setdefault(sid, []).append(
                            {"why": "short at deadline",
                             "count": total, "need_count": est_threshold(st.plan.s),
                             "fixed": done,
                             "surviving_origins": surviving,
                             "need_origins": soft_target,
                             "blocked_by": ("count" if not strong else "origins")})
                        failed_at = stage_of[sid]
                        break
            if failed_at is not None:
                break

            # 3. finished?
            if pop.generation >= total_gens and cur >= n_stages:
                break

            # 4. advance to the next point of interest.
            targets = [total_gens]
            if cur < n_stages:
                targets.append(stages[cur].gen)
            pending = [stages[stage_of[sid]] for sid in recorder.tracked
                       if not confirmed.get(sid)]
            targets += [st.deadline for st in pending]
            if pending:
                targets.append(pop.generation + chunk)
            ahead = [t for t in targets if t > pop.generation]
            if not ahead:
                raise RuntimeError(
                    f"no event ahead of generation {pop.generation}: "
                    f"cur={cur}/{n_stages} targets={sorted(set(targets))}")
            evolve(pop, min(ahead) - pop.generation)
            if _dbg:
                print(f"    [dbg] gen={pop.generation} cur={cur} pending="
                      f"{[x for x in recorder.tracked if not confirmed.get(x)]}",
                      flush=True)

        total_restarts += 1
        if total_restarts > max_total_restarts:
            raise RuntimeError(
                f"gave up after {max_total_restarts} restarts; "
                f"attempts per stage: "
                f"{dict(zip((st.plan.sweep_id for st in stages), attempts))}\n"
                + summarize_rejections(reject_tally))

        if failed_at is not None:
            attempts[failed_at] += 1
            sid_f = stages[failed_at].plan.sweep_id
            if verbose and attempts[failed_at] % stage_report_every == 0:
                print(f"  {sid_f}: {attempts[failed_at]} consecutive "
                      f"establishment failures\n"
                      + summarize_stage_failures({sid_f: stage_tally.get(sid_f, [])}),
                      flush=True)
            if attempts[failed_at] > max_attempts_per_stage:
                raise RuntimeError(
                    f"{sid_f} failed to establish in "
                    f"{max_attempts_per_stage} consecutive attempts\n"
                    + summarize_stage_failures(stage_tally))
            backtrack_to(failed_at)
            continue

        # reached the present with every sweep established
        if accept_fn is not None:
            ok, failing, problems = accept_fn(recorder.real)
            if not ok:
                ks = [stage_of[sid] for sid in failing if sid in stage_of]
                if ks:
                    k = min(ks)
                    attempts[k] += 1
                    if attempts[k] > max_attempts_per_stage:
                        raise RuntimeError(
                            f"{stages[k].plan.sweep_id} never met its acceptance "
                            f"criteria in {max_attempts_per_stage} attempts")
                    # Tally every acceptance failure, not just the ones a
                    # sparse log line happens to catch. Tuning origins in
                    # context is the step that decides whether a replicate is
                    # reachable at all, and two sampled observations out of
                    # four hundred restarts is not enough to tune on.
                    for sid in failing:
                        r = recorder.real.get(sid)
                        if r is None:
                            continue
                        reject_tally.setdefault(sid, []).append(
                            dict(done=r.completion_gens_ago if r.completed
                                 else None,
                                 freq=float(r.final_freq),
                                 origins=int(r.n_origins_surviving)))
                    if verbose:
                        live = recorder.snapshot(pop)
                        detail = ", ".join(
                            f"{sid}: live total={live[sid][0]} "
                            f"fixed={live[sid][1]} origins={live[sid][2]} "
                            f"vs recorded count={recorder.real[sid].final_count} "
                            f"origins={recorder.real[sid].n_origins_surviving}"
                            for sid in failing if sid in live)
                        print(f"  backtracking to {stages[k].plan.sweep_id} "
                              f"(restart {total_restarts}, attempt "
                              f"{attempts[k]}, gen {pop.generation}): "
                              + "; ".join(problems) + "  [" + detail + "]",
                              flush=True)
                    backtrack_to(k)
                    continue
        break

    # restore() rolls recorder.real back to a pre-add snapshot, so attempt
    # counts are written once here from the driver's own tally.
    for k, st in enumerate(stages):
        if st.plan.sweep_id in recorder.real:
            recorder.real[st.plan.sweep_id].establishment_attempts = attempts[k] + 1

    if verbose:
        print("  established: " + ", ".join(
            f"{st.plan.sweep_id}({attempts[k] + 1} try)"
            for k, st in enumerate(stages)) + f"  [{time.time() - t0:.0f}s]",
            flush=True)

    elapsed = time.time() - t0
    ts = pop.dump_tables_to_tskit()
    return ts, recorder, elapsed
