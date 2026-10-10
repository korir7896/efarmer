"""Grading logic for the confounded sweep scan.

Pure python, no network, no simulation at grade time: the sealed truth is a
static JSON file and this module compares the agent's submission against it.

Every threshold lives in GATES, so the whole specification of "correct" is
visible in one place.  The values are set from the measured spread of
competent methods (see authoring/evidence/), not from the reference
solution's own error.

Classes
-------
The four classes are the two independent binary properties of a sweep that a
genome scan can actually be asked to recover, crossed:

  * COMPLETE if the beneficial allele reached frequency >= 0.98 in the
    population, PARTIAL if it is still segregating below that;
  * SOFT if two or more independent origins of the beneficial allele (separate
    mutations at the same site, each on its own haplotype background) are
    still present today, HARD if exactly one is.

Nothing in the class labels encodes the sweep's age.  An earlier version split
the completed hard sweeps into "recent" and "old" at a fixed number of
generations, which made the age of a sweep partly a function of its class and
so let a class-conditional constant satisfy the timing gate without dating
anything.  Age is now graded only as a continuous quantity, over sweeps whose
true class is hard_complete, whose true completion times in the shipped
replicate are 116, 323, 789, 1724 and 2529 generations ago, a factor of 21.8.

The shipped zoo is fifteen sweeps: five hard_complete, three soft_complete,
four hard_partial and three soft_partial.  Their selection coefficients span
a factor of 12.0, from 0.0075 to 0.0900, and s is NON-MONOTONE in age -- by
ascending completion age the coefficients run 0.090, 0.028, 0.040, 0.013,
0.055 -- so neither graded quantity can be read off the other.  Three sweeps
sit inside wide recombination cold spots and two pairs sit 725 kb and 753 kb
apart, both above the dedupe radius so each member has to be called
separately.  Every threshold below was measured against the composition of
THAT zoo rather than against a calibration average; the tables in the
comments are those measurements, and authoring/evidence/s_gate_design.py and
fit_classifier.py --shipped-composition reproduce them.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

CLASSES = ("hard_complete", "hard_partial", "soft_complete", "soft_partial")
COMPLETE_CLASSES = ("hard_complete", "soft_complete")
# Completion time is graded only for single-origin completed sweeps.
# Diversity recovery dates a sweep by how far below background its centre
# still sits, which presumes the sweep drove diversity to zero. A soft sweep
# never did, so the recovery clock has no zero to start from and reads a soft
# sweep as an order of magnitude older than it is. Grading a quantity that no
# known estimator recovers would be grading noise.
TIMED_CLASSES = ("hard_complete",)

GATES: Dict[str, Any] = {
    # --- matching ---
    # A call matches a true sweep if it is on the same chromosome within this
    # many bp.  Set from the measured localisation spread of the best
    # statistic for each class, not from how precise the reference happens to
    # be: diversity puts completed hard sweeps within 19-31 kb (worst 121 kb)
    # and H12 puts partial sweeps within 25 kb (worst 91 kb), but soft sweeps
    # are the loose ones at 31 kb median and 211 kb worst, because their
    # diversity dip is shallow and only H12 locates them at all.
    #
    # The window cannot be ambiguous even though two sweep PAIRS now sit 725
    # and 753 kb apart: half of the smaller separation is 362 kb, so a call
    # within 250 kb of one of them is outside the window of the other, and
    # match() assigns greedily by ascending distance with each true sweep
    # taken at most once. Widening it cannot let a sloppy answer through
    # either -- spraying calls is caught by the dedupe radius and the
    # false-positive cap, not by the window.
    "match_window_bp": 250_000,
    # Two calls closer together than this are collapsed before grading, so an
    # agent cannot manufacture recall by carpeting a region with calls. It is
    # deliberately equal to the match window and well under the 725 kb
    # minimum separation of the close pairs, so resolving a pair is possible
    # but requires actually resolving it.
    "dedupe_bp": 250_000,

    # --- demography ---
    # Sizes are graded multiplicatively, and all three numbers come from the
    # measured spread of the reference's own fit rather than from how tight
    # they sound. The fit is a stochastic search against a simulated
    # spectrum, so the spread that matters is across search seeds; run at
    # five on the shipped replicate it gives:
    #
    #     seed       N_ancestral   N_current   bottleneck strength
    #     12345          1.009       1.107            2.946
    #     20240101       1.086       1.303            1.927
    #     555            1.166       1.275            1.211
    #     7              1.184       1.322            1.223
    #     99             1.006       1.144            1.327
    #
    # I had N_ancestral and N_current at 1.50 and the strength at 2.20, from
    # two agent transcripts on the PREVIOUS instance that recovered both
    # sizes to within 8% and 24%. That instance had 50 diploids and 48 Mb;
    # this one has 40 diploids, 70 Mb and fifteen sweeps rather than ten, and
    # those numbers do not survive the change. N_ancestral at 1.50 does --
    # the worst seed is 1.184 -- but N_current goes to 1.322 and the strength
    # to 2.946, so both of those are loosened to sit clear of the measured
    # worst case rather than just inside it.
    #
    # Why the fit cannot do better is worth stating, because it is the task's
    # own premise rather than a weakness in the reference. The TRUE history
    # is not the optimum of this objective: scored against the observed
    # spectrum it reaches 0.10 to 0.16 where the fitted history reaches
    # 0.009, because fifteen sweeps have skewed the observed spectrum towards
    # rare alleles and the best-fitting NEUTRAL history is the one that
    # mimics that skew. No amount of search fixes that, and it is exactly
    # what the task asks an agent to work around.
    "N_ancestral_factor": 1.50,
    "N_current_factor": 1.60,
    # The DEPTH and DURATION of a bottleneck are not separately identifiable
    # from the spectrum: a short severe contraction and a longer milder one
    # leave almost the same coalescent signature, because what the genealogy
    # integrates is duration/N_bottleneck.  Grading N_bottleneck on its own
    # would partly be grading which point of that ridge a fit happened to
    # land on.  So the gate is on the identifiable combination, and the
    # reported history must still be a real contraction followed by a
    # recovery (test_bottleneck_is_a_bottleneck). Even the combination is
    # only recovered to within a factor of three here, which is why the gate
    # is 3.50 and not something that sounds stricter: a constant-size answer
    # reports a strength of zero and fails it outright, which is the thing
    # it is there to catch.
    "bottleneck_strength_factor": 3.50,
    # The bottleneck's TIMING is not graded at all.  Measured on four
    # independent neutral replicates with known truth, a competent
    # spectrum-based fit recovered the expansion time at 0.64, 0.68, 1.31 and
    # 2.63 times its true value: one in four would fail even a 40% tolerance,
    # and a tolerance loose enough to admit 2.63 is wide enough to be
    # meaningless. The objective has a broad flat minimum spanning roughly
    # 1500 to 4000 generations for a truth of 3000, so this is a property of
    # the information in a site frequency spectrum from a sample this size,
    # not of the search. Methods that use linkage or sequentially-Markov
    # inference date bottlenecks far better, but a gate cannot demand a
    # capability the reference lacks and still be guaranteed passable.

    # --- detection ---
    # Per-class minimum recall, as a count over the true sweeps of that
    # class. The zoo carries 5 hard_complete, 3 soft_complete, 4 hard_partial
    # and 3 soft_partial, so this floor is ten matched calls out of fifteen
    # true sweeps -- against six out of ten in the version two agent
    # transcripts cleared with slack to spare.
    #
    # hard_complete is set to 4 of 5 because the timing gate is graded over
    # the matched hard_complete sweeps and needs four of them: recall has to
    # supply what timing grades.
    "min_recall_per_class": {
        "hard_complete": 4,
        "soft_complete": 2,
        "hard_partial": 2,
        "soft_partial": 2,
    },
    # Unmatched calls allowed genome-wide. Cold-spot calls count here, and
    # there are now three deep narrow cold spots carrying no sweep at all,
    # over 70 Mb rather than 48, so the budget is spent faster.
    "max_false_positives": 1,

    # --- characterisation, over matched calls only ---
    # 0.65, measured rather than chosen, and measured twice because the
    # first measurement was against the wrong population.
    #
    # The gate is a fraction of the MATCHED calls on one replicate whose
    # class mix is five hard_complete, three soft_complete, four
    # hard_partial and three soft_partial. The calibration set is
    # 42/22/31/25, which over-weights hard_complete -- the easiest class by
    # a wide margin, 41 of 42 correct. So four-way accuracy over the
    # calibration set reads 0.758 where the shipped mix reads 0.733, and the
    # classifier that wins on the calibration average reads 0.667 on the
    # shipped mix. Fraction of drawn submissions reaching each threshold,
    # over the shipped mix, with the classifier chosen on that objective:
    #
    #     threshold   all 15 matched   12 matched   10 matched
    #       0.60           0.995         0.909        0.972
    #       0.65           0.946         0.909        0.835
    #       0.70           0.756         0.695        0.835
    #       0.75           0.439         0.695        0.535
    #
    # I set this to 0.75 first on the reasoning that one agent transcript
    # scored 9 of 9 on the previous zoo, then measured 0.70 and set that,
    # then measured the shipped mix and set 0.65: at 0.70 the best measured
    # classifier fails on one draw in four. 0.65 is still a demanding gate
    # against this zoo -- the reference's own median is 0.733, so the gate
    # sits two calls out of fifteen below the measured ceiling, and a
    # submission that gets completeness right and guesses softness scores
    # about 0.50.
    #
    # The zoo is what got harder, not the threshold. Four-way accuracy fell
    # from 0.802 on the previous instance because the sample is 40 diploids
    # rather than 50 and the haplotype statistics that separate soft from
    # hard lose the most from that, because s now spans 12x so the features
    # that classify a sweep vary more across the zoo, and because the
    # realized zoo puts a hard_partial sweep at frequency 0.978 -- two
    # points under the completion threshold, with very nearly a completed
    # sweep's footprint. That one is misclassified by every calibration
    # replicate of its configuration, and it is there on purpose.
    "min_class_accuracy": 0.65,
    # Selection coefficients are graded three ways, and all three must hold.
    #
    # MAGNITUDE: at least this fraction within a multiplicative factor.
    # SPREAD: the reported coefficients must themselves span this factor
    #   across the matched calls. A single constant spans 1.0 and fails with
    #   certainty, whatever its value and whatever the match set.
    # ORDER: Spearman rank correlation against the truth. A constant is all
    #   ties, which scores zero by the convention in grading.spearman().
    #
    # All four numbers come from authoring/evidence/s_gate_design.py, which
    # scores drawn submissions over the composition of the SHIPPED zoo --
    # fifteen sweeps, three of them inside a cold spot -- rather than over
    # the calibration set as a whole, because three of fifteen is what the
    # gate will see. Fraction of draws passing all three gates at once:
    #
    #     factor  min acc  min rho    reference   base-rate-calibrated
    #       2.20     0.70     0.50        0.880                  0.781
    #       2.20     0.75     0.50        0.797                  0.740
    #       2.50     0.75     0.50        0.891                  0.781
    #       2.50     0.80     0.50        0.891                  0.781
    #       2.50     0.80     0.40        0.986                  0.892
    #       2.50     0.85     0.40        0.912                  0.887
    #       2.80     0.90     0.40        0.581                  0.596
    #
    # I had these at 2.20 / 0.75 / 0.50 on the reasoning that a tighter
    # tolerance makes a harder task. Measured, that setting fails the best
    # estimator available on one draw in five, which is not a hard gate but a
    # broken one, and the binding constraint turned out to be the rank gate
    # rather than the tolerance: pooled rank correlation has a median of
    # +0.661 over draws and a 1st percentile of +0.386, so 0.50 alone costs
    # about eleven percent of draws. 2.50 / 0.80 / 0.40 is the one setting the
    # reference clears essentially always, and it is a real tightening of the
    # fraction required (0.75 -> 0.80) bought by loosening the tolerance.
    #
    # What these gates do NOT do is worth stating, because the obvious story
    # is wrong. The cold spots stretch a footprint in base pairs, so reading
    # strength off physical width overestimates it there, and the same
    # measurement shows what that is worth: an estimator calibrated only on
    # base-rate simulations is biased 2.65x on the soft completed sweep
    # inside a cold spot and misses it by up to 5.90x -- but it is one sweep
    # in fifteen, and such a submission still clears all three gates on 89%
    # of draws against the reference's 99%. The cold spots earn their place
    # as false-positive traps for DETECTION, where three of the six carry no
    # sweep at all, not as a strength-estimation trap. Claiming otherwise
    # would be asserting a difficulty the evidence does not support.
    #
    # The true coefficients span a factor of 12.0 across the zoo and s is
    # non-monotone in sweep age, so the rank gate cannot be earned by dating
    # sweeps instead of measuring their strength.
    "s_within_factor": 2.50,
    "min_s_accuracy": 0.80,
    "min_s_spread": 3.00,
    "min_s_rank_corr": 0.40,
    "min_matched_for_rank": 8,
    # Timing is graded only over sweeps whose TRUE class is hard_complete --
    # grading against the true class, not the reported one, so a
    # misclassification is not punished twice. The five completion ages are
    # spaced so that no three tolerance windows have a point in common, which
    # gate_feasibility.design_guarantee() checks over the generator's
    # ACCEPTANCE WINDOWS rather than over the one realized replicate: the
    # best a single constant can do is two of the five, so requiring four
    # cannot be met by one, and since class no longer carries age, a
    # class-conditional constant is just a constant here.
    #
    # Tolerance and count both come from measurement over the composition of
    # the shipped zoo -- its five specific hard_complete configurations, one
    # of them inside a cold spot -- rather than from the calibration average,
    # because five sweeps is what the gate counts over. Fraction of drawn
    # submissions reaching each count, under the shipped clock:
    #
    #     tolerance     calibration    >=3 of 5   >=4 of 5   >=5 of 5
    #     50% / 150         0.88         1.000      1.000      0.671
    #     45% / 140         0.86         1.000      1.000      0.668
    #     40% / 120         0.79         1.000      0.892      0.441
    #     35% / 110         0.62         0.812      0.441      0.098
    #
    # 45% / 140 is therefore free: it is strictly tighter than 50% / 150 and
    # costs nothing at the count the gate asks for. 40% does cost, and five
    # of five is not reachable at any tolerance -- the youngest sweep
    # completed about 130 generations ago, where the absolute floor of the
    # tolerance is the whole of its age and the clock has almost no dynamic
    # range to work with.
    #
    # The previous version graded four dateable sweeps spanning a factor of
    # 15.8 and required three of them (75%); this grades five spanning a
    # factor of about 19 and requires four (80%) at a tighter tolerance, one
    # of which sits inside a recombination cold spot, and the recall gate has
    # to supply four hard_complete sweeps for four to be gradeable at all.
    "time_rel_tol": 0.45,
    "time_abs_tol_gens": 140,
    "min_time_correct": 4,
}


class SchemaError(ValueError):
    pass


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def parse_submission(obj: Any, n_chrom: int, chrom_len: int) -> Tuple[Dict, List[Dict]]:
    """Validate structure and types. Raises SchemaError with a reason."""
    if not isinstance(obj, dict):
        raise SchemaError("top level is not a JSON object")

    dem = obj.get("demography")
    if not isinstance(dem, dict):
        raise SchemaError("missing or non-object 'demography'")
    need = ("N_ancestral", "N_bottleneck", "N_current",
            "T_bottleneck_start_gens_ago", "T_bottleneck_end_gens_ago")
    for k in need:
        if k not in dem:
            raise SchemaError(f"demography missing '{k}'")
        if not _finite(dem[k]):
            raise SchemaError(f"demography['{k}'] is not a finite number: {dem[k]!r}")
        if float(dem[k]) <= 0:
            raise SchemaError(f"demography['{k}'] must be positive")

    raw = obj.get("sweeps")
    if not isinstance(raw, list):
        raise SchemaError("missing or non-list 'sweeps'")
    if len(raw) > 200:
        raise SchemaError(f"{len(raw)} calls submitted; limit is 200")

    calls = []
    for i, c in enumerate(raw):
        if not isinstance(c, dict):
            raise SchemaError(f"sweeps[{i}] is not an object")
        for k in ("chrom", "position", "class", "selection_coefficient"):
            if k not in c:
                raise SchemaError(f"sweeps[{i}] missing '{k}'")
        ch = c["chrom"]
        if isinstance(ch, str) and ch.lower().startswith("chr"):
            ch = ch[3:]
        try:
            ch = int(ch)
        except (TypeError, ValueError):
            raise SchemaError(f"sweeps[{i}]['chrom'] is not a chromosome: {c['chrom']!r}")
        if not (1 <= ch <= n_chrom):
            raise SchemaError(f"sweeps[{i}]['chrom']={ch} outside 1..{n_chrom}")
        if not _finite(c["position"]):
            raise SchemaError(f"sweeps[{i}]['position'] is not a finite number")
        p = int(round(float(c["position"])))
        if not (1 <= p <= chrom_len):
            raise SchemaError(f"sweeps[{i}]['position']={p} outside 1..{chrom_len}")
        kl = c["class"]
        if kl not in CLASSES:
            raise SchemaError(f"sweeps[{i}]['class']={kl!r} not one of {CLASSES}")
        if not _finite(c["selection_coefficient"]):
            raise SchemaError(f"sweeps[{i}]['selection_coefficient'] is not finite")
        s = float(c["selection_coefficient"])
        if not (0.0 < s < 1.0):
            raise SchemaError(f"sweeps[{i}]['selection_coefficient']={s} outside (0,1)")
        t = c.get("time_gens_ago", None)
        if kl in COMPLETE_CLASSES and t is None:
            # Required, not optional: a sweep the submission itself calls
            # complete has a completion time, and that time is graded.
            raise SchemaError(
                f"sweeps[{i}] has class {kl!r} and so must carry a numeric "
                f"'time_gens_ago'")
        if t is not None:
            if not _finite(t):
                raise SchemaError(f"sweeps[{i}]['time_gens_ago'] is neither null nor finite")
            t = float(t)
            if t <= 0:
                raise SchemaError(f"sweeps[{i}]['time_gens_ago'] must be positive")
        calls.append({"i": i, "chrom": ch, "position": p, "class": kl,
                      "s": s, "time": t})
    return dem, calls


def dedupe(calls: List[Dict], window: int) -> List[Dict]:
    """Collapse near-duplicate calls, keeping the first on each chromosome."""
    kept: List[Dict] = []
    for c in sorted(calls, key=lambda x: (x["chrom"], x["position"], x["i"])):
        if any(k["chrom"] == c["chrom"] and abs(k["position"] - c["position"]) < window
               for k in kept):
            continue
        kept.append(c)
    return kept


def match(truth_sweeps: List[Dict], calls: List[Dict], window: int
          ) -> Tuple[List[Tuple[Dict, Dict, int]], List[Dict], List[Dict]]:
    """Greedy global matching by ascending distance.

    Returns (matched pairs with distance, unmatched truths, unmatched calls).
    """
    pairs = []
    for t in truth_sweeps:
        for c in calls:
            if c["chrom"] != t["chrom"]:
                continue
            d = abs(c["position"] - t["position_local"])
            if d <= window:
                pairs.append((d, t["sweep_id"], c["i"]))
    pairs.sort(key=lambda x: (x[0], x[1], x[2]))

    tby = {t["sweep_id"]: t for t in truth_sweeps}
    cby = {c["i"]: c for c in calls}
    used_t, used_c = set(), set()
    matched = []
    for d, tid, ci in pairs:
        if tid in used_t or ci in used_c:
            continue
        used_t.add(tid); used_c.add(ci)
        matched.append((tby[tid], cby[ci], d))
    un_t = [t for t in truth_sweeps if t["sweep_id"] not in used_t]
    un_c = [c for c in calls if c["i"] not in used_c]
    return matched, un_t, un_c


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation, ties averaged. Returns 0.0 when either side is constant.

    Pure python on purpose: the verifier image carries pytest and nothing
    else, so grading must not import numpy or scipy.
    """
    n = len(x)
    if n != len(y) or n < 3:
        return 0.0

    def ranks(v: Sequence[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0 or syy <= 0:
        # One side has no variation at all -- a constant answer. There is no
        # ordering to score, so it scores nothing.
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def time_tolerance(t_true: float) -> float:
    """Half-width of the accepted window around a true completion time."""
    return max(GATES["time_rel_tol"] * float(t_true),
               float(GATES["time_abs_tol_gens"]))
