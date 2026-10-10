"""Grading logic for the confounded sweep scan.

Pure python, no network, no simulation at grade time: the sealed truth is a
static JSON file and this module compares the agent's submission against it.

Every threshold lives in GATES, so the whole specification of "correct" is
visible in one place.  The values are set from the measured spread of
competent methods (see authoring/evidence/), not from the reference
solution's own error.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

CLASSES = ("hard_recent_complete", "hard_old_complete", "soft_standing", "partial")
COMPLETE_CLASSES = ("hard_recent_complete", "hard_old_complete", "soft_standing")
# Completion time is graded only for single-origin completed sweeps.
# Diversity recovery dates a sweep by how far below background its centre
# still sits, which presumes the sweep drove diversity to zero. A soft sweep
# never did: its measured diversity at the centre is 0.74 of background, so
# the recovery clock has no zero to start from and reads a soft sweep as
# roughly eighteen times older than it is. Grading a quantity that no
# estimator recovers would be grading noise.
TIMED_CLASSES = ("hard_recent_complete", "hard_old_complete")

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
    # Widening this cannot let a sloppy answer through: true sweeps are at
    # least 2 Mb apart, so there is no ambiguity about which one a call
    # belongs to, and spraying calls is caught by the false-positive cap
    # rather than by the window.
    "match_window_bp": 250_000,
    # Two calls closer together than this are collapsed before grading, so an
    # agent cannot manufacture recall by carpeting a region with calls.
    "dedupe_bp": 250_000,

    # --- demography ---
    # Sizes are graded multiplicatively. N_ancestral and N_current are
    # identifiable from the site-frequency spectrum given the mutation rate.
    "N_ancestral_factor": 1.60,
    "N_current_factor": 1.60,
    # The DEPTH and DURATION of a bottleneck are not separately identifiable
    # from the spectrum: a short severe contraction and a longer milder one
    # leave almost the same coalescent signature, because what the genealogy
    # integrates is duration/N_bottleneck.  Grading N_bottleneck on its own
    # would partly be grading which point of that ridge a fit happened to
    # land on.  So the gate is on the identifiable combination, and the
    # reported history must still be a real contraction followed by a
    # recovery (test_bottleneck_is_a_bottleneck).
    "bottleneck_strength_factor": 2.50,
    # The bottleneck's TIMING is not graded at all.  Measured on four
    # independent neutral replicates with known truth, a competent
    # spectrum-based fit recovered the expansion time at 0.64, 0.68, 1.31 and
    # 2.63 times its true value: one in four would fail even a 40% tolerance,
    # and a tolerance loose enough to admit 2.63 is wide enough to be
    # meaningless. The objective has a broad flat minimum spanning roughly
    # 1500 to 4000 generations for a truth of 3000, so this is a property of
    # the information in a site frequency spectrum from 100 haplotypes, not
    # of the search. Methods that use linkage or sequentially-Markov
    # inference date bottlenecks far better, but a gate cannot demand a
    # capability the reference lacks and still be guaranteed passable.
    #
    # What remains is still substantive: both sizes to within 60%, the
    # bottleneck's integrated severity to within a factor of 2.5, and a
    # history that genuinely contracts and recovers. A constant-size answer
    # fails on severity and on structure.

    # --- detection ---
    # Per-class minimum recall, as (numerator, denominator) over the true
    # sweeps of that class.
    "min_recall_per_class": {
        "hard_recent_complete": 3,
        "soft_standing": 2,
        "partial": 2,
        "hard_old_complete": 1,
    },
    # Unmatched calls allowed genome-wide. Cold-spot calls count here.
    "max_false_positives": 1,

    # --- characterisation, over matched calls only ---
    "min_class_accuracy": 0.70,
    "s_within_factor": 3.0,
    "min_s_accuracy": 0.70,
    # Timing is graded on completed sweeps, where fixation time is the
    # estimable quantity, and it is graded PER CLASS.  Graded in aggregate it
    # is satisfiable by a constant: the recent and soft sweeps fix within a
    # few hundred generations of each other, so one number covers six of the
    # eight gradable sweeps and clears any aggregate threshold.  Per class, a
    # constant cannot work, because the old sweeps fix roughly five times
    # further back than the recent ones while the tolerance is ~50%.
    "time_rel_tol": 0.50,
    "time_abs_tol_gens": 150,
    "min_time_accuracy_per_class": 0.50,
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
        if t is not None:
            if not _finite(t):
                raise SchemaError(f"sweeps[{i}]['time_gens_ago'] is neither null nor finite")
            t = float(t)
            if t < 0:
                raise SchemaError(f"sweeps[{i}]['time_gens_ago'] is negative")
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
