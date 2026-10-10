"""Grade a genome scan submission against the sealed truth.

Every gate is a separate test so the CTRF report names what failed.  The
reward is 1 only if all of them pass; test.sh derives it from pytest's exit
status.
"""
from __future__ import annotations

import json
import os

import pytest

from grading import (GATES, TIMED_CLASSES, SchemaError, dedupe, match,
                     parse_submission, spearman, time_tolerance)

TRUTH_PATH = os.environ.get("TRUTH_PATH", "/tests/truth.json")
SUBMISSION_PATH = os.environ.get("SUBMISSION_PATH", "/app/results.json")


def _load_truth():
    with open(TRUTH_PATH) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def truth():
    return _load_truth()


@pytest.fixture(scope="module")
def parsed(truth):
    if not os.path.exists(SUBMISSION_PATH):
        pytest.fail(f"no submission at {SUBMISSION_PATH}")
    try:
        with open(SUBMISSION_PATH) as fh:
            obj = json.load(fh)
    except json.JSONDecodeError as e:
        pytest.fail(f"{SUBMISSION_PATH} is not valid JSON: {e}")
    try:
        dem, calls = parse_submission(obj, truth["genome"]["n_chromosomes"],
                                      truth["genome"]["chromosome_length_bp"])
    except SchemaError as e:
        pytest.fail(f"submission schema invalid: {e}")
    return dem, dedupe(calls, GATES["dedupe_bp"])


@pytest.fixture(scope="module")
def matched(truth, parsed):
    _, calls = parsed
    return match(truth["sweeps"], calls, GATES["match_window_bp"])


# ---------------------------------------------------------------- schema ---

def test_submission_is_well_formed(parsed):
    dem, calls = parsed
    assert calls, "no sweeps reported"


# ------------------------------------------------------------ demography ---

@pytest.mark.parametrize("key,gate", [
    ("N_ancestral", "N_ancestral_factor"),
    ("N_current", "N_current_factor"),
])
def test_demography_sizes(truth, parsed, key, gate):
    dem, _ = parsed
    true_v = float(truth["demography"][key])
    got = float(dem[key])
    f = GATES[gate]
    lo, hi = true_v / f, true_v * f
    assert lo <= got <= hi, (
        f"{key}: reported {got:.0f}, true {true_v:.0f}; "
        f"need within a factor of {f} ([{lo:.0f}, {hi:.0f}])")


def test_bottleneck_strength(truth, parsed):
    """Grade duration/N_bottleneck, the combination the genealogy determines."""
    dem, _ = parsed
    td = truth["demography"]
    true_strength = ((float(td["T_bottleneck_start_gens_ago"])
                      - float(td["T_bottleneck_end_gens_ago"]))
                     / float(td["N_bottleneck"]))
    dur = float(dem["T_bottleneck_start_gens_ago"]) - float(dem["T_bottleneck_end_gens_ago"])
    if dur <= 0:
        pytest.fail("reported bottleneck has non-positive duration")
    got = dur / float(dem["N_bottleneck"])
    f = GATES["bottleneck_strength_factor"]
    assert true_strength / f <= got <= true_strength * f, (
        f"bottleneck strength (duration/N_bottleneck) reported {got:.4g}, "
        f"true {true_strength:.4g}; need within a factor of {f}")


def test_bottleneck_is_a_bottleneck(truth, parsed):
    """The reported history must actually contract and then expand."""
    dem, _ = parsed
    assert float(dem["N_bottleneck"]) < float(dem["N_ancestral"]), (
        "reported N_bottleneck is not below N_ancestral")
    assert float(dem["N_bottleneck"]) < float(dem["N_current"]), (
        "reported N_bottleneck is not below N_current")
    assert float(dem["T_bottleneck_start_gens_ago"]) > float(dem["T_bottleneck_end_gens_ago"]), (
        "bottleneck start must be older than its end")


# -------------------------------------------------------------- detection ---

def test_per_class_recall(truth, matched):
    pairs, unmatched_truth, _ = matched
    found = {}
    for t, c, d in pairs:
        found[t["class"]] = found.get(t["class"], 0) + 1
    problems = []
    for klass, need in GATES["min_recall_per_class"].items():
        total = sum(1 for t in truth["sweeps"] if t["class"] == klass)
        got = found.get(klass, 0)
        if got < need:
            problems.append(f"{klass}: localized {got}/{total}, need at least {need}")
    assert not problems, "; ".join(problems) + (
        f"  [missed: {[t['sweep_id'] for t in unmatched_truth]}]")


def test_false_positives(matched):
    _, _, unmatched_calls = matched
    n = len(unmatched_calls)
    cap = GATES["max_false_positives"]
    where = [(c["chrom"], c["position"]) for c in unmatched_calls]
    assert n <= cap, (
        f"{n} reported sweeps match no true sweep (limit {cap}): {where}")


# -------------------------------------------------------- characterisation ---

def test_class_accuracy(matched):
    pairs, _, _ = matched
    assert pairs, "nothing matched, so class accuracy is undefined"
    right = sum(1 for t, c, _ in pairs if c["class"] == t["class"])
    acc = right / len(pairs)
    need = GATES["min_class_accuracy"]
    wrong = [(t["sweep_id"], t["class"], c["class"]) for t, c, _ in pairs
             if c["class"] != t["class"]]
    assert acc >= need, (
        f"class accuracy {right}/{len(pairs)} = {acc:.2f}, need {need:.2f}; "
        f"wrong: {wrong}")


def test_selection_coefficient_magnitudes(matched):
    """Reported s within a multiplicative factor of the truth."""
    pairs, _, _ = matched
    assert pairs, "nothing matched, so s accuracy is undefined"
    f = GATES["s_within_factor"]
    ok, bad = 0, []
    for t, c, _ in pairs:
        true_s = float(t["selection_coefficient"])
        if true_s / f <= c["s"] <= true_s * f:
            ok += 1
        else:
            bad.append((t["sweep_id"], round(true_s, 4), round(c["s"], 4)))
    acc = ok / len(pairs)
    need = GATES["min_s_accuracy"]
    assert acc >= need, (
        f"s within a factor of {f} for {ok}/{len(pairs)} = {acc:.2f}, "
        f"need {need:.2f}; off: {bad}")


def test_selection_coefficients_vary(matched):
    """The reported coefficients must span a factor, not be one number.

    A constant spans 1.0 and fails this with certainty, on any match set and
    whatever its value. That is what this test is for; the magnitude test
    above is satisfiable by a well-chosen constant (see GATES).
    """
    pairs, _, _ = matched
    assert pairs, "nothing matched, so the spread of s is undefined"
    got = [c["s"] for t, c, _ in pairs]
    need = GATES["min_s_spread"]
    spread = max(got) / min(got) if min(got) > 0 else float("inf")
    assert spread >= need, (
        f"reported selection coefficients span a factor of {spread:.2f} "
        f"across {len(pairs)} matched calls, need {need:.2f}; "
        f"reported {sorted(round(v, 4) for v in got)}")


def test_selection_coefficients_are_ordered(matched):
    """Reported s must rank the matched sweeps the way the truth ranks them.

    A weak requirement by design: at nine matched calls a rank correlation
    has a null standard deviation of about 0.35, so this threshold is a floor
    that a constant fails with certainty rather than a strong test of
    ordering. See GATES for why the deterministic alternative was rejected.
    """
    pairs, _, _ = matched
    need_n = GATES["min_matched_for_rank"]
    assert len(pairs) >= need_n, (
        f"only {len(pairs)} matched calls; the ordering of s is graded over "
        f"at least {need_n}")
    got = [c["s"] for t, c, _ in pairs]
    tru = [float(t["selection_coefficient"]) for t, c, _ in pairs]
    rho = spearman(got, tru)
    need = GATES["min_s_rank_corr"]
    assert rho >= need, (
        f"rank correlation between reported and true s is {rho:.3f} over "
        f"{len(pairs)} matched sweeps, need {need:.2f}; "
        f"reported {[round(v, 4) for v in got]} against true "
        f"{[round(v, 4) for v in tru]}")


def test_timing_of_completed_sweeps(truth, matched):
    """Completion time of the sweeps whose TRUE class is hard_complete.

    Graded against the true class so that a misclassification is not punished
    twice, and pooled rather than split by class: class no longer carries any
    information about age, and the four true completion times are spread so
    that no single number lands inside more than two of their tolerance
    windows (see GATES).
    """
    pairs, _, _ = matched
    items = [(t, c) for t, c, _ in pairs
             if t["class"] in TIMED_CLASSES
             and t.get("completion_gens_ago") is not None]
    if not items:
        pytest.fail("no datable sweep was localized, so timing cannot be graded")

    ok, bad = 0, []
    for t, c in items:
        true_t = float(t["completion_gens_ago"])
        tol = time_tolerance(true_t)
        if c["time"] is not None and abs(c["time"] - true_t) <= tol:
            ok += 1
        else:
            bad.append((t["sweep_id"], true_t, c["time"], round(tol)))
    need = GATES["min_time_correct"]
    assert ok >= need, (
        f"{ok} of {len(items)} localized hard_complete sweeps dated within "
        f"tolerance, need {need}; off (id, true, reported, tolerance): {bad}")
