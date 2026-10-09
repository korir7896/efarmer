"""Grade a genome scan submission against the sealed truth.

Every gate is a separate test so the CTRF report names what failed.  The
reward is 1 only if all of them pass; test.sh derives it from pytest's exit
status.
"""
from __future__ import annotations

import json
import os

import pytest

from grading import (COMPLETE_CLASSES, GATES, SchemaError, dedupe, match,
                     parse_submission)

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
    ("N_bottleneck", "N_bottleneck_factor"),
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


@pytest.mark.parametrize("key,gate", [
    ("T_bottleneck_start_gens_ago", "T_bottleneck_start_frac"),
    ("T_bottleneck_end_gens_ago", "T_bottleneck_end_frac"),
])
def test_demography_times(truth, parsed, key, gate):
    dem, _ = parsed
    true_v = float(truth["demography"][key])
    got = float(dem[key])
    frac = GATES[gate]
    tol = frac * true_v
    assert abs(got - true_v) <= tol, (
        f"{key}: reported {got:.0f}, true {true_v:.0f}; "
        f"need within {tol:.0f} generations")


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


def test_selection_coefficients(matched):
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


def test_timing_of_completed_sweeps(truth, matched):
    """Fixation time, graded separately within each completed class.

    Grading in aggregate would accept a single constant (see GATES).
    """
    pairs, _, _ = matched
    rel = GATES["time_rel_tol"]
    abs_tol = GATES["time_abs_tol_gens"]
    need = GATES["min_time_accuracy_per_class"]

    by_class = {}
    for t, c, _ in pairs:
        if t["class"] not in COMPLETE_CLASSES or t["fixation_gens_ago"] is None:
            continue
        by_class.setdefault(t["class"], []).append((t, c))
    if not by_class:
        pytest.fail("no completed sweep was localized, so timing cannot be graded")

    problems = []
    for klass, items in sorted(by_class.items()):
        ok, bad = 0, []
        for t, c in items:
            true_t = float(t["fixation_gens_ago"])
            tol = max(abs_tol, rel * true_t)
            if c["time"] is not None and abs(c["time"] - true_t) <= tol:
                ok += 1
            else:
                bad.append((t["sweep_id"], true_t, c["time"]))
        required = max(1, int(round(need * len(items))))
        if ok < required:
            problems.append(
                f"{klass}: {ok}/{len(items)} fixation times within tolerance, "
                f"need {required}; off (id, true, reported): {bad}")
    assert not problems, "; ".join(problems)
