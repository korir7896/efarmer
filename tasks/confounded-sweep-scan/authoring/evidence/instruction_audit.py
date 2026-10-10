#!/usr/bin/env python3
"""Check that every threshold instruction.md states equals the one graded.

The instruction is the contract: an agent that satisfies exactly what it says
must pass. A threshold that drifts in one file and not the other turns the
task unfair in a way no test would catch, and it has happened here once
already -- the instruction kept quoting a rank-correlation floor of 0.50
after the verifier had moved to 0.30, and separately omitted the spread gate
entirely.

This parses the numbers back OUT of the prose and compares them as floats to
tests/grading.GATES, so wording may change freely but the numbers may not
disagree. Run it from authoring/validate.sh.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(TASK, "tests"))

from grading import GATES

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _num(tok: str) -> float:
    tok = tok.strip().lower().replace(",", "").rstrip(".")
    return float(WORDS[tok]) if tok in WORDS else float(tok)


def _one(text: str, pattern: str, label: str, expected: float,
         errors: list) -> None:
    m = re.search(pattern, text, re.I)
    if not m:
        errors.append(f"{label}: instruction.md has no text matching "
                      f"/{pattern}/")
        return
    got = _num(m.group(1))
    if abs(got - float(expected)) > 1e-9:
        errors.append(f"{label}: instruction.md says {got:g}, "
                      f"grading.GATES says {float(expected):g}")


def main() -> int:
    text = open(os.path.join(TASK, "instruction.md")).read()
    errors: list = []
    rc = GATES["min_recall_per_class"]

    _one(text, r"N_ancestral and N_current must each be within a factor of\s+([\d.]+)",
         "population-size factor", GATES["N_ancestral_factor"], errors)
    _one(text, r"/ N_bottleneck within a factor of\s+([\d.]+)",
         "bottleneck severity factor", GATES["bottleneck_strength_factor"], errors)
    _one(text, r"reported sweeps closer together than\s+([\d]+)\s*kb",
         "dedupe radius (kb)", GATES["dedupe_bp"] / 1000, errors)
    _one(text, r"same chromosome and within\s+([\d]+)\s*kb",
         "match window (kb)", GATES["match_window_bp"] / 1000, errors)
    _one(text, r"At least (\w+) of the true hard_complete sweeps must be found",
         "hard_complete recall", rc["hard_complete"], errors)
    _one(text, r"at least (\w+) true sweeps of each of the other three classes",
         "other-class recall", rc["soft_complete"], errors)
    _one(text, r"At most (\w+) report may fail to match",
         "false-positive cap", GATES["max_false_positives"], errors)
    _one(text, r"at least ([\d]+) percent must carry the right class",
         "class accuracy", GATES["min_class_accuracy"] * 100, errors)
    _one(text, r"At least ([\d]+) percent of the reported selection coefficients",
         "s magnitude fraction", GATES["min_s_accuracy"] * 100, errors)
    _one(text, r"selection coefficients must be within a factor of\s+([\d.]+)",
         "s magnitude factor", GATES["s_within_factor"], errors)
    _one(text, r"divided by the smallest must be at least\s+([\d.]+)",
         "s spread", GATES["min_s_spread"], errors)
    _one(text, r"true coefficients must be at least\s+([\d.]+)",
         "s rank correlation", GATES["min_s_rank_corr"], errors)
    _one(text, r"at least (\w+) of them must be dated",
         "timing count", GATES["min_time_correct"], errors)
    _one(text, r"dated to within ([\d]+) percent of the true completion time",
         "timing relative tolerance", GATES["time_rel_tol"] * 100, errors)
    _one(text, r"or within ([\d]+) generations",
         "timing absolute tolerance", GATES["time_abs_tol_gens"], errors)

    # The other three classes share one sentence, so check they really do
    # share one threshold -- otherwise the prose cannot be accurate.
    others = {k: v for k, v in rc.items() if k != "hard_complete"}
    if len(set(others.values())) != 1:
        errors.append(f"the three non-hard_complete recall gates differ "
                      f"({others}), so one sentence cannot state them all")

    # The class names the instruction defines must be exactly the graded set.
    named = set(re.findall(r'"(hard_complete|hard_partial|soft_complete|'
                           r'soft_partial)"', text))
    if named != set(rc):
        errors.append(f"instruction.md names {sorted(named)} but the verifier "
                      f"grades {sorted(rc)}")

    if errors:
        print("instruction.md does NOT agree with the verifier:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"instruction.md agrees with grading.GATES on all 15 thresholds "
          f"and all four class names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
