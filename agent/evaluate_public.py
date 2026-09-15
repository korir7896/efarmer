"""Public diagnostic for a CrossPhase submission.

    python -m agent.evaluate_public [--solution agent/solution.py] [--note "..."]

WHAT THIS MEASURES
------------------
Your objective is trained on the two public settings, A and B, and scored against
the PROXY worlds using the PROXY pools:

    proxy_score = 100 * (Qp_A * Qp_B)^(1/2),   Qp = sqrt(I * min_j T_j)

WHAT THIS IS NOT
----------------
It is **not** the official score and it is **not** comparable to the baseline
target quoted in the README.  The proxy worlds sit at different cue-agreement
coordinates from the official ones, they are drawn from a different seed with
smaller pools, there are four of them rather than five, and the hidden third
setting contributes nothing to this number.  Read it as a relative ranking
between your own candidates and nothing more.  Chasing the README's number with
this diagnostic is a mistake -- you cannot measure that number from here.

BUDGET
------
Twelve evaluations per trial.  Usage is recorded in ``agent/.eval_budget.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from crossphase.core.protocol import proxy_score
from crossphase.core.settings import PROXY_WORLDS, PUBLIC_RUN_SEEDS

REPO_ROOT = Path(__file__).resolve().parents[1]
BUDGET_FILE = REPO_ROOT / "agent" / ".eval_budget.json"
BUDGET = 12


def _load_budget() -> dict:
    if BUDGET_FILE.exists():
        return json.loads(BUDGET_FILE.read_text())
    return {"limit": BUDGET, "used": 0, "log": []}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--solution", default="agent/solution.py")
    parser.add_argument("--note", default="", help="label recorded in the budget log")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--show-budget", action="store_true",
                        help="print remaining budget and exit")
    args = parser.parse_args(argv)

    state = _load_budget()
    if args.show_budget:
        print(f"used {state['used']} / {state['limit']}")
        for entry in state["log"]:
            print(f"  #{entry['n']:2d}  {entry['proxy_score']:7.3f}  {entry['note']}")
        return 0

    if state["used"] >= state["limit"]:
        print(f"evaluation budget exhausted ({state['limit']} used)")
        return 1

    # Imported here so that --show-budget stays cheap.
    from grader.grade import load_objective

    started = time.time()
    report = proxy_score(load_objective(args.solution), workers=args.workers)

    state["used"] += 1
    state["log"].append({"n": state["used"], "note": args.note,
                         "proxy_score": report["score"]})
    BUDGET_FILE.write_text(json.dumps(state, indent=2))

    print(f"proxy_score = {report['score']:.3f}   "
          f"(NOT the official scale -- relative comparison only)")
    for name in sorted(report["per_setting"]):
        per_seed = " ".join(f"{q:.4f}" for q in report["per_seed"][name])
        print(f"  Qp_{name} = {report['per_setting'][name]:.4f}   seeds: {per_seed}")
        print(f"          binding proxy worlds: "
              f"{', '.join(str(b) for b in report['bindings'][name])}")
    print(f"  run seeds {list(PUBLIC_RUN_SEEDS)}, "
          f"{len(PROXY_WORLDS)} proxy worlds, {time.time() - started:.0f}s")
    print(f"  evaluations used: {state['used']} / {state['limit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
