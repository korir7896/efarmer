"""Official CrossPhase grader.

    python -m grader.grade [--solution agent/solution.py] [--json out.json]

Trains the submitted objective on all three settings -- A, B and the hidden
setting C -- with the official run seeds, scores each against the five official
target worlds drawn from the private pool seed, and reports

    Q_s = sqrt(I_s * min_j T_{s,j})            per setting
    S   = 100 * (Q_A * Q_B * Q_C)^(1/3)        combined

A submission passes when ``S`` strictly exceeds the strongest reproduced
baseline, ``S*``.  There is no pass margin: the measurement is deterministic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

from crossphase.core.engine import ObjectiveError, combine, quality, run_setting
from crossphase.core.parallel import pmap
from crossphase.core.settings import (OFFICIAL_SOURCE_N, OFFICIAL_WORLD_N,
                                      OFFICIAL_WORLDS)

from .private_specs import ALL_SETTINGS, OFFICIAL_POOL_SEED, OFFICIAL_RUN_SEEDS

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_objective(path: str):
    """Import ``objective`` from a submission file."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    spec = importlib.util.spec_from_file_location("cp_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "objective"):
        raise ObjectiveError(f"{path} does not define objective(...)")
    return module.objective


def official_score(objective, seeds=OFFICIAL_RUN_SEEDS, workers=None) -> dict:
    """Score one objective on all three settings.  A run that raises scores zero."""

    def job(args):
        name, seed = args
        try:
            result = run_setting(ALL_SETTINGS[name], objective, seed,
                                 OFFICIAL_WORLDS, OFFICIAL_SOURCE_N,
                                 OFFICIAL_WORLD_N, pool_seed=OFFICIAL_POOL_SEED)
            q, binding = quality(result)
            return name, seed, q, binding, result["I"], None
        except Exception:  # noqa: BLE001 - a broken objective must score, not crash
            return name, seed, 0.0, None, 0.0, traceback.format_exc(limit=3)

    rows = pmap(job, [(n, s) for n in ALL_SETTINGS for s in seeds], workers=workers)

    per_setting: dict = {n: [] for n in ALL_SETTINGS}
    bindings: dict = {n: [] for n in ALL_SETTINGS}
    retention: dict = {n: [] for n in ALL_SETTINGS}
    errors = []
    for name, seed, q, binding, retained, error in rows:
        per_setting[name].append(q)
        bindings[name].append(str(binding))
        retention[name].append(retained)
        if error:
            errors.append(f"[{name} seed={seed}]\n{error}")

    means = {n: sum(v) / len(v) for n, v in per_setting.items()}
    return {
        "Q": means,
        "Q_per_seed": per_setting,
        "I": {n: sum(v) / len(v) for n, v in retention.items()},
        "binding_worlds": bindings,
        "S": combine(means.values()),
        "seeds": list(seeds),
        "errors": errors,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Grade a CrossPhase submission.")
    parser.add_argument("--solution", default="agent/solution.py")
    parser.add_argument("--json", default=None, help="write the full report here")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--target", type=float, default=None,
                        help="S* to compare against; read from reports/baselines.json "
                             "when omitted")
    args = parser.parse_args(argv)

    report = official_score(load_objective(args.solution), workers=args.workers)

    target = args.target
    if target is None:
        summary_path = REPO_ROOT / "reports" / "baselines.json"
        if summary_path.exists():
            target = json.loads(summary_path.read_text())["S_star"]

    print(f"solution: {args.solution}")
    for name in sorted(report["Q"]):
        per_seed = " ".join(f"{q:.4f}" for q in report["Q_per_seed"][name])
        print(f"  Q_{name} = {report['Q'][name]:.4f}   "
              f"(I={report['I'][name]:.4f}; seeds: {per_seed})")
        print(f"          binding worlds: {', '.join(report['binding_worlds'][name])}")
    print(f"  S = {report['S']:.3f}")
    if target is not None:
        verdict = "PASS" if report["S"] > target else "FAIL"
        print(f"  S* = {target:.3f}  ->  {verdict}")
        report["S_star"] = target
        report["pass"] = report["S"] > target
    for err in report["errors"]:
        print(err, file=sys.stderr)

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
