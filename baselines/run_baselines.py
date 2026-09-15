"""Reproduce every baseline family and establish ``S*``.

Selection protocol -- deliberately the same information a solving agent has:

  1. Every configuration in the public grid is scored by the PROXY diagnostic
     (settings A and B, proxy worlds, proxy pools, public run seeds).
  2. Each family's submitted configuration is the argmax of that proxy score.
     Every non-ERM family gets the same twelve-configuration budget.
  3. The submitted configurations are then scored OFFICIALLY on A, B and the
     hidden setting C.  ``S*`` is the best of those official scores.

Every configuration is *also* scored officially, on fewer seeds, because the
gates in ``gates/`` need the full proxy-to-official mapping, not just the
selected points.  Those extra numbers are author-side diagnostics and are not
available to an agent.

    python -m baselines.run_baselines [--workers N] [--out reports]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import time
from pathlib import Path

from crossphase.core.engine import quality, run_setting
from crossphase.core.methods import GRIDS
from crossphase.core.parallel import pmap
from crossphase.core.settings import (OFFICIAL_SOURCE_N, OFFICIAL_WORLD_N,
                                      OFFICIAL_WORLDS, PROXY_SOURCE_N,
                                      PROXY_WORLD_N, PROXY_WORLDS,
                                      PUBLIC_POOL_SEED, PUBLIC_RUN_SEEDS)
from grader.private_specs import ALL_SETTINGS, OFFICIAL_POOL_SEED, OFFICIAL_RUN_SEEDS

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Seeds used for the full-grid official diagnostic.  The selected configurations
#: are re-scored on the complete official seed set.
DIAGNOSTIC_SEEDS = OFFICIAL_RUN_SEEDS[:3]


def geo(values) -> float:
    values = list(values)
    if any(v <= 0 for v in values):
        return 0.0
    return 100.0 * math.exp(sum(math.log(v) for v in values) / len(values))


def _job(item):
    kind, family, label, setting, seed = item
    objective = GRIDS[family][label]()
    if kind == "proxy":
        result = run_setting(ALL_SETTINGS[setting], objective, seed, PROXY_WORLDS,
                             PROXY_SOURCE_N, PROXY_WORLD_N,
                             pool_seed=PUBLIC_POOL_SEED)
    else:
        result = run_setting(ALL_SETTINGS[setting], objective, seed, OFFICIAL_WORLDS,
                             OFFICIAL_SOURCE_N, OFFICIAL_WORLD_N,
                             pool_seed=OFFICIAL_POOL_SEED)
    q, binding = quality(result)
    return kind, family, label, setting, seed, q, str(binding), result["I"]


def build_jobs(official_seeds):
    jobs = []
    for family, grid in GRIDS.items():
        for label in grid:
            for seed in PUBLIC_RUN_SEEDS:
                for setting in ("A", "B"):
                    jobs.append(("proxy", family, label, setting, seed))
            for seed in official_seeds:
                for setting in ("A", "B", "C"):
                    jobs.append(("official", family, label, setting, seed))
    return jobs


def collect(rows) -> dict:
    table: dict = collections.defaultdict(
        lambda: {"proxy": collections.defaultdict(list),
                 "official": collections.defaultdict(list),
                 "binding": collections.defaultdict(list),
                 "I": collections.defaultdict(list)})
    for kind, family, label, setting, _seed, q, binding, retained in rows:
        cell = table[(family, label)]
        cell[kind][setting].append(q)
        if kind == "official":
            cell["binding"][setting].append(binding)
            cell["I"][setting].append(retained)
    return table


def summarise(table) -> list:
    out = []
    for (family, label), cell in table.items():
        proxy_q = {s: sum(v) / len(v) for s, v in cell["proxy"].items()}
        official_q = {s: sum(v) / len(v) for s, v in cell["official"].items()}
        out.append({
            "family": family,
            "config": label or "(none)",
            "proxy_score": geo(proxy_q.values()),
            "proxy_Q_A": proxy_q["A"],
            "proxy_Q_B": proxy_q["B"],
            "S": geo(official_q.values()),
            "Q_A": official_q["A"],
            "Q_B": official_q["B"],
            "Q_C": official_q["C"],
            "I_A": sum(cell["I"]["A"]) / len(cell["I"]["A"]),
            "I_B": sum(cell["I"]["B"]) / len(cell["I"]["B"]),
            "I_C": sum(cell["I"]["C"]) / len(cell["I"]["C"]),
            "binding_A": collections.Counter(cell["binding"]["A"]).most_common(1)[0][0],
            "binding_B": collections.Counter(cell["binding"]["B"]).most_common(1)[0][0],
            "binding_C": collections.Counter(cell["binding"]["C"]).most_common(1)[0][0],
            "binding_all": dict(collections.Counter(
                sum((cell["binding"][s] for s in ("A", "B", "C")), []))),
        })
    out.sort(key=lambda r: (-r["S"], r["family"]))
    return out


def rescore_selected(selected, seeds):
    jobs = [("official", fam, label, setting, seed)
            for fam, label in selected.items()
            for setting in ("A", "B", "C") for seed in seeds]
    rows = pmap(_job, jobs)
    per: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    binding: dict = collections.defaultdict(list)
    for _kind, family, _label, setting, _seed, q, bind, _retained in rows:
        per[family][setting].append(q)
        binding[family].append(bind)
    return ({f: {s: sum(v) / len(v) for s, v in d.items()} for f, d in per.items()},
            {f: dict(collections.Counter(v)) for f, v in binding.items()})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args(argv)

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(exist_ok=True)

    jobs = build_jobs(DIAGNOSTIC_SEEDS)
    print(f"{len(jobs)} training runs")
    started = time.time()
    rows = pmap(_job, jobs, workers=args.workers)
    print(f"sweep finished in {time.time() - started:.0f}s")

    table = collect(rows)
    summary = summarise(table)

    with (out_dir / "sweep.csv").open("w", newline="") as fh:
        fields = [k for k in summary[0] if k != "binding_all"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: v for k, v in row.items() if k != "binding_all"})

    # Selection on the proxy alone, per family.
    selected = {}
    for family in GRIDS:
        best = max((r for r in summary if r["family"] == family),
                   key=lambda r: r["proxy_score"])
        selected[family] = "" if best["config"] == "(none)" else best["config"]

    official_q, binding = rescore_selected(selected, OFFICIAL_RUN_SEEDS)
    families = []
    for family, label in selected.items():
        proxy_row = next(r for r in summary if r["family"] == family
                         and r["config"] == (label or "(none)"))
        families.append({
            "family": family,
            "selected_config": label or "(none)",
            "proxy_score": proxy_row["proxy_score"],
            "Q_A": official_q[family]["A"],
            "Q_B": official_q[family]["B"],
            "Q_C": official_q[family]["C"],
            "S": geo(official_q[family].values()),
            "binding_worlds": binding[family],
        })
    families.sort(key=lambda r: -r["S"])

    with (out_dir / "baselines.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["family", "selected_config", "proxy_score", "Q_A", "Q_B",
                         "Q_C", "S", "binding_world_counts"])
        for row in families:
            writer.writerow([row["family"], row["selected_config"],
                             f"{row['proxy_score']:.4f}", f"{row['Q_A']:.4f}",
                             f"{row['Q_B']:.4f}", f"{row['Q_C']:.4f}",
                             f"{row['S']:.4f}",
                             "; ".join(f"{k}x{v}" for k, v in
                                       sorted(row["binding_worlds"].items()))])

    payload = {
        "S_star": families[0]["S"],
        "strongest": families[0]["family"],
        "strongest_config": families[0]["selected_config"],
        "weakest": families[-1]["family"],
        "weakest_config": families[-1]["selected_config"],
        "weakest_S": families[-1]["S"],
        "spread": families[0]["S"] - families[-1]["S"],
        "official_seeds": list(OFFICIAL_RUN_SEEDS),
        "diagnostic_seeds": list(DIAGNOSTIC_SEEDS),
        "families": families,
    }
    (out_dir / "baselines.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "sweep.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'family':10s} {'config':22s} {'proxy':>7s} {'Q_A':>6s} {'Q_B':>6s} "
          f"{'Q_C':>6s} {'S':>7s}")
    for row in families:
        print(f"{row['family']:10s} {row['selected_config']:22s} "
              f"{row['proxy_score']:7.2f} {row['Q_A']:6.3f} {row['Q_B']:6.3f} "
              f"{row['Q_C']:6.3f} {row['S']:7.3f}")
    print(f"\nS* = {payload['S_star']:.3f} ({payload['strongest']})   "
          f"weakest = {payload['weakest_S']:.3f} ({payload['weakest']})   "
          f"spread = {payload['spread']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
