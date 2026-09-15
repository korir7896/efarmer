"""Reproduce every baseline family and establish ``S*``.

Protocol
--------
Every configuration in the public grid is scored two ways: by the PROXY
diagnostic (settings A and B, proxy worlds, proxy pools, public run seeds --
exactly the information a solving agent has), and OFFICIALLY on A, B and the
hidden setting C.

``S*`` is then each family's **best official score**, re-measured on the full
official seed set, maximised over families.  That is "the strongest reproduced
method" read strictly: the bar is what the best published objective actually
achieves here, not what a weak validation signal happens to pick for it.

The alternative -- letting the proxy choose each family's configuration -- was
measured and rejected.  It lowers the bar by about a point, because the proxy
correlates only weakly with the official score among the competent
configurations, and at that lower bar a sixth of the published grid clears it
without any adaptation at all.  Both numbers are reported: ``S_star`` is the
bar, ``S_selected`` records what proxy selection would have produced, and the
difference is the measured cost of tuning on what is visible.

Correspondingly, the configuration shipped in ``agent/solution.py`` is the
WEAKEST family at ITS best official configuration, so that the two ends of the
table are measured the same way.

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
DIAGNOSTIC_SEEDS = OFFICIAL_RUN_SEEDS[:2]


def geo(values) -> float:
    values = list(values)
    if any(v <= 0 for v in values):
        return 0.0
    return 100.0 * math.exp(sum(math.log(v) for v in values) / len(values))


def _job(item):
    """A configuration that blows up scores zero, exactly as the grader treats a
    submission that blows up.  It must not take the sweep down with it."""
    try:
        return _run(item)
    except Exception:  # noqa: BLE001
        kind, family, label, setting, seed = item
        return kind, family, label, setting, seed, 0.0, "error", 0.0


def _run(item):
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


def rescore_pairs(pairs, seeds):
    """Re-measure explicit (family, config) pairs on the full official seed set."""
    jobs = [("official", fam, label, setting, seed) for fam, label in pairs
            for setting in ("A", "B", "C") for seed in seeds]
    rows = pmap(_job, jobs)
    per: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    binding: dict = collections.defaultdict(list)
    for _kind, family, label, setting, _seed, q, bind, _retained in rows:
        per[(family, label)][setting].append(q)
        binding[(family, label)].append(bind)
    return ({k: {s: sum(v) / len(v) for s, v in d.items()} for k, d in per.items()},
            {k: dict(collections.Counter(v)) for k, v in binding.items()})


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
    parser.add_argument("--reuse-sweep", action="store_true",
                        help="re-summarise from an existing reports/sweep.json "
                             "instead of repeating the full grid")
    args = parser.parse_args(argv)

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(exist_ok=True)

    if args.reuse_sweep and (out_dir / "sweep.json").exists():
        summary = json.loads((out_dir / "sweep.json").read_text())
        stale = [r for r in summary
                 if r["family"] not in GRIDS
                 or ("" if r["config"] == "(none)" else r["config"]) not in GRIDS[r["family"]]]
        if stale:
            raise SystemExit(
                f"reports/sweep.json holds {len(stale)} configurations that are no "
                f"longer in the grid (e.g. {stale[0]['family']} "
                f"{stale[0]['config']}).  It predates the current method grid; "
                f"re-run without --reuse-sweep.")
        print(f"reusing {len(summary)} cached grid rows")
    else:
        jobs = build_jobs(DIAGNOSTIC_SEEDS)
        print(f"{len(jobs)} training runs")
        started = time.time()
        rows = pmap(_job, jobs, workers=args.workers)
        print(f"sweep finished in {time.time() - started:.0f}s")
        summary = summarise(collect(rows))
        # Persist the grid BEFORE anything that can fail.  These are hours of
        # training runs and nothing downstream is worth losing them to.
        (out_dir / "sweep.json").write_text(json.dumps(summary, indent=2))
        with (out_dir / "sweep.csv").open("w", newline="") as fh:
            fields = [k for k in summary[0] if k != "binding_all"]
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in summary:
                writer.writerow({k: v for k, v in row.items() if k != "binding_all"})

    def argmax_per_family(key):
        out = {}
        for family in GRIDS:
            best = max((r for r in summary if r["family"] == family), key=key)
            out[family] = "" if best["config"] == "(none)" else best["config"]
        return out

    def top_k_per_family(key, k=2):
        out = []
        for family in GRIDS:
            rows = sorted((r for r in summary if r["family"] == family),
                          key=key, reverse=True)[:k]
            out += [(family, "" if r["config"] == "(none)" else r["config"])
                    for r in rows]
        return out

    # The grid is measured on few seeds, so the top candidates of every family
    # are re-measured on the full official seed set and the winner is taken there.
    finalists = top_k_per_family(lambda r: r["S"])
    final_q, final_binding = rescore_pairs(finalists, OFFICIAL_RUN_SEEDS)
    best_configs = {}
    for family in GRIDS:
        cands = [(f, c) for f, c in finalists if f == family]
        best_configs[family] = max(cands, key=lambda fc: geo(final_q[fc].values()))[1]
    # The diagnostic: what tuning on the visible proxy would have chosen instead.
    proxy_configs = argmax_per_family(lambda r: r["proxy_score"])

    official_q = {f: final_q[(f, c)] for f, c in best_configs.items()}
    binding = {f: final_binding[(f, c)] for f, c in best_configs.items()}
    proxy_q, _ = rescore_selected(proxy_configs, OFFICIAL_RUN_SEEDS)

    families = []
    for family, label in best_configs.items():
        row = next(r for r in summary if r["family"] == family
                   and r["config"] == (label or "(none)"))
        families.append({
            "family": family,
            "best_config": label or "(none)",
            "proxy_selected_config": proxy_configs[family] or "(none)",
            "proxy_score": row["proxy_score"],
            "Q_A": official_q[family]["A"],
            "Q_B": official_q[family]["B"],
            "Q_C": official_q[family]["C"],
            "S": geo(official_q[family].values()),
            "S_if_selected_on_proxy": geo(proxy_q[family].values()),
            "binding_worlds": binding[family],
        })
    families.sort(key=lambda r: -r["S"])

    with (out_dir / "baselines.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["family", "best_config", "Q_A", "Q_B", "Q_C", "S",
                         "proxy_selected_config", "S_if_selected_on_proxy",
                         "binding_world_counts"])
        for row in families:
            writer.writerow([row["family"], row["best_config"], f"{row['Q_A']:.4f}",
                             f"{row['Q_B']:.4f}", f"{row['Q_C']:.4f}",
                             f"{row['S']:.4f}", row["proxy_selected_config"],
                             f"{row['S_if_selected_on_proxy']:.4f}",
                             "; ".join(f"{k}x{v}" for k, v in
                                       sorted(row["binding_worlds"].items()))])

    payload = {
        "S_star": families[0]["S"],
        "strongest": families[0]["family"],
        "strongest_config": families[0]["best_config"],
        "weakest": families[-1]["family"],
        "weakest_config": families[-1]["best_config"],
        "weakest_S": families[-1]["S"],
        "spread": families[0]["S"] - families[-1]["S"],
        "S_star_if_selected_on_proxy": max(r["S_if_selected_on_proxy"]
                                           for r in families),
        "configs_in_grid_above_S_star": sum(
            1 for r in summary if r["S"] > families[0]["S"]),
        "grid_size": len(summary),
        "official_seeds": list(OFFICIAL_RUN_SEEDS),
        "diagnostic_seeds": list(DIAGNOSTIC_SEEDS),
        "families": families,
    }
    (out_dir / "baselines.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "sweep.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'family':10s} {'best config':22s} {'Q_A':>6s} {'Q_B':>6s} "
          f"{'Q_C':>6s} {'S':>7s} {'S(proxy-sel)':>13s}")
    for row in families:
        print(f"{row['family']:10s} {row['best_config']:22s} "
              f"{row['Q_A']:6.3f} {row['Q_B']:6.3f} {row['Q_C']:6.3f} "
              f"{row['S']:7.3f} {row['S_if_selected_on_proxy']:13.3f}")
    print(f"\nS* = {payload['S_star']:.3f} ({payload['strongest']})   "
          f"weakest = {payload['weakest_S']:.3f} ({payload['weakest']})   "
          f"spread = {payload['spread']:.3f}")
    print(f"proxy selection would have set the bar at "
          f"{payload['S_star_if_selected_on_proxy']:.3f}; "
          f"{payload['configs_in_grid_above_S_star']} grid configurations exceed S*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
