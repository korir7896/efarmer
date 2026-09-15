"""Shared measurement protocol.

``proxy_score`` is the public diagnostic an agent may run.  ``grader/grade.py``
implements the official counterpart with the private pools and the hidden setting.
"""

from __future__ import annotations

from .engine import combine, quality, run_setting
from .parallel import pmap
from .settings import (PROXY_SOURCE_N, PROXY_WORLD_N, PROXY_WORLDS,
                       PUBLIC_POOL_SEED, PUBLIC_RUN_SEEDS, PUBLIC_SETTINGS)

def score_settings(objective, specs, worlds, source_n, world_n, seeds,
                   pool_seed, workers=None) -> dict:
    """Mean ``Q`` per setting plus the combined geometric-mean score."""
    jobs = [(name, seed) for name in specs for seed in seeds]

    def run(args):
        name, seed = args
        result = run_setting(specs[name], objective, seed, worlds,
                             source_n, world_n, pool_seed=pool_seed)
        return name, seed, result

    rows = pmap(run, jobs, workers=workers)

    per_setting: dict = {name: [] for name in specs}
    bindings: dict = {name: [] for name in specs}
    for name, _seed, result in rows:
        q, binding = quality(result)
        per_setting[name].append(q)
        bindings[name].append(binding)

    means = {n: sum(v) / len(v) for n, v in per_setting.items()}
    return {
        "per_setting": means,
        "per_seed": per_setting,
        "bindings": bindings,
        "score": combine(means.values()),
    }


def proxy_score(objective, seeds=PUBLIC_RUN_SEEDS, workers=None) -> dict:
    """The public diagnostic: settings A and B, proxy worlds, proxy pools.

    The number it returns is NOT on the official scale and is not comparable to
    the published baseline target.  It is for relative comparison between your own
    candidate objectives, nothing else.
    """
    return score_settings(objective, PUBLIC_SETTINGS, PROXY_WORLDS,
                          PROXY_SOURCE_N, PROXY_WORLD_N, seeds,
                          PUBLIC_POOL_SEED, workers=workers)
