"""Small process-pool helper used by the author-side sweeps and the grader.

The mapped callable is handed to the workers by ``fork`` inheritance rather than
by pickling, so closures and locally defined functions work.
"""

from __future__ import annotations

import multiprocessing as mp
import os

_FN = None


def _init_worker():
    import torch
    torch.set_num_threads(1)


def _apply(item):
    return _FN(item)


def pmap(fn, items, workers: int | None = None):
    global _FN
    _FN = fn
    items = list(items)
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2))
    workers = min(workers, max(1, len(items)))
    if workers <= 1:
        _init_worker()
        return [fn(i) for i in items]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        return pool.map(_apply, items, chunksize=1)
