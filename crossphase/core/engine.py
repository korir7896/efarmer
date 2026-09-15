"""Frozen training loop, evaluation and scoring for CrossPhase.

Nothing in this module may be edited by a solving agent.  The optimiser, the
schedule, the batch composition and the metric definitions are identical for
every method and every setting; the only thing that varies between submissions
is the ``objective`` callable handed to :func:`train_model`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from .generator import SettingSpec, source_pool, training_environments, world_pool
from .model import build_model


@dataclass(frozen=True)
class TrainConfig:
    """The frozen optimisation budget.

    ``weight_decay`` is deliberately zero.  Adam's update is scale-invariant up
    to epsilon, but *decoupled* weight decay is not, so a non-zero value would
    turn "multiply the loss by 10" into a real regularisation-strength knob that
    an objective could reach with a one-character edit.
    """

    steps: int = 500
    batch_per_env: int = 64
    lr: float = 3e-3
    weight_decay: float = 0.0


TRAIN = TrainConfig()

# Pools are built once per (spec, size, world, seed) and reused.  The key is the
# whole SettingSpec, not its name: author-side sweeps build variant specs that
# share a name while differing in label noise, amplitude or phase lag, and a
# name-keyed cache would silently hand every variant the first one's pools.
_POOL_CACHE: dict = {}


def _as_tensors(x: np.ndarray, y: np.ndarray) -> tuple:
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))


def get_source_pool(spec: SettingSpec, n: int, seed: int) -> tuple:
    key = ("src", spec, n, seed)
    if key not in _POOL_CACHE:
        _POOL_CACHE[key] = _as_tensors(*source_pool(spec, n, seed))
    return _POOL_CACHE[key]


def get_world_pool(spec: SettingSpec, n: int, world: tuple, seed: int) -> tuple:
    key = ("world", spec, n, world, seed)
    if key not in _POOL_CACHE:
        _POOL_CACHE[key] = _as_tensors(*world_pool(spec, n, world, seed))
    return _POOL_CACHE[key]


class ObjectiveError(RuntimeError):
    """Raised when a submitted objective violates the calling contract."""


def train_model(spec: SettingSpec, objective, seed: int,
                cfg: TrainConfig = TRAIN, trace=None):
    """Train one model.  ``trace(step, model)`` is an optional author-side probe."""
    envs = training_environments(spec, seed)
    tensors = [_as_tensors(x, y) for x, y in envs]
    model = build_model(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(int(seed) + 7717)
    state: dict = {}

    for step in range(cfg.steps):
        logits_by_env, targets_by_env = [], []
        for x, y in tensors:
            idx = torch.randint(0, x.shape[0], (cfg.batch_per_env,), generator=gen)
            logits_by_env.append(model(x[idx]))
            targets_by_env.append(y[idx])

        loss = objective(logits_by_env, targets_by_env, step, cfg.steps, state)

        if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
            raise ObjectiveError("objective must return a 0-dim torch tensor")
        if not torch.isfinite(loss):
            raise ObjectiveError(f"objective returned {loss.item()} at step {step}")
        if not loss.requires_grad:
            raise ObjectiveError("objective returned a tensor detached from the graph")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if trace is not None:
            trace(step, model)

    return model


@torch.no_grad()
def balanced_accuracy(model, x: torch.Tensor, y: torch.Tensor,
                      chunk: int = 2048) -> float:
    model.eval()
    preds = []
    for start in range(0, x.shape[0], chunk):
        preds.append((model(x[start:start + chunk]) > 0).float())
    model.train()
    pred = torch.cat(preds)
    out = []
    for cls in (0.0, 1.0):
        mask = y == cls
        if mask.any():
            out.append((pred[mask] == cls).float().mean().item())
    return float(np.mean(out))


def evaluate(model, spec: SettingSpec, worlds, pool_seed: int,
             source_n: int, world_n: int) -> dict:
    """Return source retention ``I`` and per-world balanced accuracies ``T_j``.

    ``pool_seed`` is independent of the training run seed: the official pools are
    drawn with a grader-private seed and the public proxy pools with a public one,
    so holding a setting's generator is not the same as holding its official pools.
    """
    xs, ys = get_source_pool(spec, source_n, pool_seed)
    retention = balanced_accuracy(model, xs, ys)
    per_world = {}
    for world in worlds:
        xw, yw = get_world_pool(spec, world_n, world, pool_seed)
        per_world[world] = balanced_accuracy(model, xw, yw)
    return {"I": retention, "T": per_world}


def quality(result: dict) -> tuple:
    """``Q = sqrt(I * min_j T_j)``; also returns the world that binds the minimum."""
    binding = min(result["T"], key=lambda w: result["T"][w])
    q = math.sqrt(max(result["I"], 0.0) * max(result["T"][binding], 0.0))
    return q, binding


def combine(qs) -> float:
    """``S = 100 * geometric_mean(Q_s)`` over the settings."""
    qs = list(qs)
    if any(q <= 0.0 for q in qs):
        return 0.0
    return 100.0 * math.exp(sum(math.log(q) for q in qs) / len(qs))


def run_setting(spec: SettingSpec, objective, seed: int, worlds,
                source_n: int, world_n: int, cfg: TrainConfig = TRAIN,
                pool_seed: int | None = None) -> dict:
    """Train on ``spec`` with run seed ``seed`` and score against ``worlds``."""
    model = train_model(spec, objective, seed, cfg)
    result = evaluate(model, spec, worlds,
                      seed if pool_seed is None else pool_seed,
                      source_n, world_n)
    q, binding = quality(result)
    result.update({"Q": q, "binding": binding, "seed": seed, "setting": spec.name})
    return result
