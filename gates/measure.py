"""Author-side measurements used by the gate suite."""

from __future__ import annotations

import numpy as np
import torch

from crossphase.core.engine import (TRAIN, balanced_accuracy, get_world_pool,
                                    train_model)
from crossphase.core.generator import sample, training_environments

TWO_PI = 2.0 * np.pi


def spearman(a, b) -> float:
    """Rank correlation without a scipy dependency."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


# --------------------------------------------------------------------------- #
# Transition timing
# --------------------------------------------------------------------------- #

def transition_trace(spec, objective, seed, probe_world, probe_n=600, every=10):
    """Balanced accuracy on a cue-flipped probe world, sampled during training."""
    xs, ys = get_world_pool(spec, probe_n, probe_world, 90210)
    trace = []

    def hook(step, model):
        if step % every == 0 or step == TRAIN.steps - 1:
            trace.append((step, balanced_accuracy(model, xs, ys)))

    train_model(spec, objective, seed, trace=hook)
    return trace


def transition_step(trace, window=3) -> int:
    """The step of steepest sustained rise in cue-flipped accuracy.

    This is where the model stops predicting from the nuisance cue and starts
    predicting from the stable cross-channel phase relationship.
    """
    steps = np.array([s for s, _ in trace], dtype=float)
    acc = np.array([a for _, a in trace], dtype=float)
    if len(acc) <= 2 * window:
        return int(steps[-1])
    kernel = np.ones(window) / window
    smooth = np.convolve(acc, kernel, mode="valid")
    diff = np.diff(smooth)
    idx = int(np.argmax(diff))
    return int(steps[idx + window - 1])


# --------------------------------------------------------------------------- #
# Cue recovery from public training data (the reconstruction route)
# --------------------------------------------------------------------------- #

def recover_cues(spec, x: np.ndarray) -> tuple:
    """Estimate the two nuisance signs from raw signals, as an agent would.

    * ``z1`` -- additive settings add a DC offset to both channels, so the
      per-example temporal mean reads it off almost directly (the core has zero
      mean under a uniform initial phase).  In the multiplicative setting the
      same cue lives in the overall amplitude, so the per-example standard
      deviation stands in for it.
    * ``z2`` -- additive settings select between two carrier frequencies, which
      one Fourier transform separates.  In the multiplicative setting the cue is
      a monotone drift, so the sign of a linear trend stands in for it.
    """
    pooled = x.mean(axis=1)                       # average the two channels
    L = pooled.shape[1]

    if spec.mode == "additive":
        z1_stat = pooled.mean(axis=1)
        spectrum = np.abs(np.fft.rfft(pooled - pooled.mean(axis=1, keepdims=True), axis=1))
        lo, hi = spec.f_band
        z1_hat = np.sign(z1_stat - np.median(z1_stat))
        band = spectrum[:, int(round(hi))] - spectrum[:, int(round(lo))]
        z2_hat = np.sign(band - np.median(band))
    else:
        z1_stat = pooled.std(axis=1)
        z1_hat = np.sign(z1_stat - np.median(z1_stat))
        t = np.arange(L) - (L - 1) / 2.0
        slope = (pooled * t).sum(axis=1) / (t ** 2).sum()
        z2_hat = np.sign(slope - np.median(slope))

    z1_hat[z1_hat == 0] = 1.0
    z2_hat[z2_hat == 0] = 1.0
    return z1_hat, z2_hat


def cue_recovery_accuracy(spec, n=4000, seed=771) -> dict:
    """How well the estimators above recover the true cue signs."""
    x, _y, z1, z2 = sample(spec, n, 0.5, 0.5, spec.envs[0].sigma, (91, seed))
    z1_hat, z2_hat = recover_cues(spec, x.astype(np.float64))
    acc1 = max((z1_hat == z1).mean(), (z1_hat == -z1).mean())
    acc2 = max((z2_hat == z2).mean(), (z2_hat == -z2).mean())
    # A median split predicts exactly balanced signs, so these accuracies are
    # capped by the sign imbalance of the drawn cues; at this pool size the cap
    # is what both estimators actually hit.
    cap = max((z1 > 0).mean(), (z1 < 0).mean())
    return {"z1": float(acc1), "z2": float(acc2), "median_split_cap": float(2 - 2 * cap)}


def reconstructed_world(spec, world, seed, n=1200):
    """Rebuild a target world from public TRAINING data only.

    The agent holds the training arrays with environment labels.  It recovers the
    cue signs by the estimators above, then resamples those held examples so that
    each cue agrees with the label at the requested rate.  No official pool and no
    knowledge of the official pool seed is used.
    """
    p1, p2, sigma_mult, core_mult = world
    parts_x, parts_y, parts_z1, parts_z2 = [], [], [], []
    for x, y in training_environments(spec, seed):
        z1_hat, z2_hat = recover_cues(spec, x.astype(np.float64))
        parts_x.append(x)
        parts_y.append(y)
        parts_z1.append(z1_hat)
        parts_z2.append(z2_hat)

    x = np.concatenate(parts_x)
    y = np.concatenate(parts_y)
    z1 = np.concatenate(parts_z1)
    z2 = np.concatenate(parts_z2)
    rng = np.random.default_rng(3300 + seed)
    want_y = rng.integers(0, 2, n)
    want_s = 2.0 * want_y - 1.0
    want1 = want_s * np.where(rng.random(n) < p1, 1.0, -1.0)
    want2 = want_s * np.where(rng.random(n) < p2, 1.0, -1.0)

    buckets: dict = {}
    for cls in (0, 1):
        for a in (-1.0, 1.0):
            for b in (-1.0, 1.0):
                idx = np.nonzero((y == cls) & (z1 == a) & (z2 == b))[0]
                buckets[(cls, a, b)] = idx

    picks = np.empty(n, dtype=np.int64)
    for i in range(n):
        key = (int(want_y[i]), float(want1[i]), float(want2[i]))
        pool = buckets.get(key)
        if pool is None or len(pool) == 0:
            pool = np.nonzero(y == int(want_y[i]))[0]
        picks[i] = pool[rng.integers(0, len(pool))]

    xs = x[picks].astype(np.float32)
    if core_mult != 1.0 or sigma_mult != 1.0:
        # The agent can only approximate these two axes; extra noise is the
        # honest approximation of a raised noise floor.
        extra = spec.source_sigma * max(sigma_mult ** 2 - 1.0, 0.0) ** 0.5
        xs = (xs * core_mult if core_mult != 1.0 else xs)
        if extra > 0:
            xs = xs + extra * rng.standard_normal(xs.shape).astype(np.float32)
    return torch.from_numpy(xs), torch.from_numpy(y[picks].astype(np.float32))
