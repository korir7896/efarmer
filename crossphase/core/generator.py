"""CrossPhase signal generator.

The stable (causal) feature is the SIGN OF THE CROSS-CHANNEL PHASE LAG of a core
oscillation.  It is invariant to the global initial phase, to per-example gain,
and to any additive signal that is common to both channels -- but it requires the
network to compare the two channels, which makes it strictly harder to acquire
than either nuisance cue.

Crucially the stable feature is also an IMPERFECT predictor.  A latent class ``c``
sets the phase lag; the observed label ``y`` is ``c`` flipped with probability
``label_noise``.  The nuisance cues track ``y``, not ``c``, and they do so at
per-environment rates that exceed ``1 - label_noise``.  Within any training
environment the shortcut is therefore the *better* predictor of the label, and
plain risk minimisation prefers it on the merits rather than merely by
convenience.  Without this, the core feature is deterministic, empirical risk
minimisation learns it directly, and no invariance penalty has anything to
repair -- measured, and the reason this term exists.

Two nuisance cues are correlated with the label during training:

  z1 -- the "level" cue.  Additive settings add a DC offset ``kappa1 * z1`` to both
        channels; the multiplicative setting instead scales both channels by
        ``1 + gamma * z1``.
  z2 -- the "band" cue.  Additive settings add a fixed-amplitude carrier whose
        frequency is chosen by ``z2``; the multiplicative setting instead adds a
        monotone positive drift envelope whose slope sign is ``z2``.

An environment is a triple ``(q1, q2, sigma)``: the probability that each cue
agrees with the label, and the observation-noise scale.

A *world* is an evaluation distribution, written as a 4-tuple
``(p1, p2, sigma_mult, core_mult)``: the two cue-agreement probabilities, a
multiplier on the observation noise, and a multiplier on the core amplitude.  The
last two exist so that the worst-case minimum in the score is contested.  Worlds
that flip the cues punish shortcut reliance; worlds that attenuate the core or
raise the noise floor punish a model whose core detector is weak, which is how an
over-regularised objective fails.  Without the second kind, the cue-flipped world
would bind the minimum in essentially every cell and the other worlds would be
decorative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class Environment:
    q1: float          # P(z1 agrees with label)
    q2: float          # P(z2 agrees with label)
    sigma: float       # observation noise scale


@dataclass(frozen=True)
class SettingSpec:
    """Everything that defines one CrossPhase setting."""

    name: str
    length: int                   # samples per example
    f_core: float                 # core oscillation frequency (cycles per window)
    a_core: float                 # core amplitude
    delta: float                  # half cross-channel phase lag, radians
    mode: str                     # "additive" | "multiplicative"
    label_noise: float = 0.0      # P(observed label differs from the latent class)
    kappa1: float = 0.0           # additive DC offset magnitude
    kappa2: float = 0.0           # additive carrier amplitude
    f_band: tuple = (12.0, 18.0)  # carrier frequencies selected by z2
    gamma: float = 0.0            # multiplicative gain depth
    beta: float = 0.0             # drift envelope slope
    envs: tuple = ()              # up to six Environment entries
    train_per_env: int = 1500
    source_sigma: float = 0.45    # noise for source-distribution pools
    source_mix: tuple = ()        # (q1, q2) mixture weights for source pools

    def env_count_for_seed(self, seed: int) -> int:
        """Environment count is drawn from {4, 5, 6} identically in every setting.

        Because the distribution does not depend on the setting, the count carries
        no information about which setting is running, but an objective still may
        not assume ``E == 4`` or index environments positionally.
        """
        return 4 + (np.random.default_rng(90000 + seed).integers(0, 3).item())


def _rng(*parts: int) -> np.random.Generator:
    """Deterministic generator from a tuple of integer coordinates."""
    return np.random.default_rng(np.random.SeedSequence(list(parts)))


def sample(spec: SettingSpec, n: int, p1: float, p2: float, sigma: float,
           seed_parts: tuple, core_mult: float = 1.0) -> tuple:
    """Draw ``n`` examples with cue-agreement probabilities ``(p1, p2)``.

    Returns ``(x, y, z1, z2)`` with ``x`` shaped ``(n, 2, length)``.
    """
    rng = _rng(*seed_parts)
    L = spec.length
    t = np.arange(L, dtype=np.float64) / L

    # Latent class -> phase lag; observed label -> the cues.  The cues agree with
    # the observed label more often than the phase lag does, which is what makes
    # the shortcut worth taking.
    latent = rng.integers(0, 2, size=n)
    flip = rng.random(n) < spec.label_noise
    y = np.where(flip, 1 - latent, latent)
    s = 2.0 * y - 1.0
    z1 = s * np.where(rng.random(n) < p1, 1.0, -1.0)
    z2 = s * np.where(rng.random(n) < p2, 1.0, -1.0)

    phi = rng.uniform(0.0, TWO_PI, size=n)[:, None]
    lag = spec.delta * (2.0 * latent - 1.0)[:, None]
    base = TWO_PI * spec.f_core * t[None, :]
    amplitude = spec.a_core * core_mult
    core0 = amplitude * np.sin(base + phi)
    core1 = amplitude * np.sin(base + phi + lag)

    x = np.empty((n, 2, L), dtype=np.float64)
    if spec.mode == "additive":
        f_lo, f_hi = spec.f_band
        f_sel = np.where(z2 > 0, f_hi, f_lo)[:, None]
        psi = rng.uniform(0.0, TWO_PI, size=n)[:, None]
        carrier = spec.kappa2 * np.sin(TWO_PI * f_sel * t[None, :] + psi)
        common = spec.kappa1 * z1[:, None] + carrier
        x[:, 0, :] = core0 + common
        x[:, 1, :] = core1 + common
    elif spec.mode == "multiplicative":
        gain = (1.0 + spec.gamma * z1)[:, None]
        drift = spec.beta * z2[:, None] * t[None, :]
        x[:, 0, :] = gain * core0 + drift
        x[:, 1, :] = gain * core1 + drift
    else:  # pragma: no cover - guarded by the spec registry
        raise ValueError(f"unknown mode {spec.mode!r}")

    x += sigma * rng.standard_normal(size=(n, 2, L))
    return x.astype(np.float32), y.astype(np.int64), z1, z2


def training_environments(spec: SettingSpec, seed: int) -> list:
    """Build the per-environment training arrays for one run seed."""
    count = spec.env_count_for_seed(seed)
    out = []
    for idx in range(count):
        env = spec.envs[idx]
        x, y, _, _ = sample(
            spec, spec.train_per_env, env.q1, env.q2, env.sigma,
            (1, seed, idx, int(1000 * env.q1), int(1000 * env.sigma)),
        )
        out.append((x, y))
    # Environment order is permuted per run so that positional indexing carries
    # no stable meaning across runs.
    order = _rng(2, seed).permutation(count)
    return [out[i] for i in order]


def source_pool(spec: SettingSpec, n: int, seed: int) -> tuple:
    """Held-out pool from the training environment mixture (scores ``I``)."""
    q1, q2 = spec.source_mix
    return sample(spec, n, q1, q2, spec.source_sigma, (3, seed, int(1e4 * q1)))[:2]


def world_pool(spec: SettingSpec, n: int, world: tuple, seed: int) -> tuple:
    """Pool for one target world ``(p1, p2, sigma_mult, core_mult)``."""
    p1, p2, sigma_mult, core_mult = world
    return sample(spec, n, p1, p2, spec.source_sigma * sigma_mult,
                  (4, seed, int(1e4 * p1), int(1e4 * p2),
                   int(1e3 * sigma_mult), int(1e3 * core_mult)),
                  core_mult=core_mult)[:2]
