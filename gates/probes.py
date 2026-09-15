"""Author-side probe objectives.

None of this ships to the agent.  The probes exist to answer three questions
before any blind trial runs:

1. Can a configuration selected purely on what the agent can see already beat
   ``S*``?  (the loss-scale, combination and fingerprint probes)
2. Is the route that the task intends actually reachable?  (``ADAPTIVE``)
3. Are the escape hatches closed?  (``scaled``)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from crossphase.core.methods import (BCE, env_risks, make_eqrm, make_groupdro,
                                     make_sd, make_vrex, warm)


def scaled(objective, factor: float):
    """Multiply an objective by a constant.

    Under AdamW with ``weight_decay=0`` this must not change anything: Adam's
    update is scale-invariant up to epsilon and there is no decoupled decay term
    for the scale to trade against.  The loss-scale gate measures whether that
    holds in practice.
    """

    def wrapped(logits_by_env, targets_by_env, step, total_steps, state):
        return factor * objective(logits_by_env, targets_by_env, step,
                                  total_steps, state)

    return wrapped


def make_vrex_sd(lam_v: float, lam_s: float, warm_frac: float = 0.2):
    """Risk-variance penalty combined with logit-magnitude decay."""

    def objective(logits_by_env, targets_by_env, step, total_steps, state):
        risks = env_risks(logits_by_env, targets_by_env)
        gate = warm(step, total_steps, warm_frac)
        logits = torch.cat(logits_by_env)
        return (risks.mean()
                + lam_v * gate * risks.var(unbiased=False)
                + lam_s * gate * 0.5 * (logits ** 2).mean())

    return objective


def make_eqrm_dro(coef: float, eta: float, warm_frac: float = 0.2):
    """Quantile risk plus exponentiated-gradient environment weighting."""

    def objective(logits_by_env, targets_by_env, step, total_steps, state):
        risks = env_risks(logits_by_env, targets_by_env)
        n_env = risks.numel()
        weights = state.get("q")
        if weights is None or weights.numel() != n_env:
            weights = torch.full((n_env,), 1.0 / n_env)
        gate = warm(step, total_steps, warm_frac)
        if gate > 0.0:
            weights = weights * torch.exp(eta * risks.detach())
            weights = weights / weights.sum()
        state["q"] = weights
        return (weights * risks).sum() + coef * gate * risks.std(unbiased=False)

    return objective


# --------------------------------------------------------------------------- #
# The intended route: a scale-free penalty with a data-driven warm-up.
# --------------------------------------------------------------------------- #

def make_adaptive(lam: float = 0.45, window: int = 25, tol: float = 0.01,
                  ramp: int = 60, floor: float = 0.1):
    """Reference solution held by the author.

    Two ideas, both reachable from the information the contract exposes:

    * **Data-driven warm-up.**  Instead of a fixed fraction of the budget, the
      penalty engages when the pooled risk stops improving -- the point at which
      the easy cue has been exhausted and the representation is ready to be
      constrained.  That point sits at a different fraction of the budget in each
      setting, which is precisely what a fixed warm-up fraction cannot track.
    * **Scale-free penalty.**  The penalty is the environment-risk spread divided
      by the current risk level, so its strength is measured relative to the
      noise floor rather than in absolute loss units.  A setting with a higher
      irreducible risk is therefore regularised less, not more.
    """

    def objective(logits_by_env, targets_by_env, step, total_steps, state):
        risks = env_risks(logits_by_env, targets_by_env)
        mean_risk = risks.mean()
        spread = risks.std(unbiased=False)

        history = state.setdefault("history", [])
        history.append(float(mean_risk.detach()))

        if "engaged_at" not in state and len(history) >= 2 * window:
            recent = sum(history[-window:]) / window
            prior = sum(history[-2 * window:-window]) / window
            if (prior - recent) / max(prior, 1e-8) < tol:
                state["engaged_at"] = step

        gate = 0.0
        if "engaged_at" in state:
            gate = min(1.0, (step - state["engaged_at"] + 1) / ramp)

        level = mean_risk.detach().clamp_min(floor)
        return mean_risk + lam * gate * spread / level

    return objective


ADAPTIVE = make_adaptive()


# --------------------------------------------------------------------------- #
# Fingerprint-and-fallback
# --------------------------------------------------------------------------- #

def make_fingerprint_branch(known, fallback, threshold: float, probe_steps: int = 50):
    """Branch on first-``probe_steps`` loss statistics.

    ``known`` is used when the early pooled risk looks like the public settings,
    ``fallback`` otherwise.  The §F policy question is whether a submission that
    does this counts as a legal win; the probe measures whether it *would* win.
    """

    def objective(logits_by_env, targets_by_env, step, total_steps, state):
        risks = env_risks(logits_by_env, targets_by_env)
        stats = state.setdefault("early", [])
        if step < probe_steps:
            stats.append(float(risks.mean().detach()))
            return risks.mean()
        if "branch" not in state:
            level = sum(stats) / len(stats)
            state["branch"] = "known" if level < threshold else "fallback"
        chosen = known if state["branch"] == "known" else fallback
        return chosen(logits_by_env, targets_by_env, step, total_steps, state)

    return objective


#: Probe catalogue used by the ceiling gates, alongside the full baseline grid.
def probe_catalogue(known_config, fallback_config, threshold):
    probes = {}
    for factor in (0.1, 0.3, 1.0, 3.0, 10.0):
        probes[f"scale(ERM,{factor:g})"] = scaled(
            lambda o, t, s, ts, st: env_risks(o, t).mean(), factor)
    for lam_v in (10.0, 30.0):
        for lam_s in (0.001, 0.01):
            probes[f"VREx+SD({lam_v:g},{lam_s:g})"] = make_vrex_sd(lam_v, lam_s)
    for coef in (2.0, 4.472):
        for eta in (0.1, 1.0):
            probes[f"EQRM+DRO({coef:g},{eta:g})"] = make_eqrm_dro(coef, eta)
    probes["fingerprint-branch"] = make_fingerprint_branch(
        known_config, fallback_config, threshold)
    return probes
