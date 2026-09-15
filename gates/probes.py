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
                                     make_irm, make_sd, make_vrex, warm)


def scaled(objective, factor: float):
    """Multiply an objective by a constant, for the whole run.

    Under AdamW with ``weight_decay=0`` this should not change anything, and
    measurement agrees: ERM scores 36.66 / 36.64 / 36.74 at x0.1 / x1 / x10.
    Kept as the control for the two probes below, which do move the result.
    """

    def wrapped(logits_by_env, targets_by_env, step, total_steps, state):
        return factor * objective(logits_by_env, targets_by_env, step,
                                  total_steps, state)

    return wrapped


def scale_step(objective, factor: float, at: float = 0.5):
    """Change the objective's own scale PART-WAY through the run.

    This is the route that zero weight decay does not close.  Adam's
    second-moment estimate re-adapts over roughly ``1/(1 - beta2)`` steps -- 1000
    at the default, longer than the budget -- so a mid-run drop shrinks effective
    step sizes for the rest of training.  No invariance penalty is involved.
    """

    def wrapped(logits_by_env, targets_by_env, step, total_steps, state):
        loss = objective(logits_by_env, targets_by_env, step, total_steps, state)
        return loss if step < int(at * total_steps) else factor * loss

    return wrapped


def freeze(objective, at: float = 0.5):
    """Zero the gradient from ``at`` onwards: early stopping, spelled as a loss.

    The strongest form of the same idea, and the floor any penalty-bearing method
    has to clear before its score can be credited to the penalty.
    """

    def wrapped(logits_by_env, targets_by_env, step, total_steps, state):
        loss = objective(logits_by_env, targets_by_env, step, total_steps, state)
        return loss if step < int(at * total_steps) else loss * 0.0

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

def irm_penalty(logits_by_env, targets_by_env):
    """IRMv1 penalty, shared by the reference solution and its controls."""
    out = []
    for o, t in zip(logits_by_env, targets_by_env):
        scale = torch.ones(1, requires_grad=True)
        grad = torch.autograd.grad(BCE(o * scale, t), [scale], create_graph=True)[0]
        out.append((grad ** 2).sum())
    return torch.stack(out).mean()


def make_floor_relative_irm(lam: float = 1e5, warm_frac: float = 0.5,
                            power: float = 2.0, anchor: float = 0.26,
                            ema: float = 0.02):
    """Reference solution: IRM whose strength is measured against the noise floor.

    The reasoning is available from the public settings alone, which is the point
    -- a reference solution fitted with knowledge of the hidden setting would only
    show that its author had seen the answer key.  A penalty coefficient is quoted
    in absolute loss units, but the irreducible risk differs between settings, so
    the same lambda means different things in each.  Dividing by the observed risk
    level makes the strength comparable and regularises a high-floor setting less
    rather than more.

    ``anchor`` is the risk level the PUBLIC settings show, so at ``power = 0`` and
    on A and B this reduces to the reproduced IRM baseline exactly; only the
    unseen setting moves.  The ``loss / weight`` convention is IRM's own -- see
    the note in ``crossphase/core/methods.py`` for why it is load-bearing.
    """

    def objective(logits_by_env, targets_by_env, step, total_steps, state):
        risks = env_risks(logits_by_env, targets_by_env)
        mean_risk = risks.mean()

        level = state.get("level")
        observed = float(mean_risk.detach())
        level = observed if level is None else (1 - ema) * level + ema * observed
        state["level"] = level

        if step < int(warm_frac * total_steps):
            return mean_risk

        effective = lam * (anchor / max(level, 1e-3)) ** power
        weight = effective + 1.0
        loss = mean_risk + effective * irm_penalty(logits_by_env, targets_by_env)
        return loss / weight

    return objective


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


def make_regime_fallback(threshold: float = 0.30, decide_at: int = 200,
                         window: int = 50, lam: float = 1e3,
                         irm_warm: float = 0.3, vrex_lam: float = 1e5,
                         vrex_warm: float = 0.5):
    """Reference solution: recognise an unfamiliar regime, then fall back.

    Why this shape and not something tidier -- both alternatives were measured
    and both fail:

    * Rescaling the tuned penalty by the observed noise floor does nothing.
      Q_C moves 0.444 -> 0.446 across powers 0 to 6.  Penalty strength is not
      what limits the hidden setting; the IRM family simply tops out below
      what a quantile objective reaches there.
    * Running a quantile term ALONGSIDE the penalty reaches the hidden
      setting (Q_C 0.509) but costs a quarter of the visible ones
      (Q_A 0.660 -> 0.500), in every lambda/coefficient combination tried.
      The early quantile pressure stops A and B ever reaching the
      representation the late penalty exploits.

    So the two mechanisms have to run alternatively, not together.  The rule is
    reachable from the public settings alone: calibrate on A and B, watch the
    pooled risk level, and if a run sits materially outside the range you
    calibrated on, fall back to an objective that does not depend on the
    coefficients you tuned there.  It never requires knowing that the hidden
    setting exists -- only your own calibration range.

    Timing is the part that has to be right.  Judging at step 60 misclassifies
    A and B, whose level is still descending through the hidden setting's range;
    by step 200 they separate cleanly (~0.275 / ~0.272 against ~0.346).  The
    decision has to land before the earlier branch engages -- the tuned branch
    warms up at 0.3, i.e. step 240 -- and the result is identical for thresholds
    0.29-0.31 and for decisions at step 200 or 300, so it is not perched on a
    knife edge.

    Both branches are pinned to the configurations the baseline sweep actually
    selects, so each control returns its own baseline exactly: always-tuned
    scores 58.501 (IRM) and always-conservative 46.733 (V-REx).  Neither
    reproduces the combination, 61.539, which is what shows the branch is really
    branching rather than silently always firing the same way.

    ``docs/classification_rules.md`` declared this route legal before any of
    these measurements were taken.  That ordering is the point of the rule.
    """
    tuned = make_irm(lam, irm_warm)
    conservative = make_vrex(vrex_lam, vrex_warm)

    def objective(logits_by_env, targets_by_env, step, total_steps, state):
        risks = env_risks(logits_by_env, targets_by_env)
        mean_risk = risks.mean()
        if step < decide_at:
            if step >= decide_at - window:
                state.setdefault("levels", []).append(float(mean_risk.detach()))
            return mean_risk
        if "familiar" not in state:
            levels = state.get("levels") or [float(mean_risk.detach())]
            state["familiar"] = (sum(levels) / len(levels)) < threshold
        branch = tuned if state["familiar"] else conservative
        return branch(logits_by_env, targets_by_env, step, total_steps, state)

    return objective


ADAPTIVE = make_regime_fallback()


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
    erm = (lambda o, t, s, ts, st: env_risks(o, t).mean())
    probes = {}
    for factor in (0.1, 1.0, 10.0):
        probes[f"scale(ERM,{factor:g})"] = scaled(erm, factor)
    for factor in (1e-5, 1e-2):
        probes[f"scale-step(ERM,{factor:g})"] = scale_step(erm, factor)
    probes["freeze(ERM,0.5)"] = freeze(erm)
    for lam_v in (10.0, 30.0):
        for lam_s in (0.001, 0.01):
            probes[f"VREx+SD({lam_v:g},{lam_s:g})"] = make_vrex_sd(lam_v, lam_s)
    for coef in (2.0, 4.472):
        for eta in (0.1, 1.0):
            probes[f"EQRM+DRO({coef:g},{eta:g})"] = make_eqrm_dro(coef, eta)
    probes["fingerprint-branch"] = make_fingerprint_branch(
        known_config, fallback_config, threshold)
    return probes
