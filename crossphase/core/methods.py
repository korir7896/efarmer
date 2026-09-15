"""Reference objectives.

Every method here fits the frozen contract

    objective(logits_by_env, targets_by_env, step, total_steps, state) -> scalar

which sees per-environment logits and targets and nothing else: no features, no
representations, no per-environment parameter gradients.  That restriction is
what makes the method set closed and comparable -- see ``agent/method_references.md``
for the methods it excludes and why.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

BCE = F.binary_cross_entropy_with_logits


def env_risks(logits_by_env, targets_by_env):
    return torch.stack([BCE(o, t) for o, t in zip(logits_by_env, targets_by_env)])


def warm(step: int, total_steps: int, frac: float) -> float:
    """Published penalty-annealing convention: off, then on at a fixed fraction."""
    return 0.0 if step < int(frac * total_steps) else 1.0


def erm(logits_by_env, targets_by_env, step, total_steps, state):
    """Empirical risk minimisation."""
    return env_risks(logits_by_env, targets_by_env).mean()


def make_irm(lam: float, warm_frac: float):
    """IRMv1 -- Arjovsky et al., *Invariant Risk Minimization* (arXiv:1907.02893).

    Penalty is the squared gradient of the per-environment risk with respect to a
    constant scalar multiplier on the logits, so it is computable from logits alone.
    """

    def objective(logits_by_env, targets_by_env, step, total_steps, state):
        risks, penalties = [], []
        for o, t in zip(logits_by_env, targets_by_env):
            scale = torch.ones(1, device=o.device, requires_grad=True)
            risk = BCE(o * scale, t)
            grad = torch.autograd.grad(risk, [scale], create_graph=True)[0]
            risks.append(risk)
            penalties.append((grad ** 2).sum())
        mean_risk = torch.stack(risks).mean()
        penalty = torch.stack(penalties).mean()
        weight = lam * warm(step, total_steps, warm_frac) + 1.0
        loss = mean_risk + (weight - 1.0) * penalty
        if weight > 1.0:
            # Faithful to the reference implementation -- and load-bearing, which
            # is worth stating because the obvious assumption is wrong.  This is
            # NOT a harmless rescaling: the loss is returned undivided before
            # warm-up and divided by (1 + lambda) after it, so at lambda = 1e5 its
            # magnitude drops five orders of magnitude at the warm-up boundary.
            # Adam's second-moment estimate is adapted to the old scale and
            # re-adapts over ~1/(1 - beta2) = 1000 steps, longer than the run has
            # left, so effective step sizes stay small afterwards.  Measured on
            # this task: 57.71 with the division, 41.55 without.  A UNIFORM
            # rescaling really is neutral here (ERM scores 36.66 / 36.64 / 36.74
            # at x0.1 / x1 / x10); it is the mid-run discontinuity that bites.
            loss = loss / weight
        return loss

    return objective


def make_vrex(lam: float, warm_frac: float):
    """V-REx -- Krueger et al., *Out-of-Distribution Generalization via Risk
    Extrapolation* (arXiv:2003.00688).  Penalises the variance of the
    per-environment risks."""

    def objective(logits_by_env, targets_by_env, step, total_steps, state):
        risks = env_risks(logits_by_env, targets_by_env)
        penalty = risks.var(unbiased=False)
        return risks.mean() + lam * warm(step, total_steps, warm_frac) * penalty

    return objective


def make_groupdro(eta: float, warm_frac: float):
    """GroupDRO -- Sagawa et al., *Distributionally Robust Neural Networks*
    (arXiv:1911.08731).  Exponentiated-gradient weights over environments.

    The weight vector is sized from the environments actually supplied, so the
    method is agnostic to the per-run environment count.
    """

    def objective(logits_by_env, targets_by_env, step, total_steps, state):
        risks = env_risks(logits_by_env, targets_by_env)
        n_env = risks.numel()
        weights = state.get("dro_q")
        if weights is None or weights.numel() != n_env:
            weights = torch.full((n_env,), 1.0 / n_env)
        if warm(step, total_steps, warm_frac) > 0.0:
            weights = weights * torch.exp(eta * risks.detach())
            weights = weights / weights.sum()
        else:
            weights = torch.full((n_env,), 1.0 / n_env)
        state["dro_q"] = weights
        return (weights * risks).sum()

    return objective


def make_sd(lam: float, warm_frac: float):
    """Spectral Decoupling -- Pezeshki et al., *Gradient Starvation: A Learning
    Proclivity in Neural Networks* (arXiv:2011.09468).  ``L = R + (lam/2)*mean(o^2)``."""

    def objective(logits_by_env, targets_by_env, step, total_steps, state):
        risks = env_risks(logits_by_env, targets_by_env)
        logits = torch.cat(logits_by_env)
        penalty = 0.5 * (logits ** 2).mean()
        return risks.mean() + lam * warm(step, total_steps, warm_frac) * penalty

    return objective


def make_eqrm(coef: float, warm_frac: float):
    """EQRM -- Eastwood et al., *Probable Domain Generalization via Quantile Risk
    Minimization* (arXiv:2207.09944).  Under a Gaussian fit to the environment-risk
    distribution the alpha-quantile is ``mean(R) + Phi^-1(alpha)*std(R)``, with the
    published asymptotic form ``Phi^-1(alpha) ~ sqrt(-2*ln(1-alpha))``.  The
    warm-up fraction plays the role of the published anneal-from-ERM phase."""

    def objective(logits_by_env, targets_by_env, step, total_steps, state):
        risks = env_risks(logits_by_env, targets_by_env)
        spread = risks.std(unbiased=False)
        return risks.mean() + coef * warm(step, total_steps, warm_frac) * spread

    return objective


WARM_FRACS = (0.1, 0.3, 0.5)

#: The public selection grid.  Every non-ERM family gets the same
#: twelve-configuration budget (four coefficients x three warm-up fractions).
#:
#: The grids bracket each paper's published range.  IRM's headline lambda ~ 1e4
#: and EQRM's alpha ~ 1 - e^-100 (coefficient 14.142) both sit inside their grids
#: here, and both are live rather than degenerate: in this regime the shortcut is
#: the better within-environment predictor, so recovering the stable feature takes
#: penalties of published strength.  Warm-up fractions are 0.1, 0.3 and 0.5 -- a
#: penalty of this strength from step zero destroys the run in every family, so
#: zero warm-up is not a usable configuration and is not offered.
GRIDS = {
    "ERM": {"": lambda: erm},
    "IRM": {f"lam={l:g},warm={w:g}": (lambda l=l, w=w: make_irm(l, w))
            for l in (1e2, 1e3, 1e4, 1e5) for w in WARM_FRACS},
    "VREx": {f"lam={l:g},warm={w:g}": (lambda l=l, w=w: make_vrex(l, w))
             for l in (1e2, 1e3, 1e4, 1e5) for w in WARM_FRACS},
    "GroupDRO": {f"eta={e:g},warm={w:g}": (lambda e=e, w=w: make_groupdro(e, w))
                 for e in (0.1, 1.0, 10.0, 100.0) for w in WARM_FRACS},
    "SD": {f"lam={l:g},warm={w:g}": (lambda l=l, w=w: make_sd(l, w))
           for l in (0.001, 0.01, 0.1, 1.0) for w in WARM_FRACS},
    "EQRM": {f"coef={c:g},warm={w:g}": (lambda c=c, w=w: make_eqrm(c, w))
             for c in (4.472, 14.142, 31.623, 44.721) for w in WARM_FRACS},
}


#: Boundary expansions, recorded rather than folded into the declared grid.
#:
#: Every family's selected configuration in the first sweep sat on an edge of its
#: declared range -- six for six -- and an agent is not bound by that range, so an
#: unexplored edge means ``S*`` may be understated.  The protocol is to expand only
#: where an optimum sits on a boundary and to record the expansion, which is what
#: this table is.
#:
#: Two edges are expanded in every family regardless: warm-up 0 (the ordinary
#: no-warm-up form of each method, which the declared grid omitted) and warm-up
#: 0.65 / 0.8.  The latter matters because the measured shortcut-to-stable
#: transition in setting A sits at step 520 of 800 -- fraction 0.65 -- i.e. past
#: the top of the declared warm-up range, which is precisely where this task's own
#: mechanism says the optimum should be.
#:
#: EQRM's coefficient is deliberately NOT expanded upward: 44.721 is
#: sqrt(-2*ln(1-alpha)) at alpha = 1 - e^-1000, the top of the published sweep.
#: Going beyond it would leave the reproduced method behind.  See
#: ``agent/method_references.md``.
EXPANSIONS = {
    "IRM": {f"lam={l:g},warm={w:g}": (lambda l=l, w=w: make_irm(l, w))
            for l, w in [(1e6, 0.3), (1e6, 0.5), (1e6, 0.65),
                         (1e5, 0.0), (1e5, 0.65), (1e5, 0.8), (1e4, 0.65)]},
    "VREx": {f"lam={l:g},warm={w:g}": (lambda l=l, w=w: make_vrex(l, w))
             for l, w in [(1e6, 0.1), (1e6, 0.5), (1e6, 0.65),
                          (1e5, 0.0), (1e5, 0.65), (1e5, 0.8)]},
    "GroupDRO": {f"eta={e:g},warm={w:g}": (lambda e=e, w=w: make_groupdro(e, w))
                 for e, w in [(1e3, 0.5), (1e4, 0.5), (1e3, 0.65),
                              (100.0, 0.0), (100.0, 0.65), (100.0, 0.8)]},
    "SD": {f"lam={l:g},warm={w:g}": (lambda l=l, w=w: make_sd(l, w))
           for l, w in [(1e-4, 0.3), (1e-4, 0.5),
                        (0.001, 0.0), (0.001, 0.65), (0.001, 0.8)]},
    "EQRM": {f"coef={c:g},warm={w:g}": (lambda c=c, w=w: make_eqrm(c, w))
             for c, w in [(44.721, 0.0), (44.721, 0.65), (44.721, 0.8),
                          (31.623, 0.0), (31.623, 0.65)]},
}


def all_configs() -> dict:
    """Declared grid plus recorded expansions, as ``{family: {label: factory}}``."""
    merged = {f: dict(g) for f, g in GRIDS.items()}
    for family, extra in EXPANSIONS.items():
        merged[family].update(extra)
    return merged


def is_expansion(family: str, label: str) -> bool:
    return label in EXPANSIONS.get(family, {})


def iter_configs():
    for family, grid in all_configs().items():
        for label, factory in grid.items():
            yield family, label, factory
