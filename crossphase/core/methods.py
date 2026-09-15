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
            # Faithful to the reference implementation.  Under AdamW with
            # weight_decay=0 this rescaling cannot change the trajectory.
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


WARM_FRACS = (0.0, 0.2, 0.5)

#: The public selection grid.  Every non-ERM family gets the same
#: twelve-configuration budget (four coefficients x three warm-up fractions).
#:
#: IRM and EQRM are swept over ranges shifted down from the headline values in
#: their papers.  IRM's published lambda ~ 1e4 is paired with the ``loss / lambda``
#: rescaling, which under this task's AdamW(weight_decay=0) optimiser is a no-op,
#: so the effective penalty strength is lambda itself; and EQRM's published
#: alpha ~ 1 - e^-100 gives a coefficient of ~14 on a risk spread computed from
#: four to six environments, which is degenerate here.  Both ranges bracket the
#: usable regime and include a published-scale endpoint.
GRIDS = {
    "ERM": {"": lambda: erm},
    "IRM": {f"lam={l:g},warm={w:g}": (lambda l=l, w=w: make_irm(l, w))
            for l in (0.1, 1.0, 10.0, 100.0) for w in WARM_FRACS},
    "VREx": {f"lam={l:g},warm={w:g}": (lambda l=l, w=w: make_vrex(l, w))
             for l in (1.0, 10.0, 30.0, 100.0) for w in WARM_FRACS},
    "GroupDRO": {f"eta={e:g},warm={w:g}": (lambda e=e, w=w: make_groupdro(e, w))
                 for e in (1e-3, 1e-2, 1e-1, 1.0) for w in WARM_FRACS},
    "SD": {f"lam={l:g},warm={w:g}": (lambda l=l, w=w: make_sd(l, w))
           for l in (0.001, 0.01, 0.1, 1.0) for w in WARM_FRACS},
    "EQRM": {f"coef={c:g},warm={w:g}": (lambda c=c, w=w: make_eqrm(c, w))
             for c in (1.414, 2.0, 4.472, 14.142) for w in WARM_FRACS},
}


def iter_configs():
    for family, grid in GRIDS.items():
        for label, factory in grid.items():
            yield family, label, factory
