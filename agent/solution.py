"""CrossPhase submission file.

Everything outside the SOLUTION block is frozen.  Edit only the region between
the two markers; the data, the model, the optimiser, the schedule and the
evaluation are fixed and are not yours to change.

CONTRACT
--------
``objective(logits_by_env, targets_by_env, step, total_steps, state) -> scalar``

    logits_by_env   list of 1-D float tensors, one per training environment,
                    each of shape (batch,), attached to the autograd graph.
                    The number of environments varies between runs and the
                    order is permuted per run: do not assume a fixed count and
                    do not index positionally.
    targets_by_env  list of 1-D float tensors of 0/1 labels, same shapes.
    step            current optimisation step, 0-indexed.
    total_steps     total steps in this run.
    state           a plain dict that persists across the steps of one run.
                    It starts empty.  Use it for anything you need to carry.

    returns         a 0-dim torch tensor attached to the graph.  A non-finite
                    value, a detached tensor or a non-scalar aborts the run and
                    scores zero for that setting.

RULES
-----
* Logits and targets only.  You have no access to inputs, features, parameters
  or per-parameter gradients, and reaching for them through globals is out of
  contract.
* You may adapt to statistics you observe during training (loss levels, risk
  spread, how they move with ``step``).  You may not branch on anything that
  identifies which setting you are in from outside the training signal.
* No file, network or environment-variable access; no wall-clock branching.
* The objective is called once per step and must be deterministic given its
  arguments.

THE SHIPPED OBJECTIVE
--------------------
EQRM (Eastwood et al., arXiv:2207.09944) at the configuration the public proxy
diagnostic selects for it: ``coef = sqrt(-2*ln(1-alpha)) = 14.142``, warm-up at
half the budget.  Of the six reproduced families this is the weakest, which is
why it is what you start from.  Its official score and every other family's are
in the README.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# BEGIN SOLUTION
# --------------------------------------------------------------------------- #


COEF = 14.142      # Phi^-1(alpha) for alpha = 1 - e^-100, the published scale
WARM_FRACTION = 0.5


def objective(logits_by_env, targets_by_env, step, total_steps, state):
    """EQRM: the alpha-quantile of the environment-risk distribution.

    Under a Gaussian fit to the per-environment risks the alpha-quantile is
    ``mean(R) + Phi^-1(alpha) * std(R)``.  The penalty is held off for the first
    ``WARM_FRACTION`` of the budget, which is the published anneal-from-ERM phase.
    """
    risks = torch.stack([
        F.binary_cross_entropy_with_logits(o, t)
        for o, t in zip(logits_by_env, targets_by_env)
    ])
    engaged = 0.0 if step < int(WARM_FRACTION * total_steps) else 1.0
    return risks.mean() + COEF * engaged * risks.std(unbiased=False)


# --------------------------------------------------------------------------- #
# END SOLUTION
# --------------------------------------------------------------------------- #
