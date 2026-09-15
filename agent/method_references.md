# Methods that fit the CrossPhase contract

The objective you may edit sees per-environment **logits** and **targets**, the
step counter and a persistent `state` dict.  Nothing else.  That boundary is
deliberate and it decides which published methods are in scope.

## Reproduced in this task

All six are reproduced under the identical scaffold; the selected configuration of
each, and its official score, are in `reports/baselines.csv`.

| Family | Reference | Form |
|---|---|---|
| **ERM** | -- | mean per-environment risk |
| **IRMv1** | Arjovsky, Bottou, Gulrajani, Lopez-Paz, *Invariant Risk Minimization*, arXiv:1907.02893 | `mean(R_e) + lambda * mean_e (d/dw R_e(w*logits)|_{w=1})^2` |
| **V-REx** | Krueger et al., *Out-of-Distribution Generalization via Risk Extrapolation*, arXiv:2003.00688 | `mean(R_e) + lambda * var(R_e)` |
| **GroupDRO** | Sagawa, Koh, Hashimoto, Liang, *Distributionally Robust Neural Networks for Group Shifts*, arXiv:1911.08731 | exponentiated-gradient weights over environments |
| **SD** | Pezeshki et al., *Gradient Starvation: A Learning Proclivity in Neural Networks*, arXiv:2011.09468 | `mean(R_e) + (lambda/2) * mean(logits^2)` |
| **EQRM** | Eastwood et al., *Probable Domain Generalization via Quantile Risk Minimization*, arXiv:2207.09944 | `mean(R_e) + Phi^-1(alpha) * std(R_e)` under a Gaussian fit |

Spectral Decoupling was included because the failure it targets -- a model
latching onto a dominant easy feature and starving the harder one -- is exactly
the dynamic this generator builds, so it was expected to be competitive.  That
expectation is **falsified here**, and the result is reported rather than
quietly dropped: SD finishes last, and its score decreases monotonically as
lambda rises (37.66, 37.36, 35.91, 35.61 across its published range at warm-up
0.1).  Its best configuration is the smallest coefficient in the grid, where the
penalty is inert and the objective is ERM to within 0.017 points.  Penalising
logit magnitude does not recover the stable feature in this regime.  A reader who
knows the SD paper should expect it to do better here; it does not.

## Coefficient ranges, and one deliberate non-expansion

Each family is swept over a declared twelve-configuration grid, and every range
whose optimum landed on an edge was then expanded and the expansion recorded --
IRM and V-REx to `1e6`, GroupDRO to `1e4`, SD down to `1e-4`, and warm-up
fractions out to 0 and 0.8.  After expansion every selected configuration sits in
the interior, so no family's score is limited by the edge of its grid.

**EQRM is the exception, on purpose.** Its optimum sits at `coef = 44.721`, the
top of its range, and that range was not expanded upward.  `44.721` is
`sqrt(-2*ln(1-alpha))` at `alpha = 1 - e^-1000`, the largest quantile in the
published EQRM sweep; going beyond it would no longer be reproducing the
published method.  The trade-off is stated rather than hidden: if EQRM is still
improving past that point, its reproduced score is a lower bound on what the
method could do with a coefficient its authors did not use.

## Excluded by the contract, not by oversight

These are strong domain-generalisation methods that simply cannot be expressed
from logits alone:

* **Fishr** (Rame et al., arXiv:2109.02934) -- needs the variance of per-environment
  gradients with respect to features or parameters.
* **IGA** (Koyama & Yamaguchi, arXiv:2008.01883) -- needs per-environment parameter
  gradients.
* **CORAL** (Sun & Saenko, arXiv:1607.01719) and **MMD**-based alignment -- need
  intermediate representations.
* **IB-IRM** (Ahuja et al., arXiv:2106.06607) -- needs a representation entropy term.

Reaching around the contract to obtain any of this -- through globals, module
introspection, or the autograd graph behind the logits -- is out of contract and
invalidates the submission.
