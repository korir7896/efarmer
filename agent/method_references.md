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

Spectral Decoupling is included because the failure it targets -- a model latching
onto a dominant easy feature and starving the harder one -- is exactly the dynamic
this generator builds.  It is meant to be competitive, not a straw man.

Two grids are shifted down from their papers' headline values, and the reason is
recorded in `crossphase/core/methods.py`: IRM's published `lambda ~ 1e4` is paired
with a `loss / lambda` rescaling that is a no-op under this task's
`AdamW(weight_decay=0)`, and EQRM's published `alpha ~ 1 - e^-100` gives a
coefficient of ~14 on a risk spread computed from four to six environments, which
is degenerate here.  Both ranges bracket the usable regime and keep a
published-scale endpoint.

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
