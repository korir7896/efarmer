# CrossPhase: design notes

This document records *why* the task is shaped the way it is.  The measured
evidence for each claim lives in `reports/`.

## 1. The hypothesis the task is built on

A network offered an easy nuisance cue and a harder stable cue fits the nuisance
first, then — if training continues and the objective permits it — transitions to
the stable relationship.  Three claims follow, and all three are gated:

1. That transition happens at a **different fraction of the budget in each
   setting**, so a fixed warm-up fraction selected on the visible settings is
   mistimed on the hidden one.
2. The **useful penalty strength depends on the noise floor**, so a coefficient
   selected where the floor is low over-regularises where it is high.
3. Therefore **no single fixed objective is optimal across all three settings**,
   and the gap between the best fixed baseline and what an adaptive objective can
   reach is the headroom the task is asking an agent to claim.

If (3) were false, `S*` would sit at the achievable ceiling and there would be
nothing to win.  It is a designed-in property, gated before any trial, not a
hoped-for one.

## 2. Generator

Signals are two-channel, length 128 (A and B) or 160 (C).

* **Stable feature.** A core oscillation at `f_core` with a uniformly random
  initial phase, present in both channels.  The label is the *sign of the
  cross-channel phase lag*, `+delta` versus `-delta`.  This is invariant to the
  global phase, to per-example gain and to any signal added equally to both
  channels, and it cannot be read from either channel alone.
* **Level cue `z1`.** Additive settings add `kappa1 * z1` to both channels — the
  per-example temporal mean reads it almost directly.  Setting C instead scales
  both channels by `1 + gamma * z1`.
* **Band cue `z2`.** Additive settings add a carrier whose frequency `z2`
  selects from a fixed pair; one Fourier transform separates them.  Setting C
  instead adds a monotone drift whose slope sign is `z2`.

Both cues are far easier to extract than the stable one.  That asymmetry is the
shortcut.

**Environments.** Each is `(q1, q2, sigma)`: the probability that each cue agrees
with the label, and the observation-noise scale.  We write them as a
contamination *level* and a cue *split* (`q1 = l + s`, `q2 = l - s`) with the
split alternating in sign, so the two cues disagree about which environment
favours them — that disagreement is the variation an invariance penalty can use.
The level moves monotonically against sigma, and the **sign of that relationship
is flipped in C**: in A and B the noisiest environment is the least contaminated,
in C it is the most.  That flip is the mechanism behind the required rank
reversal between settings.

**Environment count.** Drawn from `{4, 5, 6}` as a deterministic function of the
run seed, with the same distribution in every setting, so the count carries no
information about which setting is running while still forbidding a hardcoded
`E = 4`.  Environment order is permuted per run, so positional indexing carries
no stable meaning.

## 3. Worlds and scoring

A world is `(p1, p2, sigma_mult, core_mult)`.  The official five:

| World | Stresses |
|---|---|
| `(.10, .10, 1.0, 1.0)` | both cues flipped — shortcut reliance |
| `(.10, .90, 1.0, 1.0)` | cue conflict |
| `(.90, .10, 1.0, 1.0)` | cue conflict, the other way |
| `(.50, .50, 1.6, 1.0)` | cues uninformative, noise floor raised 60% |
| `(.90, .90, 1.0, 0.42)` | cues aligned, core attenuated to 42% |

The last two exist because of the binding-world problem: with cue-flipped worlds
alone, `(.10,.10)` binds the minimum in essentially every cell and the rest are
decorative.  The two core-stress worlds fail for the *opposite* reason — a model
that has been regularised into a weak core detector — so the minimum is
contested.  `reports/baselines.csv` records which world actually binds per cell.

```
Q_s = sqrt(I_s * min_j T_{s,j})          I = balanced accuracy on a held-out
S   = 100 * (Q_A * Q_B * Q_C)^(1/3)          source-distribution pool
```

Source retention `I` is half of `Q` so that an objective cannot buy target-world
robustness by destroying the model.  The geometric mean is used rather than an
outer minimum across settings: an outer minimum would compound with the inner one
and flatten the differences between methods.  Both rules are computed and
compared in `reports/gates.md`; the geometric mean is kept because it
discriminates better, not because it scores lower.

## 4. What the agent can and cannot see

Public: the generators for A and B, their training seeds, the model, the training
loop, the scoring formula, the official world coordinates, the baseline names and
`S*`.

Private: setting C's generator entirely, the official pool seed, and the official
run seeds.

That split is deliberate and its consequence is measured rather than assumed.  An
agent holding A and B's generators can recover the cue signs from raw signals
almost perfectly (`cue_recovery_accuracy` in `reports/gates.md`) and rebuild
approximations of the official A and B worlds.  This is legitimate inference from
public data — arguably good research work — so the task does not pretend it is
impossible.  It gates on it: the **reconstruction ceiling** is the score of the
best configuration selected on agent-rebuilt worlds, and the task is only sound
if that ceiling sits at or below `S*`.

## 5. Escape hatches, and how they are closed

* **Loss scaling.** `AdamW(weight_decay=0)`.  Adam's update is scale-invariant up
  to epsilon; with a non-zero decoupled decay, multiplying the objective by ten
  would cut the effective regularisation strength — a real hyperparameter behind
  an objective's mask, reachable by a one-character edit.  Setting the decay to
  zero closes it at the source rather than trying to catch it with a probe grid.
  The residual slope is measured anyway.
* **Public tuning.** Scored by the public-tuning ceiling gate.
* **Cue reconstruction.** Scored by the reconstruction ceiling gate.
* **Setting fingerprinting.** Branching on first-steps loss statistics is legal
  under the contract — recognising an unfamiliar regime and falling back is a
  reasonable invariance strategy — and the policy is declared in
  `docs/classification_rules.md` before any trial, with the probe's measured
  outcome in `reports/gates.md`.
