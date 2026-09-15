# CrossPhase

A small, repeatable machine-learning experiment about **when** an invariance
penalty should engage and **how strongly**, measured across three settings, one
of which you cannot see.

You get working code.  You may change **one function**.  The data, the model, the
training loop, the hyperparameters and the evaluation are fixed.

---

## The research question

A network offered an easy nuisance cue and a harder stable cue fits the nuisance
first and only later — if the objective lets it — transitions to the stable
relationship.  Published invariance objectives all decide two things up front: a
**warm-up fraction**, the point in the budget at which the penalty switches on,
and a **coefficient**, how hard it pushes.  Both are chosen on data you can see.

CrossPhase asks whether those two choices transfer.  It is built so that the
transition happens at a materially different point in the budget in each setting
and the useful penalty strength depends on the local noise floor, so that **no
single fixed objective is optimal in all three**.  That is a measured property of
this task, not a hope — see [`reports/gates.md`](reports/gates.md).

## The data

Two-channel signals.  The label is the **sign of the cross-channel phase lag** of
a core oscillation: invariant to the global phase, to per-example gain and to
anything added equally to both channels, and readable only by comparing the two
channels.  Two nuisance cues — a level cue and a band cue — are much easier to
read and are correlated with the label during training at rates that vary by
environment.  That asymmetry is the shortcut.

Nothing is downloaded and no data file is shipped: every array is generated from
an integer seed.

## The settings

| Setting | Visible? | Nuisance mechanism | Window | Notes |
|---|---|---|---|---|
| **A** | yes | additive DC offset + carrier-frequency selection | 128 | level cue dominant |
| **B** | yes | additive DC offset + carrier-frequency selection | 128 | band cue dominant |
| **C** | **no** | multiplicative gain + monotone drift envelope | 160 | weaker core, higher noise floor |

Setting C is the hidden shift and it changes the *mechanism* of the nuisance
rather than its strength.  Gain preserves the core's shape while rescaling
everything downstream of it, including the logits, so objectives keyed to logit
magnitude or to risk variance behave differently there.  C also inverts the
relationship between an environment's noise and its contamination: in A and B the
noisiest environment is the least contaminated, in C it is the most.

## Scoring

For each setting `s`, over a held-out source pool and five target worlds:

```
Q_s = sqrt( I_s * min_j T_{s,j} )     I = balanced accuracy on the source pool
S   = 100 * ( Q_A * Q_B * Q_C )^(1/3) T_j = balanced accuracy on target world j
```

The five official worlds are

| `(p1, p2, sigma_mult, core_mult)` | stresses |
|---|---|
| `(.10, .10, 1.00, 1.00)` | both cues flipped — shortcut reliance |
| `(.10, .90, 1.00, 1.00)` | cue conflict |
| `(.90, .10, 1.00, 1.00)` | cue conflict, the other way |
| `(.50, .50, 1.60, 1.00)` | cues uninformative, noise floor raised 60% |
| `(.90, .90, 1.00, 0.42)` | cues aligned, core attenuated to 42% |

`p1`/`p2` are the probabilities that each cue agrees with the label.  Three
worlds punish shortcut reliance; two punish a model that has been regularised
into a weak core detector.  Source retention is half of `Q`, so robustness cannot
be bought by destroying the model.

**A submission passes when `S` strictly exceeds `S*`.**  There is no margin.

## Baselines

Six published objectives are reproduced under this exact scaffold.  Each family
is given the same twelve-configuration selection budget, and each family's
configuration is chosen the way you would have to choose it — on the public proxy
diagnostic, never on the official score.

<!-- AUTO:baselines -->
(run `python -m baselines.run_baselines` then `python -m tools.sync_docs`)
<!-- /AUTO:baselines -->

Papers, forms and the methods deliberately excluded by the contract are in
[`agent/method_references.md`](agent/method_references.md).

## What you may change

Exactly one function, between the two markers in
[`agent/solution.py`](agent/solution.py):

```python
def objective(logits_by_env, targets_by_env, step, total_steps, state):
    ...  # returns a 0-dim torch tensor
```

You see per-environment **logits** and **targets**, the step counter, the total
budget, and a `state` dict that persists across the steps of one run.  Nothing
else.  The number of environments varies per run and their order is permuted, so
do not assume a count and do not index positionally.

The full rules — including what counts as a legal win, decided in advance — are
in [`docs/classification_rules.md`](docs/classification_rules.md).

## The public diagnostic

```bash
python -m agent.evaluate_public --note "vrex + adaptive warmup"
```

Trains on A and B and reports a **proxy score**.  Twelve evaluations per trial.

> The proxy score is **not** on the official scale and is **not** comparable to
> `S*`.  Different world coordinates, a different pool seed, smaller pools, four
> worlds rather than five, and no contribution at all from the hidden setting.
> Use it to rank your own candidates against each other.  Trying to read `S*` off
> it will mislead you.

## Layout

```
crossphase/core/    frozen scaffold: generator, model, training loop, metrics
agent/              solution.py (edit this), the public diagnostic, method notes
grader/             official grader, and setting C  (not visible during a trial)
baselines/          reproduces all six families and establishes S*
gates/              the gate suite that had to pass before this task shipped
reports/            every measured number, regenerated by the two commands above
docs/               design notes, classification rules, originality memo
tests/              scaffold invariants
```

## Gates

Every gate below blocked release; the numbers behind them are in
[`reports/gates.md`](reports/gates.md).

<!-- AUTO:gates -->
(run `python -m gates.run_gates` then `python -m tools.sync_docs`)
<!-- /AUTO:gates -->

## Running it

```bash
pip install -r requirements.txt
python -m tests.run                  # scaffold invariants
python -m agent.evaluate_public      # the public diagnostic (budgeted)
python -m grader.grade               # author/reviewer only
```

CPU only, no GPU, no network, no data files.
