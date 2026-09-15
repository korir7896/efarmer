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
| Family | Selected configuration | `S` |
|---|---|---|
| IRM | `lam=100000,warm=0.5` | **57.71** |
| EQRM | `coef=44.721,warm=0.1` | **51.33** |
| VREx | `lam=100000,warm=0.1` | **45.56** |
| GroupDRO | `eta=100,warm=0.5` | **40.87** |
| ERM | `(none)` | **36.64** |
| SD | `lam=0.001,warm=0.5` | **36.62** |

`S*` = **57.71** (IRM, `lam=100000,warm=0.5`).  The weakest family is SD at 36.62, and that is what `agent/solution.py` ships with.  Scores are means over 5 run seeds; the spread between weakest and strongest is 21.09 points.

Each family is shown at its best official configuration.  Letting the public proxy diagnostic pick each family's configuration instead would set the bar at 57.14 rather than 57.71.  The proxy therefore ranks well enough to be worth running and badly enough that the best configuration it finds still falls short -- but read it as a relative ranking of your own candidates, never as an estimate of this number.  2 of the 61 swept configurations exceed `S*`.
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

## Layout, and what is visible during a trial

Visible to a solving agent:

```
README.md                     this file
crossphase/core/              frozen scaffold: generator (A and B), model,
                              training loop, metrics, the six reference methods
agent/solution.py             the one function you may edit
agent/evaluate_public.py      the budgeted proxy diagnostic
agent/method_references.md    papers, forms, and what the contract excludes
docs/classification_rules.md  what counts as a legal win
tests/                        scaffold invariants
```

Withheld during a trial — author and reviewer material:

```
grader/                       the official grader, the private pool and run
                              seeds, and setting C's generator
reports/                      the full sweep, the per-setting breakdown of every
                              baseline, and the gate measurements
baselines/  gates/  tools/    the reproduction and gate harnesses
docs/task_design.md           why the task is shaped the way it is
docs/originality.md  docs/reviewer_checklist.md
```

The per-setting scores `Q_A`, `Q_B` and `Q_C` of each baseline live in
`reports/baselines.csv` rather than in this file: publishing `Q_C` would hand
over a partial reading of the hidden setting.  The combined `S` of every family,
and `S*`, are published above, because a task that hides the score to beat turns
a research problem into a guessing game.

## Gates

Every gate below blocked release; the numbers behind them are in
[`reports/gates.md`](reports/gates.md).

<!-- AUTO:gates -->
| Gate | Verdict |
|---|---|
| public tuning ceiling | PASS |
| reconstruction ceiling | PASS |
| proxy correlation | PASS |
| transition separation | PASS |
| warmup anti transfer | PASS |
| rank reversal | PASS |
| penalty anti transfer | PASS |
| loss scale routes | PASS |
| binding world audit | FAIL |
| fingerprint policy | PASS |
| headroom | PASS |

Full numbers in [`reports/gates.md`](reports/gates.md).
<!-- /AUTO:gates -->

## Running it

```bash
pip install -r requirements.txt
python -m tests.run                  # scaffold invariants
python -m agent.evaluate_public      # the public diagnostic (budgeted)
python -m grader.grade               # author/reviewer only
```

CPU only, no GPU, no network, no data files.
