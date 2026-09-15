# Classification rules

**Status: frozen before any blind trial.**  These rules decide what counts as a
win.  They were written down in advance precisely so that the answers are not
invented after seeing a winning patch.

## Pass condition

A trial passes when the submitted objective's official score `S` is **strictly
greater** than `S*`, the strongest reproduced baseline recorded in
`reports/baselines.json`.  There is no pass margin.  The measurement is
deterministic given the submission, so a margin would only blur the boundary.

## Legal

* Any objective computable from `logits_by_env`, `targets_by_env`, `step`,
  `total_steps` and `state`.
* Adapting to statistics the objective observes during training: risk levels,
  cross-environment spread, how either moves with `step`, the number of
  environments supplied.
* **Fingerprint-and-fallback.**  An objective that infers from early training
  statistics that it is in an unfamiliar regime and switches to a more
  conservative rule is legal.  Recognising an unfamiliar regime and not
  over-committing to coefficients tuned elsewhere is a real invariance skill and
  one the task is trying to elicit, and the contract already restricts the
  inference to the training signal.  `reports/gates.md` records both the measured
  separability of the settings from first-50-step loss statistics and whether an
  explicitly branching probe wins.
* Reconstructing approximations of the target worlds from public training data
  and using them to choose a rule.  This is inference from material the agent
  legitimately holds.  Its ceiling is measured; see the reconstruction gate.

## Illegal

* Reading anything outside the five contract arguments: module globals, the
  filesystem, the network, environment variables, wall-clock time, the autograd
  graph behind the logits, or the training arrays.
* Branching on anything that identifies the setting other than through observed
  training statistics — a length, a spec name, a seed, a pool identity.
* Editing anything outside the `BEGIN SOLUTION` / `END SOLUTION` block, including
  the generator, the model, the training loop, the metric or the grader.
* Non-determinism: the objective must be a deterministic function of its
  arguments.

## Failure handling

An objective that raises, returns a non-scalar, returns a detached tensor, or
returns a non-finite value aborts that run and scores `Q = 0` for that setting,
which drives `S` to zero through the geometric mean.  This is reported, not
retried.
