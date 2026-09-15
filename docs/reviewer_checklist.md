# Reviewer checklist

A short guide to auditing CrossPhase without rerunning everything.

| Question | Where to look |
|---|---|
| Is the research question original? | `docs/originality.md` |
| Why is the task shaped this way? | `docs/task_design.md` |
| What may the agent change? | the SOLUTION block in `agent/solution.py`; rules in `docs/classification_rules.md` |
| Are three published methods reproduced under one setup? | `reports/baselines.csv`, `agent/method_references.md` |
| Is the shipped starting code the weakest of them? | `reports/baselines.json` -> `weakest`, compared with `agent/solution.py` |
| What is the score to beat? | `reports/baselines.json` -> `S_star` |
| Does public tuning alone already win? | `reports/gates.md`, public-tuning ceiling row |
| Can the agent rebuild the targets and win that way? | `reports/gates.md`, reconstruction ceiling row |
| Is the hidden setting a real shift? | `docs/task_design.md` §2, `grader/private_specs.py` |
| Is there anything to win? | `reports/gates.md`, headroom row |
| Which world binds the score? | `reports/gates.md`, binding-world audit; per-cell in `reports/baselines.csv` |
| Was the fingerprint question decided in advance? | `docs/classification_rules.md`, committed before any trial |

## Reproducing from scratch

```bash
pip install -r requirements.txt
python -m tests.run                     # scaffold invariants, ~1 min
python -m baselines.run_baselines       # reproduces all six families, ~40 min on 4 cores
python -m gates.run_gates               # the gate suite, ~40 min on 4 cores
python -m grader.grade                  # grade whatever is in agent/solution.py
```

Every number in `reports/` comes from those three commands on a 4-core CPU
container.  No GPU, no downloads, no data files: every array in the task is
generated from an integer seed.
