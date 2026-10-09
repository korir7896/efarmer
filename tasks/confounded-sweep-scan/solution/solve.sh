#!/usr/bin/env bash
# Reference solution. Infers the demographic history, simulates a null under
# it with the provided recombination map, and scans for sweeps against that
# null rather than against the genome's own empirical distribution.
set -euo pipefail

# Single-threaded for reproducibility: the grading tolerances are wide enough
# that thread-order differences would not change the verdict, but a
# deterministic oracle is worth more than a slightly faster one.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$HERE/pipeline/main.py" \
    --vcf /app/data/variants.vcf.gz \
    --rate-map /app/data/recombination_map.bed \
    --out /app/results.json \
    --mu 2.0e-8 \
    --n-null 10

test -s /app/results.json
python - <<'PY'
import json
d = json.load(open("/app/results.json"))
assert "demography" in d and "sweeps" in d, "results.json is missing a top-level key"
print(f"results.json: {len(d['sweeps'])} sweep call(s)")
PY
