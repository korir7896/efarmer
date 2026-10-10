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

# Preflight: fail in a second rather than after a ten-minute run if the
# pipeline is missing a piece. An editing slip once deleted a function that
# sits between two others, and the run got all the way through the
# demography fit and ten null replicates before noticing.
PYTHONPATH="$HERE" python - <<'PY'
from pipeline import characterize, demography, detect, main, sweepstats
need = {
    detect: ("load_rate_map", "scan_genome", "rate_bin", "build_null",
             "sweep_scores", "null_score_threshold", "call_regions"),
    demography: ("observed_summaries", "fit_demography", "History"),
    characterize: ("background_levels", "features", "classify",
                   "estimate_s", "estimate_time", "shape_features",
                   "contiguous_low_width", "contiguous_high_width"),
    sweepstats: ("load_vcf", "windowed_sfs_stats", "garud_scan_bp", "nsl_scan"),
    main: ("main",),
}
missing = [f"{m.__name__}.{a}" for m, attrs in need.items()
           for a in attrs if not hasattr(m, a)]
if missing:
    raise SystemExit("pipeline is incomplete: " + ", ".join(missing))
print("preflight ok")
PY

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
