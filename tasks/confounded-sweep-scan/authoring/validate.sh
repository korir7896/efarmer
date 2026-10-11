#!/usr/bin/env bash
# Run the reference against the assembled bundle and grade it, outside harbor.
#
#   validate.sh [WORKDIR]
#
# Checks the two things that matter most: that the reference solution reaches
# reward 1 on the shipped data, and that it does so reproducibly.
set -euo pipefail

TASK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$(mktemp -d)}"
PY="${PYTHON:-python3}"
mkdir -p "$WORK"

test -f "$TASK/environment/data/variants.vcf.gz" || {
    echo "environment/data is not populated; run assemble.sh first"; exit 1; }
test -f "$TASK/tests/truth.json" || {
    echo "tests/truth.json is missing; run assemble.sh first"; exit 1; }

echo "=== preflight ==="
PYTHONPATH="$TASK/solution" "$PY" - <<'PY'
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

echo "=== running the reference pipeline ==="
time "$PY" "$TASK/solution/pipeline/main.py" \
    --vcf "$TASK/environment/data/variants.vcf.gz" \
    --rate-map "$TASK/environment/data/recombination_map.bed" \
    --out "$WORK/results.json" \
    --mu 2.0e-8 --n-null 10 2>&1 | tail -30

echo
echo "=== grading it with the sealed verifier ==="
TRUTH_PATH="$TASK/tests/truth.json" SUBMISSION_PATH="$WORK/results.json" \
    PYTHONPATH="$TASK/tests" \
    "$PY" -m pytest "$TASK/tests/test_sweep_scan.py" -p no:cacheprovider -v --no-header 2>&1 | tail -30

echo
echo "=== the submission ==="
"$PY" -c "
import json
d = json.load(open('$WORK/results.json'))
print(json.dumps(d['demography'], indent=2))
print(f\"{len(d['sweeps'])} calls:\")
for s in sorted(d['sweeps'], key=lambda x: (x['chrom'], x['position'])):
    print(f\"  chr{s['chrom']}:{s['position']:>9,}  {s['class']:<22} \"
          f\"s={s['selection_coefficient']:.4f} t={s['time_gens_ago']}\")
"
echo
echo "=== gate feasibility, triviality and the design guarantee ==="
"$PY" "$TASK/authoring/evidence/gate_feasibility.py"

echo
echo "=== saddle rule self-test ==="
"$PY" "$TASK/authoring/evidence/merge_calibration.py" --selftest

echo
echo "=== instruction.md against the graded thresholds ==="
"$PY" "$TASK/authoring/evidence/instruction_audit.py"

echo
echo "=== verifier self-test against the shipped truth ==="
"$PY" "$TASK/authoring/evidence/verifier_selftest.py"

echo
echo "results in $WORK"
