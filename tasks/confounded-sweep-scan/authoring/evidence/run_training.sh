#!/usr/bin/env bash
# Simulate the single-sweep calibration replicates, one 12 Mb chromosome and
# one sweep each, grouped by what each sweep turned out to be rather than
# what it was asked to be.
#
#   run_training.sh [OUTDIR]
#
# Several configurations per class, because s spans a factor of 9.4 in the
# shipped zoo and the features that classify a sweep, the width of its
# diversity trough and the depth of that trough all depend on s. One
# configuration per class would calibrate the classifier and both estimators
# at a single strength and leave their behaviour across the range unmeasured.
#
# The configurations are the shipped ones (authoring/provenance/config.py)
# plus, for the classes that carry only two sweeps, a third intermediate
# strength so the calibration is not a two-point fit.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/training}"
PY="${PYTHON:-python3}"
JOBS="${JOBS:-4}"
REPS="${REPS:-3}"

mkdir -p "$OUT"

# label                s      origin  n_origins  tag
CONFIGS=(
  "hard_complete    0.0800     280          1    _s0800"
  "hard_complete    0.0300     710          1    _s0300"
  "hard_complete    0.0085    2300          1    _s0085"
  "hard_complete    0.0250    2500          1    _s0250"
  "soft_complete    0.0180     650        100    _s0180"
  "soft_complete    0.0300     480        100    _s0300"
  "soft_complete    0.0450     380        100    _s0450"
  "hard_partial     0.0140     530          1    _s0140"
  "hard_partial     0.0220     330          1    _s0220"
  "hard_partial     0.0350     230          1    _s0350"
  "soft_partial     0.0110     740         12    _s0110"
  "soft_partial     0.0250     260         12    _s0250"
  "soft_partial     0.0600     120         12    _s0600"
)

i=0
for cfg in "${CONFIGS[@]}"; do
    read -r label s origin nd tag <<<"$cfg"
    seed0=$((600000 + 1000 * i))
    i=$((i + 1))
    (
      OMP_NUM_THREADS=1 "$PY" "$HERE/make_training.py" \
          --label "$label" --s "$s" --origin "$origin" --n-origins "$nd" \
          --tag "$tag" --n "$REPS" --seed0 "$seed0" --outdir "$OUT" \
          > "$OUT/log_${label}${tag}.txt" 2>&1
    ) &
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
done
wait

echo
echo "replicates written:"
ls -1 "$OUT"/*.npz | wc -l
echo
grep -h 'realized=' "$OUT"/log_*.txt | sort
