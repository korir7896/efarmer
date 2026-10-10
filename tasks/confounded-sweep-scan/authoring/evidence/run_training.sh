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
#
# SET=extra adds the second pass. The first pass put the soft_complete
# intermediate at s=0.030 origin 480, and two of its three replicates came
# out HARD: a soft sweep stops being soft once it has been complete for a
# few hundred generations, because the surviving beneficial haplotypes drift
# and one takes over. The measured elapsed time from origin to completion is
# about 148 generations at s=0.030 with a hundred origins, so the origin has
# to sit around 330 for the sweep to be young enough at the present. The
# extra pass also tops up the two measured soft_complete configurations,
# since the first pass lost replicates to the same effect.
CONFIGS_MAIN=(
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

CONFIGS_EXTRA=(
  "soft_complete    0.0300     330        100    _s0300b"
  "soft_complete    0.0180     650        100    _s0180b"
  "soft_complete    0.0450     380        100    _s0450b"
)

# Third pass, matching the shipped sw09 after the in-context probes moved it
# to 150 origins. What the haplotype statistics respond to is the number of
# SURVIVING origins, which is three to five either way, but calibrating at
# the shipped configuration costs ten minutes and removes the question.
CONFIGS_EXTRA2=(
  "soft_partial     0.0110     560        150    _s0110c"
)

# A set of configurations the shipped replicate does NOT use, for testing
# whether the calibration depends on having been fitted at the shipped
# sweeps' own (s, origin, n_origins). Every entry differs from all ten
# shipped configurations in s AND in origin; the ages they realize span
# roughly the same range.
CONFIGS_HOLDOUT=(
  "hard_complete    0.0550     400          1    _ho_s0550"
  "hard_complete    0.0200    1000          1    _ho_s0200"
  "hard_complete    0.0130    1800          1    _ho_s0130"
  "hard_complete    0.0150    2450          1    _ho_s0150"
  "soft_complete    0.0250     420        100    _ho_s0250"
  "soft_complete    0.0350     350        100    _ho_s0350"
  "hard_partial     0.0180     420          1    _ho_s0180"
  "hard_partial     0.0280     280          1    _ho_s0280"
  "soft_partial     0.0160     420         80    _ho_s0160"
  "soft_partial     0.0450     170         30    _ho_s0450"
)

case "${SET:-main}" in
  main)  CONFIGS=("${CONFIGS_MAIN[@]}") ;;
  extra) CONFIGS=("${CONFIGS_EXTRA[@]}") ;;
  extra2) CONFIGS=("${CONFIGS_EXTRA2[@]}") ;;
  holdout) CONFIGS=("${CONFIGS_HOLDOUT[@]}") ;;
  all)   CONFIGS=("${CONFIGS_MAIN[@]}" "${CONFIGS_EXTRA[@]}"
                  "${CONFIGS_EXTRA2[@]}") ;;
  *) echo "SET must be main, extra, extra2, holdout or all"; exit 1 ;;
esac

i=0
for cfg in "${CONFIGS[@]}"; do
    read -r label s origin nd tag <<<"$cfg"
    seed0=$((${SEED_BASE:-600000} + 1000 * i))
    i=$((i + 1))
    # Already simulated? Skip it. Each replicate is a few minutes, so a run
    # interrupted part-way through should resume rather than start over.
    # find, not ls: with `set -o pipefail` a glob that matches nothing makes
    # `ls | wc -l` fail and takes the whole script down.
    have=$(find "$OUT" -maxdepth 1 -name "${label}${tag}_*.npz" | wc -l)
    if [ "$have" -ge "$REPS" ]; then
        echo "skip ${label}${tag}: $have replicate(s) already present"
        continue
    fi
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
find "$OUT" -maxdepth 1 -name '*.npz' | wc -l
echo
grep -h 'realized=' "$OUT"/log_*.txt | sort
