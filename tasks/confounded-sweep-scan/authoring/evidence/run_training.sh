#!/usr/bin/env bash
# Simulate the single-sweep calibration replicates, one chromosome and one
# sweep each, grouped by what each sweep turned out to be rather than what it
# was asked to be.
#
#   run_training.sh [OUTDIR]            env: SET, REPS, JOBS, PYTHON, SEED_BASE
#
# Several configurations per class, because s spans a factor of 12 in the
# shipped zoo and the features that classify a sweep, the width of its
# diversity trough and the depth of that trough all depend on s. One
# configuration per class would calibrate the classifier and both estimators
# at a single strength and leave their behaviour across the range unmeasured.
#
# Four sets, and the split is what makes the honesty claims checkable:
#
#   shipped   the fifteen configurations the shipped replicate uses.
#   coldspot  five of those re-run with the sweep inside a wide cold spot,
#             which is where three of the shipped sweeps sit. Without these
#             the local-rate scaling the reference applies to every width
#             feature would be asserted rather than measured.
#   holdout   sixteen configurations the shipped replicate does NOT use.
#             Every entry differs from all fifteen shipped ones in s AND in
#             origin, and the ages they realize span the same range.
#   holdcold  four holdout configurations inside a cold spot, so the
#             out-of-sample refit in holdout_calibration.py can calibrate the
#             rate scaling too and is not handicapped against the shipped fit.
#
# Rows are: label  s  origin  n_origins  cold_factor  tag
# cold_factor is "-" for the base rate throughout.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/training}"
PY="${PYTHON:-python3}"
JOBS="${JOBS:-3}"
REPS="${REPS:-3}"

mkdir -p "$OUT"

# The fifteen shipped configurations, copied from provenance/config.py
# SWEEP_SPEC. Origins and origin counts are the ones the probe runs recorded
# in ACCEPT settled on, not the first guesses.
CONFIGS_SHIPPED=(
  "hard_complete  0.0900   274    1  -  _s0900"
  "hard_complete  0.0075  1569    1  -  _s0075"
  "hard_complete  0.0400  1083    1  -  _s0400"
  "hard_complete  0.0130  2492    1  -  _s0130"
  "hard_complete  0.0550  2722    1  -  _s0550"
  "soft_complete  0.0220   700  300  -  _s0220"
  "soft_complete  0.0500   420   80  -  _s0500"
  "soft_complete  0.0095  1500  150  -  _s0095"
  "hard_partial   0.0160   500    1  -  _s0160"
  "hard_partial   0.0380   210    1  -  _s0380"
  "hard_partial   0.0085   850    1  -  _s0085"
  "hard_partial   0.0650   160    1  -  _s0650"
  "soft_partial   0.0120   430  150  -  _s0120"
  "soft_partial   0.0580   130   14  -  _s0580"
  "soft_partial   0.0300   230   40  -  _s0300"
)

# The three shipped cold-spot sweeps at their own rate factors, plus two more
# so the scaling is measured across a range of factors rather than at three
# points that happen to be the shipped ones.
CONFIGS_COLDSPOT=(
  "hard_complete  0.0550  2722    1  0.22  _cs_s0550"
  "soft_complete  0.0095  1500  150  0.28  _cs_s0095"
  "hard_partial   0.0380   210    1  0.25  _cs_s0380"
  "hard_complete  0.0400  1083    1  0.25  _cs_s0400"
  "soft_partial   0.0300   230   40  0.22  _cs_s0300"
)

CONFIGS_HOLDOUT=(
  "hard_complete  0.0700   350    1  -  _ho_s0700"
  "hard_complete  0.0100  1250    1  -  _ho_s0100"
  "hard_complete  0.0480   900    1  -  _ho_s0480"
  "hard_complete  0.0170  2200    1  -  _ho_s0170"
  "hard_complete  0.0250  2600    1  -  _ho_s0250"
  "soft_complete  0.0280   620  250  -  _ho_s0280"
  "soft_complete  0.0420   460   90  -  _ho_s0420"
  "soft_complete  0.0110  1350  170  -  _ho_s0110"
  "hard_partial   0.0200   450    1  -  _ho_s0200"
  "hard_partial   0.0320   240    1  -  _ho_s0320"
  "hard_partial   0.0105   780    1  -  _ho_s0105"
  "hard_partial   0.0520   190    1  -  _ho_s0520"
  "soft_partial   0.0145   400  120  -  _ho_s0145"
  "soft_partial   0.0500   150   18  -  _ho_s0500"
  "soft_partial   0.0260   260   45  -  _ho_s0260"
  "soft_partial   0.0350   210   35  -  _ho_s0350"
)

CONFIGS_HOLDCOLD=(
  "hard_complete  0.0480   900    1  0.24  _hoc_s0480"
  "soft_complete  0.0280   620  250  0.26  _hoc_s0280"
  "hard_partial   0.0320   240    1  0.23  _hoc_s0320"
  "soft_partial   0.0260   260   45  0.27  _hoc_s0260"
)

case "${SET:-all}" in
  shipped)  CONFIGS=("${CONFIGS_SHIPPED[@]}") ;;
  coldspot) CONFIGS=("${CONFIGS_COLDSPOT[@]}") ;;
  holdout)  CONFIGS=("${CONFIGS_HOLDOUT[@]}") ;;
  holdcold) CONFIGS=("${CONFIGS_HOLDCOLD[@]}") ;;
  all)      CONFIGS=("${CONFIGS_SHIPPED[@]}" "${CONFIGS_COLDSPOT[@]}"
                     "${CONFIGS_HOLDOUT[@]}" "${CONFIGS_HOLDCOLD[@]}") ;;
  *) echo "SET must be shipped, coldspot, holdout, holdcold or all"; exit 1 ;;
esac

# Throttle on a counter of our own PIDs rather than on `jobs`: `$(jobs -rp)`
# runs in a subshell that has no jobs of its own, so it always reports none
# and the loop launches every configuration at once. That is not a style
# point -- it is what made an earlier run put sixteen fwdpy11 processes on
# four cores and finish a third of them.
pids=()
reap() {
    local alive=()
    for p in "${pids[@]:-}"; do
        [ -n "$p" ] && kill -0 "$p" 2>/dev/null && alive+=("$p")
    done
    pids=("${alive[@]:-}")
}

i=0
for cfg in "${CONFIGS[@]}"; do
    read -r label s origin nd cold tag <<<"$cfg"
    seed0=$((${SEED_BASE:-600000} + 1000 * i))
    i=$((i + 1))
    # Already simulated? Skip it. Each replicate is about forty seconds, so a
    # run interrupted part-way through should resume rather than start over.
    # find, not ls: with `set -o pipefail` a glob that matches nothing makes
    # `ls | wc -l` fail and takes the whole script down.
    have=$(find "$OUT" -maxdepth 1 -name "${label}${tag}_*.npz" | wc -l)
    if [ "$have" -ge "$REPS" ]; then
        echo "skip ${label}${tag}: $have replicate(s) already present"
        continue
    fi
    cold_args=()
    [ "$cold" != "-" ] && cold_args=(--coldspot-factor "$cold")
    (
      OMP_NUM_THREADS=1 "$PY" "$HERE/make_training.py" \
          --label "$label" --s "$s" --origin "$origin" --n-origins "$nd" \
          "${cold_args[@]}" \
          --tag "$tag" --n "$REPS" --seed0 "$seed0" --outdir "$OUT" \
          > "$OUT/log_${label}${tag}.txt" 2>&1
    ) &
    pids+=("$!")
    while reap; [ "${#pids[@]}" -ge "$JOBS" ]; do sleep 2; done
done
wait

echo
echo "replicates written:"
find "$OUT" -maxdepth 1 -name '*.npz' | wc -l
echo
grep -h 'realized=' "$OUT"/log_*.txt | sort
