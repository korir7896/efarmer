#!/usr/bin/env bash
# Grade the submission. Writes /logs/verifier/reward.txt containing 0 or 1 on
# every path, including crashes.
set -uo pipefail

LOGDIR=/logs/verifier
mkdir -p "$LOGDIR"
REWARD_FILE="$LOGDIR/reward.txt"

# Default to failure; only an all-green pytest run overwrites this.
echo -n 0 > "$REWARD_FILE"

cd /tests || exit 0

python -m pytest test_sweep_scan.py \
    -p no:cacheprovider \
    --ctrf "$LOGDIR/ctrf.json" \
    -v > "$LOGDIR/pytest.log" 2>&1
STATUS=$?

cat "$LOGDIR/pytest.log" || true

if [ "$STATUS" -eq 0 ]; then
    echo -n 1 > "$REWARD_FILE"
else
    echo -n 0 > "$REWARD_FILE"
fi

echo "reward=$(cat "$REWARD_FILE")"
exit 0
