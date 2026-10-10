#!/usr/bin/env bash
# Move a generated replicate into the bundle and re-run every check.
#
#   assemble.sh GENERATED_DIR
#
# GENERATED_DIR is the output of authoring/provenance/generate.py and must
# contain public/ and sealed/.
set -euo pipefail

SRC="${1:?usage: assemble.sh GENERATED_DIR}"
TASK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"

test -f "$SRC/public/variants.vcf.gz" || { echo "no variants.vcf.gz in $SRC/public"; exit 1; }
test -f "$SRC/sealed/truth.json"      || { echo "no truth.json in $SRC/sealed"; exit 1; }

echo "=== copying agent-facing data into the environment image ==="
cp "$SRC/public/variants.vcf.gz"      "$TASK/environment/data/"
cp "$SRC/public/recombination_map.bed" "$TASK/environment/data/"
cp "$SRC/public/README_DATA.txt"        "$TASK/environment/data/"
ls -la "$TASK/environment/data/"

echo
echo "=== sealing the truth into the verifier image ==="
cp "$SRC/sealed/truth.json" "$TASK/tests/truth.json"

echo
echo "=== provenance ==="
cp "$SRC/provenance.json" "$TASK/authoring/provenance/generated_provenance.json"
"$PY" - "$TASK/authoring/provenance/generated_provenance.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print("  accepted replicate", p["accepted_replicate"], "seed", p["accepted_seed"])
print("  versions:", p["versions"])
for k, v in p["sha256"].items():
    print(f"  sha256 {k}: {v}")
PY

echo
echo "=== the instance, from the sealed truth ==="
"$PY" "$TASK/authoring/evidence/doc_facts.py"

echo
echo "=== leak audit ==="
"$PY" "$TASK/authoring/evidence/leak_audit.py" "$TASK/environment/data" "$TASK/tests/truth.json"

echo
echo "=== verifier self-test ==="
"$PY" "$TASK/authoring/evidence/verifier_selftest.py"

echo
echo "assembled"
