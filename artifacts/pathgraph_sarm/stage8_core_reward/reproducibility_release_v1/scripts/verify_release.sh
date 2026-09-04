#!/usr/bin/env bash
set -euo pipefail
release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$release_root"
sha256sum -c RELEASE_SHA256SUMS.txt
test -f configs/final_claim_scope.json
test -f results/reproduced_reward_main_table.csv
test -s manifests/external_artifact_manifest.tsv
awk -F '\t' 'NR==1 { if ($1 != "path") exit 1 } NR>1 { if ($1 == "") exit 1 }' manifests/external_artifact_manifest.tsv
echo VERIFY_RELEASE_PASS
