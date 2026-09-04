#!/usr/bin/env bash
set -euo pipefail
[ $# -ge 3 ]
ROUND_ID="$1"; ROUND_DIR="$2"; PURPOSE="$3"; mkdir -p "$ROUND_DIR"/{configs,logs,metrics,plots,jobs,manifests,tables,predictions}
cat > "$ROUND_DIR/run_manifest.md" <<EOF
# Run Manifest
- round_id: $ROUND_ID
- purpose: $PURPOSE
- started_at: $(date -Iseconds)
- statistics_unit: content_group_id
- checkpoint_packaging: omitted_by_default
EOF
