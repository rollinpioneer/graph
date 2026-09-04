#!/usr/bin/env bash
set -euo pipefail
[ "$#" -eq 2 ] || { echo "Usage: $0 ROUND_ID ROUND_DIR" >&2; exit 2; }
ROUND_ID="$1"; ROUND_DIR="$(realpath "$2")"; ROOT="${STAGE2_ROOT:-$(realpath artifacts/pathgraph_sarm/stage2)}"; mkdir -p "$ROOT/downloads"
test -s "$ROUND_DIR/run_manifest.md"; test -s "$ROUND_DIR/summary.md"
printf 'path\tsize_bytes\tartifact_type\treason_omitted\n' > "$ROUND_DIR/large_file_manifest.tsv"
printf 'path\tsize_bytes\tjob_id\tepoch_or_step\tmetric\n' > "$ROUND_DIR/checkpoint_manifest.tsv"
find "$ROUND_DIR" -type f \( -name '*.pkl' -o -name '*.mp4' -o -name '*.hdf5' -o -name '*.ckpt' -o -name '*.pt' \) -printf '%p\t%s\theavy\tdefault_omit_from_zip\n' >> "$ROUND_DIR/large_file_manifest.tsv" || true
ZIP="$ROOT/downloads/${ROUND_ID}.zip"; rm -f "$ZIP" "$ZIP.sha256"; (cd "$ROUND_DIR" && find . -type f ! -name '*.pkl' ! -name '*.mp4' ! -name '*.hdf5' ! -name '*.ckpt' ! -name '*.pt' -print | sort | zip -q "$ZIP" -@)
sha256sum "$ZIP" > "$ZIP.sha256"; unzip -tq "$ZIP" > "$ROOT/downloads/${ROUND_ID}_unzip_test.txt"
