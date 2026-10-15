#!/usr/bin/env bash
set -euo pipefail
[ "$#" -eq 2 ] || { echo "Usage: $0 ROUND_ID ROUND_DIR" >&2; exit 2; }
ROUND_ID="$1"; ROUND_DIR="$(realpath "$2")"; ROOT="${STAGE3_ROOT:-$(realpath artifacts/pathgraph_sarm/stage3)}"
mkdir -p "$ROOT/downloads"; test -s "$ROUND_DIR/run_manifest.md"; test -s "$ROUND_DIR/summary.md"
grep -q '^- finished_at:' "$ROUND_DIR/run_manifest.md" || echo "- finished_at: $(date -Iseconds)" >> "$ROUND_DIR/run_manifest.md"
printf 'path\tsize_bytes\tartifact_type\treason_omitted\n' > "$ROUND_DIR/large_file_manifest.tsv"
printf 'path\tsize_bytes\tjob_id\tepoch_or_step\tmetric\n' > "$ROUND_DIR/checkpoint_manifest.tsv"
find "$ROUND_DIR" -type f \( -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name '*.safetensors' -o -name '*.pkl' -o -name '*.hdf5' -o -name '*.mp4' \) -printf '%p\t%s\theavy\tdefault_omit_from_zip\n' >> "$ROUND_DIR/large_file_manifest.tsv" || true
ZIP="$ROOT/downloads/${ROUND_ID}.zip"; rm -f "$ZIP" "$ZIP.sha256"
(cd "$ROUND_DIR" && find . -type f ! -path '*/__pycache__/*' ! -name '*.ckpt' ! -name '*.pt' ! -name '*.pth' ! -name '*.bin' ! -name '*.safetensors' ! -name '*.pkl' ! -name '*.hdf5' ! -name '*.mp4' -print | LC_ALL=C sort | zip -q "$ZIP" -@)
sha256sum "$ZIP" > "$ZIP.sha256"; unzip -t "$ZIP" > "$ROOT/downloads/${ROUND_ID}_unzip_test.txt"
