#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, zipfile
from pathlib import Path

SKIP_EXT={'.pt','.pth','.ckpt','.safetensors','.bin','.npy','.npz','.parquet'}
SKIP_NAMES={'ensemble_val_predictions.jsonl.gz','ensemble_test_predictions.jsonl.gz','ensemble_stage3_diagnostic_predictions.jsonl.gz','combined_six_seed_rollouts.csv'}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--round-dir',type=Path,required=True); p.add_argument('--output-zip',type=Path,required=True); p.add_argument('--max-file-mb',type=float,default=200); a=p.parse_args()
    root=a.round_dir.resolve(); a.output_zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output_zip,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob('*')):
            if not f.is_file(): continue
            rel=f.relative_to(root)
            if f.suffix in SKIP_EXT or f.name in SKIP_NAMES or 'cache' in rel.parts or 'checkpoints' in rel.parts: continue
            if f.stat().st_size > a.max_file_mb*1024*1024: continue
            z.write(f,rel)
    if not zipfile.is_zipfile(a.output_zip): raise RuntimeError('invalid zip')
    h=hashlib.sha256(a.output_zip.read_bytes()).hexdigest(); print(str(a.output_zip)); print(h)
if __name__=='__main__': main()
