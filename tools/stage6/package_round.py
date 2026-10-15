#!/usr/bin/env python3
"""Make a small auditable round archive; large files remain external in a manifest."""
from __future__ import annotations
import argparse, hashlib, zipfile
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--exclude-large',action='store_true');p.add_argument('--exclude-raw',action='store_true');a=p.parse_args();root=a.root.resolve();a.out.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(a.out,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(x for x in root.rglob('*') if x.is_file()):
   rel=f.relative_to(root)
   if a.exclude_large and f.suffix in {'.pt','.npz','.parquet','.gz'}:continue
   if a.exclude_raw and ('episodes' in rel.parts or rel.name in {'frozen_test_rollouts.csv','refine1_new_seed_rollouts.csv','combined_six_seed_rollouts.csv','chunk_weights.csv'} or ('jobs' in rel.parts and rel.name == 'rollouts.csv')):continue
   z.write(f,f.relative_to(root))
 if not zipfile.is_zipfile(a.out):raise RuntimeError('zip verification failed')
 print(f'{a.out}\n{sha(a.out)}')
if __name__=='__main__':main()
