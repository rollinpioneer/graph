#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,zipfile
from pathlib import Path
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--round-zip-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve();out=a.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
 # Keep only one final archive in the user-facing directory; round ZIPs are /tmp only.
 (root/'final_package_manifest.tsv').write_text('archive\tsha256\n')
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(root.rglob('*')):
   if not f.is_file() or f==out:continue
   rel=f.relative_to(root)
   if f.suffix in {'.pt','.pth','.ckpt','.safetensors','.bin','.npz','.parquet'}:continue
   if rel.name in {'combined_six_seed_rollouts.csv','refine1_new_seed_rollouts.csv'}:continue
   if 'episodes' in rel.parts or ('jobs' in rel.parts and (f.suffix=='.log' or rel.name=='rollouts.csv')):continue
   z.write(f,Path('stage6_refine1')/rel)
 if not zipfile.is_zipfile(out):raise RuntimeError('final archive is invalid')
 print(str(out));print(digest(out))
if __name__=='__main__':main()
