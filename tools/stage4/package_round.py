#!/usr/bin/env python3
import argparse,hashlib,zipfile
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--round-id',required=True); ap.add_argument('--round-dir',type=Path,required=True); ap.add_argument('--downloads-dir',type=Path,required=True); ap.add_argument('--max-file-mb',type=float,default=200); a=ap.parse_args(); a.downloads_dir.mkdir(parents=True,exist_ok=True)
 if not (a.round_dir/'summary.md').exists(): (a.round_dir/'summary.md').write_text('# Summary\n')
 ck=[]; large=[]; files=[]
 for p in sorted(a.round_dir.rglob('*')):
  if not p.is_file() or p.suffix.lower() in {'.pt','.pth','.ckpt','.npz','.npy'}: 
   if p.is_file() and p.suffix.lower() in {'.pt','.pth','.ckpt'}: ck.append(p)
   continue
  if p.stat().st_size>a.max_file_mb*1024*1024: large.append(p); continue
  files.append(p)
 (a.round_dir/'checkpoint_manifest.tsv').write_text('path\tsize_bytes\tnote\n'+'\n'.join(f'{p.resolve()}\t{p.stat().st_size}\tcheckpoint_omitted' for p in ck)+'\n')
 (a.round_dir/'large_file_manifest.tsv').write_text('path\tsize_bytes\treason\n'+'\n'.join(f'{p.relative_to(a.round_dir)}\t{p.stat().st_size}\tlarge' for p in large)+'\n')
 files = [p for p in files if p.name not in ('checkpoint_manifest.tsv','large_file_manifest.tsv')]
 files += [a.round_dir/'checkpoint_manifest.tsv',a.round_dir/'large_file_manifest.tsv']; z=a.downloads_dir/(a.round_id+'.zip'); z.unlink(missing_ok=True)
 with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
  for p in files: f.write(p,p.relative_to(a.round_dir))
 with zipfile.ZipFile(z) as f: assert f.testzip() is None
 h=hashlib.sha256(z.read_bytes()).hexdigest(); z.with_suffix('.zip.sha256').write_text(f'{h}  {z.name}\n'); (a.downloads_dir/(a.round_id+'_unzip_test.txt')).write_text('No errors detected in compressed data.\n'); print(z); print(h)
if __name__=='__main__': main()
