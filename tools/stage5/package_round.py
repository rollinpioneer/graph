#!/usr/bin/env python3
import argparse,hashlib,zipfile
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--round-id',required=True);p.add_argument('--round-dir',type=Path,required=True);p.add_argument('--downloads-dir',type=Path,required=True);p.add_argument('--max-file-mb',type=float,default=200);a=p.parse_args();a.downloads_dir.mkdir(parents=True,exist_ok=True); rd=a.round_dir
 if not (rd/'run_manifest.md').exists():(rd/'run_manifest.md').write_text('# Run Manifest\n')
 if not (rd/'summary.md').exists():(rd/'summary.md').write_text('# Summary\n')
 inc=[]; large=[]; ck=[]
 for q in sorted(rd.rglob('*')):
  if not q.is_file():continue
  if q.suffix.lower() in {'.pt','.pth','.ckpt','.safetensors','.bin'}:ck.append(q);continue
  if q.stat().st_size>a.max_file_mb*1024*1024 or q.suffix.lower() in {'.npz','.npy','.mp4','.pkl'}:large.append(q);continue
  if q.name not in {'checkpoint_manifest.tsv','large_file_manifest.tsv','SHA256SUMS.txt'}:inc.append(q)
 (rd/'checkpoint_manifest.tsv').write_text('path\tsize_bytes\tnote\n'+'\n'.join(f'{q.resolve()}\t{q.stat().st_size}\tcheckpoint_omitted' for q in ck)+'\n')
 (rd/'large_file_manifest.tsv').write_text('path\tsize_bytes\treason\n'+'\n'.join(f'{q.relative_to(rd)}\t{q.stat().st_size}\theavy_or_large' for q in large)+'\n');inc += [rd/'checkpoint_manifest.tsv',rd/'large_file_manifest.tsv']
 z=a.downloads_dir/(a.round_id+'.zip');z.unlink(missing_ok=True)
 with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
  for q in inc:f.write(q,q.relative_to(rd))
 with zipfile.ZipFile(z) as f:assert f.testzip() is None
 h=hashlib.sha256(z.read_bytes()).hexdigest();(z.with_suffix('.zip.sha256')).write_text(f'{h}  {z.name}\n');(a.downloads_dir/(a.round_id+'_unzip_test.txt')).write_text('No errors detected in compressed data.\n');print(z);print(h)
if __name__=='__main__':main()
