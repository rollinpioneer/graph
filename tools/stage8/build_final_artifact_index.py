#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from tools.stage8.common import sha256,write_csv
def main():
 p=argparse.ArgumentParser();p.add_argument('--m6-root',type=Path,required=True);p.add_argument('--round-zip-dir',type=Path,required=True);p.add_argument('--external-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();rows=[]
 for f in sorted(a.m6_root.rglob('*')):
  if f.is_file() and f!=a.output:rows.append({'artifact_path':str(f.relative_to(a.m6_root)),'artifact_type':'m6_compact','size_bytes':f.stat().st_size,'sha256':sha256(f),'external':False})
 rows.append({'artifact_path':str(a.external_manifest.resolve()),'artifact_type':'external_manifest','size_bytes':a.external_manifest.stat().st_size,'sha256':sha256(a.external_manifest),'external':True});write_csv(a.output,rows);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(f'# Final Artifact Index\n\nCompact M6 artifacts indexed: {len(rows)-1}. External checkpoints, source data, raw predictions, and bootstrap distributions remain referenced through the external manifest.\n',encoding='utf-8')
if __name__=='__main__':main()
