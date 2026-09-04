#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv
from pathlib import Path
from tools.stage8.common import write_csv
def rows(path:Path):
 with path.open(newline='',encoding='utf-8') as h:
  sample=h.read(2048);h.seek(0);return list(csv.DictReader(h,delimiter='\t' if sample.count('\t')>sample.count(',') else ','))
def main():
 p=argparse.ArgumentParser()
 for x in ('checkpoint_manifest','input_large_files','reproduction_large_files','statistics_large_files'):p.add_argument('--'+x.replace('_','-'),type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=[]
 for source in (a.checkpoint_manifest,a.input_large_files,a.reproduction_large_files,a.statistics_large_files):
  for row in rows(source):
   path=row.get('path') or row.get('checkpoint_path') or row.get('source_path')
   if not path:continue
   out.append({'path':path,'size_bytes':row.get('size_bytes',''),'sha256':row.get('sha256',''),'artifact_type':row.get('artifact_type','checkpoint' if source==a.checkpoint_manifest else 'external_artifact'),'source_manifest':str(source.resolve()),'reason_externalized':row.get('reason_omitted','lightweight_release_excludes_large_or_source_data'),'required_for_full_recompute':row.get('required_for_full_recompute','true')})
 unique={row['path']:row for row in out};write_csv(a.output,[unique[k] for k in sorted(unique)],delimiter='\t')
if __name__=='__main__':main()
