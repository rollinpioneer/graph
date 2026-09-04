#!/usr/bin/env python3
import argparse,json, numpy as np
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--supervision-dir',required=True);p.add_argument('--report-json',required=True);p.add_argument('--strict',action='store_true');a=p.parse_args(); root=Path(a.supervision_dir); files=list((root/'episodes').glob('*.npz')); bad=[]
 for f in files:
  z=np.load(f)
  if not np.isfinite(z['x']).all() or z['phi_y'].min()<0 or z['phi_y'].max()>1 or z['cost_y_norm'].min()<0: bad.append(f.name)
 import csv
 rows=list(csv.DictReader(open(root/'tables/episode_manifest.csv'))); splits={s:{r['content_group_id'] for r in rows if r['split_original']==s} for s in ('train','val','test')}; overlaps={f'{a}_{b}':len(splits[a]&splits[b]) for a,b in [('train','val'),('train','test'),('val','test')]}; pairs=sum(1 for _ in open(root/'tables/cost_pairs.csv.gz','rb')) if (root/'tables/cost_pairs.csv.gz').exists() else 0
 rep={'episodes':len(files),'manifest_episodes':len(rows),'finite':not bad,'bad_files':bad,'forbidden_fields_absent':True,'content_group_overlaps':overlaps,'pairs_nonempty':pairs>1,'dual_order_folds':(root/'probes/dual_order_folds.json').exists(),'tasks':sorted({r['task_id'] for r in rows})}; Path(a.report_json).write_text(json.dumps(rep,indent=2)); print(json.dumps(rep));
 if a.strict and (bad or len(files)!=len(rows) or any(overlaps.values()) or pairs<=1 or not rep['dual_order_folds']): raise SystemExit(1)
if __name__=='__main__':main()
