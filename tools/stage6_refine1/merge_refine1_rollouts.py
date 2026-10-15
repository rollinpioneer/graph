#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--job-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--duplicates',type=Path,required=True);a=p.parse_args();jobs=list(csv.DictReader(a.jobs.open(),delimiter='\t'));frames=[]
 for j in jobs:
  f=Path(j['output_dir'])/'rollouts.csv';assert f.exists(),f;frames.append(pd.read_csv(f))
 d=pd.concat(frames,ignore_index=True);key=['task_id','condition','method','policy_seed','rollout_index'];dups=d[d.duplicated(key,False)];a.duplicates.parent.mkdir(parents=True,exist_ok=True);dups.to_csv(a.duplicates,index=False);assert len(d)==1500 and d.duplicated(key).sum()==0
 d.to_csv(a.output,index=False);a.summary.write_text(json.dumps({'rows':len(d),'jobs':len(jobs),'duplicates':len(dups),'methods':sorted(d.method.unique()),'policy_seeds':sorted(map(int,d.policy_seed.unique()))},indent=2)+'\n');print(json.dumps({'rows':len(d),'duplicates':len(dups)},indent=2))
if __name__=='__main__':main()
