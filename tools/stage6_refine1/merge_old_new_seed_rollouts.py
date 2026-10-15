#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from tools.stage6_refine1.common import OLD_SEEDS,NEW_SEEDS,METHODS
def main():
 p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--duplicates',type=Path,required=True);a=p.parse_args();old=pd.read_csv(a.old);new=pd.read_csv(a.new);old=old[old.method.isin(METHODS)&old.policy_seed.isin(OLD_SEEDS)].copy();new=new[new.method.isin(METHODS)&new.policy_seed.isin(NEW_SEEDS)].copy();old=old.rename(columns={'rollout_id':'rollout_index'});assert len(old)==1500 and len(new)==1500
 d=pd.concat([old,new],ignore_index=True);key=['task_id','condition','method','policy_seed','rollout_index'];dups=d[d.duplicated(key,False)];a.duplicates.parent.mkdir(parents=True,exist_ok=True);dups.to_csv(a.duplicates,index=False);assert len(dups)==0
 d.to_csv(a.output,index=False);a.summary.parent.mkdir(parents=True,exist_ok=True);a.summary.write_text(json.dumps({'old_rows':len(old),'new_rows':len(new),'combined_rows':len(d),'duplicates':len(dups),'old_source_sha256':__import__('hashlib').sha256(a.old.read_bytes()).hexdigest(),'new_source_sha256':__import__('hashlib').sha256(a.new.read_bytes()).hexdigest()},indent=2)+'\n');print(json.dumps({'combined_rows':len(d),'duplicates':len(dups)},indent=2))
if __name__=='__main__':main()
