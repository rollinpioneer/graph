#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--effects',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--new-seeds',required=True);a=p.parse_args();d=pd.read_csv(a.effects);seeds=list(map(int,a.new_seeds.split(',')));x=d[(d.seed_block=='new')&(d.metric=='graph_task_success')].set_index('policy_seed').loc[seeds];result={'new_seed_count':len(seeds),'new_seed_improved_count':int(x.improved.sum()),'new_seed_ceiling_tie_count':int(x.ceiling_tie.sum()),'new_seed_degraded_count':int(x.degraded.sum()),'per_seed_difference':{str(s):float(x.loc[s,'difference']) for s in seeds},'strict_improvement_definition':'difference > 0; ceiling ties are not improvements'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');a.report.write_text('# New-seed replication\n\n'+json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
