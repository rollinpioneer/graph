#!/usr/bin/env python3
from __future__ import annotations
import argparse,pandas as pd
from tools.stage6_refine1.common import policy_seed_rows,write_tsv
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-registry',required=True);p.add_argument('--tasks',required=True);p.add_argument('--new-policy-seeds',required=True);p.add_argument('--output',required=True);a=p.parse_args();parent=pd.read_csv(a.parent_registry);new=list(map(int,a.new_policy_seeds.split(',')));tasks=a.tasks.split(',');old=set(parent.policy_seed);assert not old.intersection(new)
 rows=[r for r in policy_seed_rows(new) if r['task_id'] in tasks];assert len(rows)==6;write_tsv(a.output,rows);print(f'REFINE1_SEED_REGISTRY_OK rows={len(rows)}')
if __name__=='__main__':main()
