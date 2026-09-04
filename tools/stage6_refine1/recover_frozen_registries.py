#!/usr/bin/env python3
"""Recover missing explicit registries from the already frozen Stage-6 evidence.

The prior run stored paired_eval_seed in every rollout but did not materialise a
registry.  This script derives the registry from those immutable rows and asserts
that every key/seed agrees, making the provenance explicit rather than inventing a
new evaluation design.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import pandas as pd
from tools.stage6_refine1.common import OLD_SEEDS,TASKS,CONDITIONS,policy_seed_rows,write_tsv,write_json,sha256
def main():
 p=argparse.ArgumentParser();p.add_argument('--old-rollouts',type=Path,required=True);p.add_argument('--protocol-root',type=Path,required=True);p.add_argument('--eval-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=pd.read_csv(a.old_rollouts)
 assert set(d.policy_seed)==set(OLD_SEEDS)
 # Parent registry was omitted in the original Stage-6 implementation; create its
 # canonical six rows from the exact seeds that generated the selected checkpoints.
 parent=a.protocol_root/'seed_registry/policy_seed_registry.csv';write_tsv(parent,policy_seed_rows(OLD_SEEDS))
 registry=[]
 for (task,cond,rid),q in d.groupby(['task_id','condition','rollout_id']):
  vals=set(q.paired_eval_seed)
  assert len(vals)==1 and len(q)==12, (task,cond,rid,len(q),vals)
  registry.append({'task_id':task,'condition':cond,'rollout_index':int(rid),'env_seed':int(next(iter(vals))),'source':'recovered_from_frozen_test_rollouts'})
 registry=sorted(registry,key=lambda x:(x['task_id'],x['condition'],x['rollout_index']))
 out=a.eval_root/'locks/evaluation_seed_registry.csv';write_tsv(out,registry)
 summary={'source_rollouts':str(a.old_rollouts.resolve()),'source_sha256':sha256(a.old_rollouts),'parent_registry':str(parent.resolve()),'evaluation_registry':str(out.resolve()),'conditions':len(registry),'recovery_method':'exact unique paired_eval_seed per frozen rollout key','new_random_seeds_generated':False}
 write_json(a.out/'recovered_registry_summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
