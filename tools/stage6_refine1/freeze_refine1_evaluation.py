#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from tools.stage6_refine1.common import sha256
def main():
 p=argparse.ArgumentParser();p.add_argument('--selection-lock',type=Path,required=True);p.add_argument('--seed-registry',type=Path,required=True);p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d={'locked':True,'checkpoint_selection_lock_sha256':sha256(a.selection_lock),'parent_evaluation_seed_registry_sha256':sha256(a.seed_registry),'new_rollouts_sha256':sha256(a.rollouts),'test_used_for_selection':False,'paired_evaluation':True,'policy_seeds':[20260912,20260913,20260914],'methods':['linear_sarm_equiv','pathgraph_reward_v1_locked'],'rollout_count':1500};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':main()
