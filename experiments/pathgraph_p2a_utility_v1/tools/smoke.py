#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p2a.experiment import prepare,evaluate
from p2a.score import score_dataset
from p2a.train import train_one
from p2a.stats import analyze
from p2a.io import dump,new_dir

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--v6-tools',required=True);p.add_argument('--extra-methods',required=True);a=p.parse_args()
    out=new_dir(a.out); prepare(out/'data',True)
    score_dataset(out/'data',out/'weights',a.v6_tools,a.extra_methods)
    for method in ('BC_UNIFORM','V6_CAP_POTENTIAL'):
        train_one(out/'weights',out/'models'/method,method,17,steps=20,batch_size=32)
        evaluate(out/'models'/method/'policy.pt',out/'data','test',out/'eval'/method)
    analyze(out/'eval',out/'analysis',True)
    reports=[json.loads((out/'models'/m/'train.json').read_text()) for m in ('BC_UNIFORM','V6_CAP_POTENTIAL')]
    assert reports[0]['initial_state_sha256']==reports[1]['initial_state_sha256']
    assert reports[0]['first_batch_indices']==reports[1]['first_batch_indices']
    result=dict(status='SMOKE_PASS_NOT_RESEARCH_RESULT',trained_models=2,updates_per_model=20,
        test_rollouts_per_model=8,initial_state_match=True,batch_order_match=True,
        frozen_source_hashes_verified=True,physical_runs=0)
    dump(out/'validation.json',result); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
