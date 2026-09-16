#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p2a.io import dump
from p2a.weights import METHODS

def audit(root):
    jobs=[json.loads(p.read_text()) for p in Path(root).glob('**/train.json')]
    seeds=sorted({r['policy_seed'] for r in jobs}); errors=[]
    for seed in seeds:
        group=[r for r in jobs if r['policy_seed']==seed]
        if {r['method'] for r in group}!=set(METHODS):errors.append(f'method matrix {seed}')
        for field in ('initial_state_sha256','first_batch_indices','dataset_hash','steps','batch_size'):
            v={json.dumps(r[field],sort_keys=True) for r in group}
            if len(v)!=1:errors.append(f'{field} mismatch {seed}')
    return dict(status='PASS' if not errors else 'FAIL',jobs=len(jobs),seeds=seeds,errors=errors)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if Path(a.out).exists():raise FileExistsError(a.out)
    r=audit(a.models);dump(a.out,r)
    if r['errors']:raise SystemExit(1)
