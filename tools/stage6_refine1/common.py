from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np

OLD_SEEDS=(20260909,20260910,20260911)
NEW_SEEDS=(20260912,20260913,20260914)
METHODS=('linear_sarm_equiv','pathgraph_reward_v1_locked')
TASKS=('transport_recovery','transport_dual_order')
CONDITIONS={
    'transport_recovery': ('natural','drop_regrasp','gripper_reopen'),
    'transport_dual_order': ('A_first','B_first'),
}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def write_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def write_tsv(path,rows,fields=None):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows)
    if fields is None: fields=list(rows[0]) if rows else []
    with Path(path).open('w',newline='') as f:
        # Registry files carry a .csv suffix in the frozen Stage-6 contract.
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def policy_seed_rows(seeds):
    return [{'task_id':task,'policy_seed':int(seed),'model_seed':int(seed),'data_seed':int(seed)+100000,'augmentation_seed':int(seed)+200000,'validation_seed_start':int(seed)+300000,'test_seed_registry_id':'stage6_frozen_eval_registry'} for task in TASKS for seed in seeds]

def aggregate_seed(d,method,seed):
    """Equal-condition aggregate used by both original Stage 6 and R1."""
    x=d[(d.method==method)&(d.policy_seed==seed)]
    def rate(task,cond,col='success'):
        q=x[(x.task_id==task)&(x.condition==cond)][col]
        return float(q.mean())
    a=rate('transport_dual_order','A_first'); b=rate('transport_dual_order','B_first')
    n=rate('transport_recovery','natural'); dr=rate('transport_recovery','drop_regrasp'); ro=rate('transport_recovery','gripper_reopen')
    return {'graph_task_success':float(np.mean([n,dr,ro,a,b])),'recovery_success':float(np.mean([dr,ro])),'worst_order_success':min(a,b),'fixed_order_success':float(np.mean([n,a,b])),'long_horizon_completion':float(np.mean([n,dr,ro,a,b])),'order_gap':abs(a-b),'A_first':a,'B_first':b}
