#!/usr/bin/env python3
"""Frozen, paired closed-loop evaluation for one validation-selected checkpoint."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np,torch
from tools.stage6.policy_env import EpisodeSpec,TransportGraphEnv
from tools.stage6.policy_model import ChunkPolicy
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--rollouts',type=int,default=50);p.add_argument('--device',default='cuda:0');a=p.parse_args();dev=torch.device(a.device if torch.cuda.is_available() else 'cpu');rows=[]
 for ent in csv.DictReader(a.selection.open()):
  ck=Path(ent['checkpoint_path']);got=sha(ck)
  if got!=ent['checkpoint_sha256']:raise RuntimeError(f'checkpoint hash changed: {ck}')
  z=torch.load(ck,map_location='cpu',weights_only=False);m=ChunkPolicy().to(dev);m.load_state_dict(z['model']);m.eval();mean=np.asarray(z['normalizer_mean']);std=np.asarray(z['normalizer_std']);task=ent['task_id'];conditions=['natural','drop_regrasp','gripper_reopen'] if task=='transport_recovery' else ['A_first','B_first']
  for condition in conditions:
   for i in range(a.rollouts):
    spec=EpisodeSpec(task,condition,condition if task=='transport_dual_order' else '',20260920+i);env=TransportGraphEnv(spec);obs=env.reset()
    while not env.done:
     with torch.no_grad():act=m(torch.tensor(((obs-mean)/std)[None],dtype=torch.float32,device=dev))[0,0].cpu().numpy()
     obs,_,_,info=env.step(act)
    rows.append({'task_id':task,'method':ent['method'],'policy_seed':int(ent['policy_seed']),'condition':condition,'rollout_id':i,'paired_eval_seed':20260920+i,'success':int(info['success']),'recovery_success':int(info['success'] and condition in ('drop_regrasp','gripper_reopen')),'failed':int(info['failed']),'steps':info['t'],'checkpoint_sha256':got})
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'rows':len(rows),'checkpoint_count':len({r['checkpoint_sha256'] for r in rows}),'cuda_used':dev.type=='cuda'},indent=2))
if __name__=='__main__':main()
