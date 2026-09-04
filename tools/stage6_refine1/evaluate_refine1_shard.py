#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from tools.stage6.policy_model import ChunkPolicy
from tools.stage6.policy_env import EpisodeSpec,TransportGraphEnv
from tools.stage6_refine1.common import sha256
def main():
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--condition',required=True);p.add_argument('--method',required=True);p.add_argument('--policy-seed',type=int,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--seed-registry',type=Path,required=True);p.add_argument('--rollout-start',type=int,required=True);p.add_argument('--rollout-count',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();assert sha256(a.checkpoint)==a.checkpoint_sha256
 regs={(r['task_id'],r['condition'],int(r['rollout_index'])):int(r['env_seed']) for r in csv.DictReader(a.seed_registry.open())};z=torch.load(a.checkpoint,map_location='cpu',weights_only=False);dev=torch.device(a.device if torch.cuda.is_available() else 'cpu');m=ChunkPolicy().to(dev);m.load_state_dict(z['model']);m.eval();mean=np.asarray(z['normalizer_mean']);std=np.asarray(z['normalizer_std']);rows=[]
 for rid in range(a.rollout_start,a.rollout_start+a.rollout_count):
  env_seed=regs[(a.task,a.condition,rid)];spec=EpisodeSpec(a.task,a.condition,a.condition if a.task=='transport_dual_order' else '',env_seed);env=TransportGraphEnv(spec);obs=env.reset();steps=0;last={}
  while not env.done:
   with torch.no_grad():act=m(torch.tensor(((obs-mean)/std)[None],dtype=torch.float32,device=dev))[0,0].cpu().numpy()
   obs,_,_,last=env.step(act);steps+=1
  rows.append({'task_id':a.task,'method':a.method,'policy_seed':a.policy_seed,'condition':a.condition,'rollout_index':rid,'paired_eval_seed':env_seed,'success':int(last['success']),'recovery_success':int(last['success'] and a.condition in ('drop_regrasp','gripper_reopen')),'failed':int(last['failed']),'steps':steps,'checkpoint_sha256':a.checkpoint_sha256,'cuda_used':dev.type=='cuda'})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'rows':len(rows),'cuda_used':dev.type=='cuda'},indent=2))
if __name__=='__main__':main()
