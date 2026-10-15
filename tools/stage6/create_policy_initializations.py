#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch,yaml
from tools.stage6.policy_model import ChunkPolicy
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text()); rows=[]
 for task in c['tasks']:
  for seed in c['policy']['policy_seeds']:
   torch.manual_seed(int(seed)); m=ChunkPolicy(c['policy']['obs_dim'],c['policy']['action_dim'],c['policy']['action_chunk_size'],c['policy']['hidden_dim']); d=a.out/task/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True);f=d/'init.pt';torch.save({'model':m.state_dict(),'seed':int(seed),'task_id':task},f); rows.append({'task_id':task,'policy_seed':seed,'path':str(f),'sha256':sha(f),'loadable':True})
 a.out.mkdir(parents=True,exist_ok=True);(a.out/'initialization_manifest.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
