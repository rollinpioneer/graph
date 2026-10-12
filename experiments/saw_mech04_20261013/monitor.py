"""Compact progress, source identity and actual checkpoint initialization pairing."""
import json,pickle,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def equal(a,b):
 if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return np.array_equal(np.asarray(a),np.asarray(b))
def main():
 r=ROOT/'receipts';parts=[]
 resources=[json.loads(p.read_text()) for p in r.glob('C_*_resource.json') if json.loads(p.read_text())['GPU']]
 live=[]
 for p in r.glob('C_*_active.json'):
  try:x=json.loads(p.read_text())
  except FileNotFoundError:continue
  if x['GPU']:live.append(x)
 cost=sum(x['residency_seconds'] for x in resources)+sum(time.time()-x['start'] for x in live)
 for kind,v,seed in [('stitch',v,s) for s in (1,2) for v in ('A','B')]+[('navigate','A',s) for s in (1,2)]:
  label=f'C_{kind}_{v}_s{seed}';p=ROOT/'results'/f'{label}_training_status.json';q=r/f'{label}_progress.json'
  if p.exists():x=json.loads(p.read_text());s=f"{x['step']//1000}k:{x['status']}"
  elif q.exists():s=f"{json.loads(q.read_text())['step']//1000}k"
  else:s='QUEUED/INIT'
  parts.append(f'{kind[0].upper()}{v}{seed}={s}')
 print(time.strftime('%H:%M:%SZ',time.gmtime()),'GPUh',round(cost/3600,3))
 for a in parts:print(a)
 for seed in (1,2):
  dest=ROOT/'results'/f'GPU_UPDATE0_s{seed}.json'
  paths=[ROOT/'runs'/f'C_stitch_{v}_s{seed}'/'params_0.pkl' for v in ('A','B')]
  if not dest.exists() and all(p.exists() for p in paths):
   states=[pickle.loads(p.read_bytes()) for p in paths];same=equal(states[0]['agent'],states[1]['agent'])
   dest.write_text(json.dumps(dict(seed=seed,agent_optimizer_rng_equal=same,step=0,paths=[str(p) for p in paths],sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]),indent=2)+'\n')
   print('GPU_UPDATE0_PAIR',seed,same)
 print('TRAINING_ALL_EXITED' if (r/'SCHEDULER_EXIT.json').exists() else 'TRAINING_ACTIVE')
if __name__=='__main__':main()
