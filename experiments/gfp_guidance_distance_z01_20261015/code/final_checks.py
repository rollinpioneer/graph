from pathlib import Path
import json,hashlib,subprocess,sys,importlib.metadata as md,numpy as np
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
checks=[]
for row in json.loads((R/'receipts/checkpoints_before.json').read_text()):
 p=Path(row['path']);assert sha(p)==row['sha256'];meta=json.loads(p.with_suffix('.identity.json').read_text());assert meta==row['identity']
 rng=p.with_suffix('.rng.pkl');assert sha(rng)==meta['data_rng_sha256']
 checks.append(dict(seed=row['seed'],checkpoint=row['checkpoint'],path=str(p),checkpoint_unchanged=True,identity_unchanged=True,rng_file_unchanged=True))
for row in json.loads((R/'configs/protocol.json').read_text())['sources']:
 assert sha(Path(row['path']))==row['sha256'];assert sha(R/row['archive'])==row['sha256']
data=json.loads((R/'receipts/data_identity.json').read_text())
for s in ['train','validation']:assert sha(Path(data[s]['path']))==data[s]['sha256']
probe=json.loads((R/'receipts/probe_result.json').read_text());ex=json.loads((R/'receipts/probe_exit.json').read_text());assert ex['exit_code']==0
assert probe['optimizer_updates']==probe['environment_steps']==probe['forbidden_attempts']==0
mem=[]
for line in (R/'receipts/probe_resources.jsonl').read_text().splitlines():
 for a in json.loads(line)['gpu']:mem.append(int(a.split(',')[-1].split()[0]))
version={k:md.version(k) for k in ['jax','jaxlib','flax','optax','numpy','mujoco','ogbench']}
result=dict(status='PASS',checkpoints=checks,source_files_unchanged=True,data_files_unchanged=True,optimizer_updates=0,environment_steps=0,exit_code=0,observed_peak_gpu_memory_MiB=max(mem),sampling_interval_seconds=5,gpu_resident_hours_upper_bound=ex['wall_seconds']/3600,process_wall_seconds=ex['wall_seconds'],inference_loop_including_compile_seconds=probe['seconds'],cpu_process_seconds=probe['cpu_user_seconds']+probe['cpu_system_seconds'],peak_rss_GiB=probe['peak_rss_kb']/1024**2,versions=version,python=sys.version,gpu_name=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip(),primary_state_evaluations=307200,rng_sensitivity_state_evaluations=61440,bootstrap_unit='dataset trajectory, not states nor training seeds')
(R/'receipts/final_integrity.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='checkpoints'})
# Descriptive second-horizon check on the exact same already evaluated states.
out=[]
for seed in [0,1]:
 for cp in [200000,600000,1000000]:
  z=dict(np.load(R/'arrays'/f's{seed}_{cp}.npz'))
  for splitid,split in enumerate(['train','validation']):
   f=dict(np.load(R/'arrays'/f'{split}_samples.npz'));sel=(z['repeat']==0)&(z['split']==splitid);g=z['g'][sel]
   for h in [25,100]:
    pr=f['progress'+str(h)]
    for region,m in [('near',f['bfs']<=3),('far',f['bfs']>=8)]:
     out.append(dict(seed=seed,checkpoint=cp,split=split,horizon=h,region=region,useful_n=int((m&(pr>=1)).sum()),unhelpful_n=int((m&(pr<=0)).sum()),mean_g_useful=float(g[m&(pr>=1)].mean()),mean_g_unhelpful=float(g[m&(pr<=0)].mean())))
(R/'reports/proxy_horizon_sensitivity.json').write_text(json.dumps(out,indent=2))
