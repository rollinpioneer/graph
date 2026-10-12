"""Frozen four-scene normal sampling review; two concurrent independent workers."""
import os,pathlib,json,subprocess,time,hashlib,concurrent.futures
R=pathlib.Path(__file__).resolve().parent;py=os.environ['PY']
freeze=json.loads((R/'receipts/normal_freeze.json').read_text())
for name,h in freeze['source_sha256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
def worker(plans):
 for p in plans:
  name=p['name'];cmd=['python3',str(R/'guard.py'),'--out',str(R/'runs'/name),'--seconds',str(freeze['max_gpu_seconds_each']),'--gpu',str(p['gpu']),'--',py,'-u',str(R/'rollout.py'),'--out',str(R/'runs'/name/'data'),'--n',str(p['n']),'--scene-seed',str(p['scene_seed']),'--policy-seed',str(p['policy_seed']),'--role','independent_normal_review','--arm','normal']
  print('LAUNCH',name,flush=True)
  with (R/'logs'/(name+'_guard.log')).open('w') as f:res=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  print('FINISH',name,res.returncode,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 results=list(ex.map(worker,[[p for p in freeze['plans'] if p['gpu']==g] for g in freeze['gpus']]))
(R/'receipts/normal_launcher_complete.json').write_text(json.dumps({'completed':True,'conditions_not_replaced_after_outcomes':True}))
print('NORMAL_SEQUENCE_COMPLETE',flush=True)
