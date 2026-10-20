from common import *
import subprocess

cfg=json.load(open(R/'frozen_config.json'));prior=cfg['P0']['GPU_process_seconds_upper']+json.load(open(R/'results/training_complete.json'))['GPU_process_seconds_upper'];start=time.time()
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
slots=sorted([tuple(map(int,s.split(','))) for s in raw.strip().splitlines() if int(s.split(',')[1])>=8000 and int(s.split(',')[0])!=1],key=lambda v:(v[2],-v[1]));assert slots
tasks=[(s,k) for s in sum(cfg['evaluation']['C3'].values(),[]) for k in range(2)];active=[];done=[];failed=[];peak=0
write(R/'results/P2_GPU_resource_start.json',dict(time=time.time(),gpu=raw))
for i,(scene,shard) in enumerate(tasks):
 slot=slots[i%len(slots)][0];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(slot);log=open(R/f'logs/score_{scene:04d}_{shard}.log','w');p=subprocess.Popen([sys.executable,str(R/'code/p2_score.py'),'--scene',str(scene),'--shard',str(shard)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active.append(dict(scene=scene,shard=shard,slot=slot,p=p,log=log,start=time.time()))
peak=len(active)
while active:
 for v in active[:]:
  rc=v['p'].poll()
  if rc is not None:
   v['log'].close();receipt=dict(scene=v['scene'],shard=v['shard'],GPU_slot=v['slot'],pid=v['p'].pid,returncode=rc,seconds=time.time()-v['start']);done.append(receipt);active.remove(v)
   if rc:failed.append(receipt)
 used=prior+sum(v['seconds'] for v in done)+sum(time.time()-v['start'] for v in active)
 if used>=2*3600-120:
  for v in active:v['p'].terminate()
  failed.append(dict(reason='GPU budget stop with120seconds reserve'));write(R/'results/scoring_budget_stop.json',dict(used=used,failures=failed));raise RuntimeError('GPU budget')
 write(R/'results/scoring_status.json',dict(stage='SCORING' if active else 'COMPLETE',actual_peak=peak,active=[dict(scene=v['scene'],shard=v['shard'],slot=v['slot'],pid=v['p'].pid) for v in active],completed=done,failures=failed,cumulative_GPU_process_seconds_upper=used))
 if active:time.sleep(3)
assert not failed,failed
write(R/'results/scoring_complete.json',dict(status='PASS',frames=3072,heads=6,seconds=time.time()-start,actual_peak=peak,GPU_process_seconds_upper=sum(v['seconds'] for v in done),CPU_worker_seconds_upper=sum(v['seconds'] for v in done),jobs=done))
print('FULL_C3_SCORES_COMPLETE',time.time()-start,flush=True)
