from common import *
import subprocess

cfg=json.load(open(R/'frozen_config.json'));profile=json.load(open(R/'results/eval_0100_profile.json'));assert profile['status']=='PASS' and len(profile['native_checks'])==4
initial=cfg['P0']['CPU_worker_seconds_upper']+sum(json.load(open(R/f'results/{f}.json'))['CPU_worker_seconds_upper'] for f in ['preparation','training_complete','scoring_complete'])+profile['seconds']
start=time.time();tasks=[(s,k) for s in sum(cfg['evaluation']['C3'].values(),[]) for k in range(6)];active=[];done=[];failed=[];peak=0
raw=subprocess.check_output(['free','-b'],text=True);write(R/'results/P2_CPU_resource_start.json',dict(time=time.time(),memory=raw))
assert int(raw.splitlines()[1].split()[-1])>=64*1024**3
for scene,shard in tasks:
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';log=open(R/f'logs/eval_{scene:04d}_{shard}.log','w');p=subprocess.Popen([sys.executable,str(R/'code/p2_eval.py'),'--scene',str(scene),'--shard',str(shard)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active.append(dict(scene=scene,shard=shard,p=p,log=log,start=time.time()))
peak=len(active)
while active:
 for v in active[:]:
  rc=v['p'].poll()
  if rc is not None:
   v['log'].close();receipt=dict(scene=v['scene'],shard=v['shard'],pid=v['p'].pid,returncode=rc,seconds=time.time()-v['start']);done.append(receipt);active.remove(v)
   if rc:failed.append(receipt)
 used=initial+sum(v['seconds'] for v in done)+sum(time.time()-v['start'] for v in active)
 if used>=70*3600-72*15:
  for v in active:v['p'].terminate()
  failed.append(dict(reason='CPU worker hard budget reserve reached'));write(R/'results/evaluation_budget_stop.json',dict(used=used,failures=failed));raise RuntimeError('CPU budget')
 write(R/'results/evaluation_status.json',dict(stage='EVALUATING' if active else 'COMPLETE',actual_peak=peak,active=[dict(scene=v['scene'],shard=v['shard'],pid=v['p'].pid) for v in active],completed=done,failures=failed,cumulative_CPU_worker_seconds_upper=used,seconds=time.time()-start))
 if active:time.sleep(5)
assert not failed,failed
write(R/'results/evaluation_complete.json',dict(status='PASS',frames=3072,configurations=24,records=73728,seconds=time.time()-start,actual_peak=peak,CPU_worker_seconds_upper=sum(v['seconds'] for v in done),cumulative_CPU_worker_seconds_upper=used,jobs=done))
print('FULL_C3_EVALUATION_COMPLETE',time.time()-start,flush=True)
