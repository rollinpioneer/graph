from common import *
import subprocess

start=time.time();assert json.load(open(R/'results/preparation.json'))['status']=='PASS';cfg=json.load(open(R/'frozen_config.json'))
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
slots=sorted([tuple(map(int,s.split(','))) for s in raw.strip().splitlines() if int(s.split(',')[1])>8000 and int(s.split(',')[0])!=1],key=lambda v:(v[2],-v[1]))
assert len(slots)>=1;write(R/'results/P1_resource_start.json',dict(time=time.time(),gpu=raw,CPU_memory=subprocess.check_output(['free','-b'],text=True)))
active=[];completed=[]
for i,(family,seed) in enumerate((f,s) for f in ['C','S'] for s in SEEDS):
 slot=slots[i%len(slots)][0];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(slot);log=open(R/f'logs/train_{family}_{seed}.log','w');p=subprocess.Popen([sys.executable,str(R/'code/p1_train.py'),'--family',family,'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active.append(dict(family=family,seed=seed,slot=slot,p=p,log=log,start=time.time()))
while active:
 for v in active[:]:
  rc=v['p'].poll()
  if rc is not None:
   v['log'].close();completed.append(dict(family=v['family'],seed=v['seed'],GPU_slot=v['slot'],pid=v['p'].pid,returncode=rc,seconds=time.time()-v['start']));active.remove(v)
 used=sum(v['seconds'] for v in completed)+sum(time.time()-v['start'] for v in active)+cfg['P0']['GPU_process_seconds_upper']
 if used>=2*3600:
  for v in active:v['p'].terminate()
  raise RuntimeError('GPU cumulative process-hour hard budget reached')
 write(R/'results/training_status.json',dict(stage='TRAINING' if active else 'COMPLETE',active=[dict(family=v['family'],seed=v['seed'],slot=v['slot'],pid=v['p'].pid) for v in active],completed=completed,actual_peak=6,GPU_process_seconds_upper=used))
 if active:time.sleep(3)
assert all(v['returncode']==0 for v in completed),completed
write(R/'results/training_complete.json',dict(status='PASS',heads=6,total_updates=6000,seconds=time.time()-start,GPU_process_seconds_upper=sum(v['seconds'] for v in completed),CPU_worker_seconds_upper=sum(v['seconds'] for v in completed),jobs=completed))
print('SIX_HEADS_COMPLETE',time.time()-start,flush=True)
