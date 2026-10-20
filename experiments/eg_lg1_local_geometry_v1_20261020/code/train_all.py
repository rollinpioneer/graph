from common import *
import subprocess
assert json.load(open(R/'results/preparation.json'))['status']=='PASS'
t=time.time();running=[];peak=0
for seed in SEEDS:
 for family in ['G','Gshuffle']:
  gpu='6' if seed==20261019 else '0';env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
  log=open(R/f'logs/train_{family}_{seed}.log','w');p=subprocess.Popen([sys.executable,str(R/'code/train_head.py'),'--seed',str(seed),'--family',family],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  running.append(dict(p=p,seed=seed,family=family,gpu=gpu,log=log,start=time.time()))
while any(v['p'].poll() is None for v in running):
 active=[v for v in running if v['p'].poll() is None];peak=max(peak,len(active));write(R/'results/training_status.json',dict(active=[{k:v[k] for k in ['seed','family','gpu']} for v in active],peak_running_processes_observed=peak,seconds=time.time()-t));time.sleep(.5)
for v in running:v['log'].close()
assert all(v['p'].returncode==0 for v in running),[(v['family'],v['seed'],v['p'].returncode) for v in running]
heads=[json.load(open(R/f"results/train_{v['family']}_{v['seed']}.json")) for v in running]
for s in SEEDS:
 vv=[v for v in heads if v['seed']==s];assert vv[0]['initial_state_sha256']==vv[1]['initial_state_sha256'] and vv[0]['sample_stream_sha256']==vv[1]['sample_stream_sha256']
assert sum(v['seconds'] for v in heads)<.15*3600
lock=dict(frozen_config_sha256=sha(R/'frozen_config.json'),geometry_core_sha256=sha(R/'code/geometry_core.py'),new_heads=[dict(family=v['family'],seed=v['seed'],path=v['checkpoint'],sha256=v['checkpoint_sha256']) for v in heads],old_O_heads=json.load(open(P/'C3_method_lock.json'))['heads']['B'],F05=.5,CD=.01,C4_opened=False)
write(R/'method_lock.json',lock);write(R/'results/training_complete.json',dict(status='PASS',heads=heads,training_updates=6000,additional_backbone_updates=0,new_training_quality_labels=0,peak_running_processes_observed=peak,wall_seconds=time.time()-t,GPU_process_seconds=sum(v['seconds'] for v in heads)))
print('SIX_HEADS_COMPLETE',time.time()-t,flush=True)
