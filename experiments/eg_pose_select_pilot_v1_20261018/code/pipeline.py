from core import *
import subprocess,concurrent.futures
start=time.time();stage='waiting_labels'
def run(script,args=(),gpu=False):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 with open(R/f'logs/{script}_{"_".join(map(str,args)).replace("/","-")}.log','a') as f:p=subprocess.run([sys.executable,str(R/'code'/script),*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,(script,args,p.returncode)
def budget():
 wall=sum(json.loads(p.read_text())['seconds'] for p in (R/'results').glob('labels_*.json'))+sum(json.loads(p.read_text())['seconds'] for p in (R/'evaluation').rglob('COMPLETE.json'))
 size=sum(p.stat().st_size for sub in ['labels','heads','results','evaluation','logs'] for p in (R/sub).rglob('*') if p.is_file());assert wall/3600<40 and size<CFG['new_disk_bytes_limit'],('EXPLORATION_BUDGET',wall,size)
 return dict(worker_wall_hours_conservative=wall/3600,disk_bytes=size)
def evaluate(stage,scenes):
 def one(s):run('evaluate.py',['--stage',stage,'--scene',s]);budget();return s
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for s in pool.map(one,scenes):write(R/'results/pipeline_status.json',dict(status='RUNNING',stage=stage,last_scene=s,resources=budget()))
 run('summarize.py',['--stage',stage]);return json.loads((R/f'results/{stage}_decision.json').read_text())
try:
 while True:
  p=R/'results/label_queue.json'
  if p.exists():
   q=json.loads(p.read_text())
   if q['status']=='PASS':break
   assert all(x['exit']==0 for x in q['done']),q
  time.sleep(15)
 stage='training';budget();write(R/'results/pipeline_status.json',dict(status='RUNNING',stage=stage));run('fit_gate.py',gpu=True)
 stage='application_micro';run('evaluate.py',['--scene',CFG['development'][0],'--micro'])
 micro=json.loads((R/f'evaluation/development/{CFG["development"][0]:04d}/MICRO.json').read_text());resources=budget();projection=resources['worker_wall_hours_conservative']+micro['seconds']*256/3600
 write(R/'results/application_projection.json',dict(micro_seconds=micro['seconds'],projected_development_worker_hours=projection,limit=40));assert projection<40,('AP_BUDGET_PROJECTION',projection)
 stage='development';d=evaluate(stage,CFG['development']);gate=json.loads((R/'results/gate_calibration.json').read_text())
 if gate['tau'] is None or not d['pass_stage']:
  write(R/'results/pipeline_status.json',dict(status='COMPLETED_AT_S1_STOP',stage=stage,reason='No finite usable threshold' if gate['tau'] is None else 'Predeclared main development gate failed',transfer='NOT_RUN_UPSTREAM_STOP',confirmation='NOT_OPENED',seconds=time.time()-start,resources=budget()))
 else:
  stage='transfer';t=evaluate(stage,sum(CFG['transfer_development'].values(),[]));write(R/'results/pipeline_status.json',dict(status='READY_FOR_LOCKED_CONFIRMATION' if t['pass_stage'] else 'COMPLETED_AT_S1_STOP',stage=stage,reason='Transfer gate passed' if t['pass_stage'] else 'Transfer gate failed',confirmation='NOT_OPENED',seconds=time.time()-start,resources=budget()))
except Exception as e:
 write(R/'results/pipeline_status.json',dict(status='TECHNICAL_STOP',stage=stage,error=repr(e),seconds=time.time()-start));raise
