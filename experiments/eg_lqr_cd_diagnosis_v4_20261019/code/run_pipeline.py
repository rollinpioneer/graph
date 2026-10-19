"""Bounded conditional scheduler. No scheduled monitoring automation."""
import time,subprocess,resource,datetime,shlex
from common import *
start=time.time();workers=[];completed=[];active={};export=None;phase='waiting_diagnosis';gpu=None
def status(state,**extra):
 write('results/pipeline_status.json',dict(status=state,phase=phase,wall_seconds=time.time()-start,completed=completed,active_scenes=sorted(active),export_active=export is not None and export.poll() is None,max_processes=8,new_trainings=0,**extra))
def diag_worker_seconds():
 return sum(json.loads(p.read_text())['seconds'] for p in (R/'diagnosis').glob('*/COMPLETE.json'))
def consumed():
 return diag_worker_seconds()+sum(w['seconds'] for w in workers)+sum(time.time()-v['start'] for v in active.values())
def launch(scene,ph):
 log=(R/f'logs/{ph}_{scene:04d}.log').open('a');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
 proc=subprocess.Popen([sys.executable,str(R/'code/evaluate.py'),'--phase',ph,'--scene',str(scene)],stdout=log,stderr=subprocess.STDOUT,env=env)
 active[scene]=dict(process=proc,start=time.time(),log=log)
def collect(ph):
 for scene,v in list(active.items()):
  rc=v['process'].poll()
  if rc is None:continue
  v['log'].close();assert rc==0,(ph,scene,rc)
  receipt=json.loads((R/f'evaluation/{ph}/{scene:04d}/COMPLETE.json').read_text());assert receipt['status']=='PASS';workers.append(dict(phase=ph,scene=scene,seconds=receipt['seconds'],cpu_seconds=receipt['cpu_seconds']));completed.append(scene);del active[scene];status('RUNNING')
try:
 while True:
  p=R/'results/diagnosis_status.json'
  if p.exists():
   state=json.loads(p.read_text())
   if state['status']=='COMPLETE':break
   if state['status']=='TECHNICAL_STOP':raise RuntimeError('Diagnosis needs same-config repair before proceeding')
  status('WAITING_DIAGNOSIS');time.sleep(5)
 subprocess.run([sys.executable,str(R/'code/summarize_diagnosis.py')],check=True)
 if not json.loads((R/'results/diagnosis_gate.json').read_text())['execute_fixed_F05']:
  status('STOP_DIAGNOSIS_DID_NOT_SUPPORT_FIXED_F05');sys.exit(0)
 phase='exploration';pending=sum([CFG['exploration'][s] for s in SPLITS],[]);completed=[]
 while pending or active:
  assert consumed()<CFG['budget']['worker_wall_hours']*3600,'CPU worker-wall budget'
  collect(phase)
  while pending and len(active)<7:launch(pending.pop(0),phase)
  status('RUNNING');time.sleep(3)
 subprocess.run([sys.executable,str(R/'code/report_phase.py'),'--phase',phase],check=True)
 gate=json.loads((R/'results/exploration_decision.json').read_text());reservation=json.loads((R/'C2_reservation.json').read_text())
 if gate['status']!='PASS_FOR_C2':status('COMPLETE_STOP_F05_EXPLORATORY_GATE_FAILED');sys.exit(0)
 if reservation['status']!='PASS_NO_KNOWN_EXPOSURE_AND_RAW_FILES_PRESENT':status('PARTIAL_BLOCKED_INSUFFICIENT_UNEXPOSED_SCENES');sys.exit(0)
 assert consumed()*2+1800<CFG['budget']['worker_wall_hours']*3600,'Conservative same-scale C2 budget forecast'
 lock=dict(family='F05',formula=CFG['F05'],heads=CFG['heads'],config_sha256=sha256(R/'frozen_config.json'),reservation_sha256=sha256(R/'C2_reservation.json'),audit_sha256=reservation['audit_sha256'],selected=reservation['selected'],exploration_decision_sha256=sha256(R/'results/exploration_decision.json'),C2_performance_opened=False,locked_at=datetime.datetime.now().astimezone().isoformat(),ensemble=False)
 assert not (R/'C2_method_lock.json').exists();write('C2_method_lock.json',lock)
 scan=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);choices=[]
 for row in scan.splitlines():
  i,uuid,used,free,util=[s.strip() for s in row.split(',')]
  if float(free)>=8192:choices.append((float(util),float(used),int(i),uuid))
 assert choices,'No GPU with sufficient currently free memory';chosen=min(choices);gpu=chosen[2]
 write('results/C2_resource_launch.json',dict(gpu=gpu,uuid=chosen[3],snapshot=scan,CPU_threads_per_worker=1,max_processes=8,parallelism='GPU1+CPU6+controller1; after export CPU7+controller1'))
 phase='confirmation';pending=sum([lock['selected'][s] for s in SPLITS],[]);completed=[];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);elog=(R/'logs/C2_export.log').open('a');export=subprocess.Popen([sys.executable,str(R/'code/export_confirmation.py')],stdout=elog,stderr=subprocess.STDOUT,env=env);export_start=time.time()
 while pending or active or export.poll() is None:
  assert consumed()+time.time()-export_start<CFG['budget']['worker_wall_hours']*3600,'Worker-wall budget with export'
  collect(phase);rc=export.poll();assert rc is None or rc==0,('export failed',rc)
  capacity=6 if rc is None else 7
  ready=[s for s in pending if (R/f'exports/subset/scene_{s:04d}/COMPLETE.json').exists()]
  for scene in ready[:max(0,capacity-len(active))]:pending.remove(scene);launch(scene,phase)
  status('RUNNING',gpu=gpu);time.sleep(3)
 elog.close();subprocess.run([sys.executable,str(R/'code/report_phase.py'),'--phase',phase],check=True)
 phase='complete';status('COMPLETE',worker_wall_seconds=consumed(),export_wall_seconds=json.loads((R/'results/C2_export_status.json').read_text())['wall_seconds'],child_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,workers=workers)
except Exception as e:
 status('TECHNICAL_STOP',error=repr(e),workers=workers);raise
