from common import *
import subprocess

def command(name,timeout=None,env=None):
 t=time.time();p=subprocess.Popen([sys.executable,str(R/'code'/name)],env=env,stdout=open(R/f'logs/{name}.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
 try:rc=p.wait(timeout=timeout)
 except subprocess.TimeoutExpired:
  p.terminate();p.wait(timeout=10);raise RuntimeError(name+' reached remaining hard resource budget')
 assert rc==0,(name,rc);return time.time()-t

while not (R/'results/evaluation_complete.json').exists():
 p=R/'results/evaluation_status.json'
 if p.exists():
  status=json.load(open(p))
  if status['failures'] and not status['active']:raise RuntimeError('Evaluation technical stop; preserve partial results')
 if (R/'results/evaluation_budget_stop.json').exists():raise RuntimeError('CPU budget stop')
 time.sleep(5)
summary_time=command('p2_summary.py');cfg=json.load(open(R/'frozen_config.json'))
gpu_used=cfg['P0']['GPU_process_seconds_upper']+sum(json.load(open(R/f'results/{f}.json'))['GPU_process_seconds_upper'] for f in ['training_complete','scoring_complete'])
cpu_used=json.load(open(R/'results/evaluation_complete.json'))['cumulative_CPU_worker_seconds_upper']+summary_time
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
slots=sorted([tuple(map(int,x.split(','))) for x in raw.strip().splitlines() if int(x.split(',')[1])>8000 and int(x.split(',')[0])!=1],key=lambda x:(x[2],-x[1]));assert slots
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(slots[0][0]);remaining=min(7200-gpu_used,70*3600-cpu_used)-30;assert remaining>0
speed_time=command('p3_speed.py',timeout=remaining,env=env);speed=json.load(open(R/'results/speed_summary.json'))
actual_gpu_time=speed_time if speed['status']=='PASS' else 0
logical_bytes=int(subprocess.check_output(['du','-sb',str(R)],text=True).split()[0]);disk_bytes=int(subprocess.check_output(['du','-s','-B1',str(R)],text=True).split()[0])
resources=dict(status='PASS',GPU_process_hours_upper=(gpu_used+actual_gpu_time)/3600,CPU_worker_hours_upper=(cpu_used+speed_time)/3600,new_logical_bytes=logical_bytes,new_disk_bytes=disk_bytes,limits=dict(GPU_process_hours=2,CPU_worker_hours=70,new_GiB=15),CPU_preparation_peak=72,GPU_training_peak=6,GPU_scoring_peak=24,CPU_evaluation_peak=72,GPU_timing_peak=1,formal_train_updates=6000,EG_training=0,EG_inference=0,new_training_GT=0,C4=0,remote_push=False,GPU_speed_slot=env['CUDA_VISIBLE_DEVICES'],speed_outer_seconds=speed_time,summary_outer_seconds=summary_time,time=time.time())
assert resources['GPU_process_hours_upper']<=2 and resources['CPU_worker_hours_upper']<=70 and max(logical_bytes,disk_bytes)<=15*1024**3
write(R/'results/resource_report.json',resources);write(R/'results/computation_complete.json',dict(status='PASS',time=time.time(),phases=['P0','P1','P2','P3'],report_pending=True))
print('P1_P3_COMPUTATION_COMPLETE',json.dumps(resources),flush=True)
