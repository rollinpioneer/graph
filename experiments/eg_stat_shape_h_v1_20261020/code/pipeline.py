from common import *
import subprocess,traceback
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
def stage(name,args=None,gpu=False):
 env=os.environ.copy()
 if gpu:
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  slots=sorted([tuple(map(int,x.split(','))) for x in raw.strip().splitlines() if int(x.split(',')[1])>8000 and int(x.split(',')[0])!=1],key=lambda x:(x[2],-x[1]))
  assert slots;env['CUDA_VISIBLE_DEVICES']=str(slots[0][0])
 else:env['CUDA_VISIBLE_DEVICES']=''
 write(R/'results/pipeline_status.json',dict(stage=name,status='RUNNING',time=time.time(),controller_pid=os.getpid()))
 with (R/f'logs/{name}.log').open('w') as log:
  p=subprocess.Popen([sys.executable,str(R/'code'/name),*(args or [])],env=env,stdout=log,stderr=subprocess.STDOUT)
  rc=p.wait()
 assert rc==0,(name,rc)
 print('STAGE_DONE',name,flush=True)
try:
 stage('p0_validate.py',gpu=True)
 stage('p1_prepare.py')
 stage('train_dispatch.py')
 stage('p2_score.py',['--scene','100','--shard','0','--shards','256'],gpu=True)
 stage('p2_eval.py',['--scene','100','--profile'])
 # One validated score frame is reused by production scoring; never overwrite it.
 stage('score_dispatch.py')
 stage('eval_dispatch.py')
 stage('p2_summary.py')
 stage('p3_speed.py',gpu=True)
 write(R/'results/pipeline_status.json',dict(status='COMPUTATION_COMPLETE_REPORT_PENDING',time=time.time(),controller_pid=os.getpid()))
except Exception as e:
 write(R/'results/pipeline_status.json',dict(status='TECHNICAL_STOP',error=repr(e),traceback=traceback.format_exc(),time=time.time(),controller_pid=os.getpid()))
 raise
