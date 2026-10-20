from common import *
import subprocess
t=time.time()
while not (R/'results/evaluation_complete.json').exists():
 if (R/'results/pipeline_status.json').exists():assert json.load(open(R/'results/pipeline_status.json'))['stage']!='TECHNICAL_STOP'
 assert time.time()-t<7200,'C4 finish wait exceeded'
 time.sleep(10)
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
with open(R/'logs/novel_official_check.log','w') as f:subprocess.run([sys.executable,str(R/'code/evaluate.py'),'--scene','173','--profile'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(R/'code/report_c4.py')],env=env,check=True)
print('C4_RESULT_FROZEN_READY_FOR_ACCELERATION',flush=True)
