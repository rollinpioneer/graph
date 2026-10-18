from common_v2 import *
import subprocess,concurrent.futures
start=time.time();micro=R/'evaluation/development/0037/MICRO.json';assert micro.exists() and json.loads(micro.read_text())['status']=='PASS';projected=json.loads(micro.read_text())['seconds']*256/3600+.5;assert projected<40
write(R/'results/application_projection.json',dict(conservative_worker_wall_hours=projected,budget=40,parallel_evaluation_workers=7,controller=1,max_all_compute_processes=8))
write(R/'results/pipeline_status.json',dict(status='RUNNING',stage='development',new_trainings=0))
def one(s):
 t=time.time();f=open(R/f'logs/development_{s}.log','w');p=subprocess.run([sys.executable,str(R/'code/evaluate_v2.py'),'--scene',str(s)],stdout=f,stderr=subprocess.STDOUT);f.close();assert p.returncode==0,(s,p.returncode)
 return dict(scene=s,launcher_wall_seconds=time.time()-t)
try:
 done=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
  futures=[pool.submit(one,s) for s in CFG['development']]
  for job in concurrent.futures.as_completed(futures):
   done.append(job.result());write(R/'results/pipeline_status.json',dict(status='RUNNING',stage='development',done=done));assert sum(x['launcher_wall_seconds'] for x in done)<40*3600
 subprocess.run([sys.executable,str(R/'code/summarize_v2.py')],check=True);d=json.loads((R/'results/development_decision.json').read_text())
 write(R/'results/pipeline_status.json',dict(status='READY_FOR_LOCKED_TRANSFER' if d['pass_stage'] else 'COMPLETED_AT_DEVELOPMENT_STOP',reason='Development gates passed' if d['pass_stage'] else 'Preregistered development gate failed',stage='development',done=done,seconds=time.time()-start,child_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,new_trainings=0,confirmation='NOT_OPENED_REQUIRES_SEPARATE_APPROVAL'))
except Exception as e:
 write(R/'results/pipeline_status.json',dict(status='TECHNICAL_STOP',stage='development',error=repr(e)));raise
