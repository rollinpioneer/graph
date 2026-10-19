import subprocess,time,concurrent.futures,resource
from common import *
start=time.time();done=[]
def run(scene):
 receipt=R/f'diagnosis/{scene:04d}/COMPLETE.json'
 if receipt.exists():
  saved=json.loads(receipt.read_text());assert saved['status']=='PASS';return scene
 with (R/f'logs/diagnosis_{scene:04d}.log').open('a') as f:
  subprocess.run([sys.executable,str(R/'code/diagnose.py'),'--scene',str(scene)],stdout=f,stderr=subprocess.STDOUT,check=True)
 return scene
try:
 subprocess.run([sys.executable,str(R/'code/diagnose.py')],check=True)
 scenes=sum([CFG['exploration'][s] for s in SPLITS],[])
 with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
  fs=[pool.submit(run,s) for s in scenes]
  for f in concurrent.futures.as_completed(fs):
   done.append(f.result());write('results/diagnosis_status.json',{'status':'RUNNING','done':done,'wall_seconds':time.time()-start,'max_processes':8})
 if not (R/'C2_reservation.json').exists():subprocess.run([sys.executable,str(R/'code/audit_reservation.py')],check=True)
 else:assert json.loads((R/'C2_reservation.json').read_text())['audit_sha256']==sha256(R/'results/C2_exposure_audit.json')
 write('results/diagnosis_status.json',{'status':'COMPLETE','done':done,'wall_seconds':time.time()-start,'child_cpu_seconds':resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,'max_processes':8})
except Exception as e:
 write('results/diagnosis_status.json',{'status':'TECHNICAL_STOP','error':repr(e),'done':done,'wall_seconds':time.time()-start});raise
