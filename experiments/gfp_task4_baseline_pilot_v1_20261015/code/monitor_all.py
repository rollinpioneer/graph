import os,json,time,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];out={'time':time.time(),'runs':{},'controllers':{}}
for name in ['GFP_seed0','GFP_seed1','ReBRAC_seed0','ReBRAC_seed1']:
 d=R/'runs'/name;v={}
 for f in ['status.json','training_progress.json','dispatch_admission.json','exit.json','matched/progress.json','matched/completed.json','matched_exit.json']:
  p=d/f
  if p.exists():v[f]=json.loads(p.read_text())
 out['runs'][name]=v
for name in ['evaluation_queue.json','evaluation_finished.json','z2_progress.json','z2_completed.json']:
 p=R/'receipts'/name
 if p.exists():out['controllers'][name]=json.loads(p.read_text())
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
out['ram']=subprocess.check_output(['free','-m'],text=True)
out['own_processes']=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid,ppid,etimes,%cpu,rss,args'],text=True)
d=R/'receipts/monitor';d.mkdir(exist_ok=True);p=d/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.json');p.write_text(json.dumps(out,indent=2));(d/'latest.json').write_text(p.read_text());print(json.dumps({k:{n:z.get(n) for n in ['status.json','training_progress.json','matched/progress.json','matched_exit.json']} for k,z in out['runs'].items()}))
