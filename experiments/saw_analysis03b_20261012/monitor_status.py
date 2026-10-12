"""Compact status from own PID registry; read-only, no model import."""
import json,time,datetime,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent;r=root/'receipts';out=[]
for kind in ('navigate','stitch'):
    p=r/f'{kind}_progress.json';last=json.loads(p.read_text())['step'] if p.exists() else 0
    status=root/'results'/f'{kind}_training_status.json'
    if status.exists():x=json.loads(status.read_text());last=x['step'];state=x['status']
    else:state='RUNNING'
    short={'SINGLE_SEED_B_COMPARISON':'DONE','BUDGET_STOP':'STOP','FAILED':'FAIL'}.get(state,state)
    out.append(f'{kind[0].upper()}={last//1000}k' + ('' if state=='RUNNING' else ':'+short))
live=[json.loads(f.read_text()) for f in r.glob('*_active.json')];done=[json.loads(f.read_text()) for f in r.glob('*_resource.json')]
hours=(sum(x['gpu_residency_upper_seconds'] for x in done)+sum(time.time()-x['start'] for x in live))/3600
pids={x['pid'] for x in live};mem=0
if pids:
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()
    mem=sum(int(x.split(',')[1]) for x in lines if int(x.split(',')[0]) in pids)
print(datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%SZ'),' '.join(out),f'GPU={hours:.3f}h MEM={mem}MiB',flush=True)
print('GUARDS_FINISHED' if not live and len(done)>=4 else 'GUARDS_ACTIVE',flush=True)
