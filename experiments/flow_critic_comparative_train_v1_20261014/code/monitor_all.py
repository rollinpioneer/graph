"""Read only registered runs; append lightweight account/resource receipts."""
import os,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
jobs=read(ROOT/'configs/jobs.json') or []
runs={j['job']:dict(status=read(ROOT/'runs'/j['job']/'status.json'),
    progress=read(ROOT/'runs'/j['job']/'training_progress.json'),
    exits={phase:read(ROOT/'runs'/j['job']/f'{phase}_exit.json') for phase in ['T1','T2','T3']},
    native_evals=len(list((ROOT/'runs'/j['job']/'native').glob('*.json'))),
    matched_evals=len(list((ROOT/'runs'/j['job']/'matched_verified').glob('[0-9]*.json')))) for j in jobs}
current=dict(epoch=time.time(),queue=read(ROOT/'receipts/queue.json'),evaluation_queue=read(ROOT/'receipts/evaluation_queue.json'),runs=runs,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),
    own_processes=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid,ppid,etimes,pcpu,pmem,rss,args'],text=True),
    available_memory_kb=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))))
dest=ROOT/'receipts/monitor';dest.mkdir(exist_ok=True)
previous=read(dest/'latest.json')
if previous:(dest/'previous.json').write_text(json.dumps(previous,indent=2))
changed={k:dict(previous=(previous or {}).get('runs',{}).get(k,{}).get('progress'),current=v['progress'])
    for k,v in runs.items() if v['progress']!=(previous or {}).get('runs',{}).get(k,{}).get('progress')}
current['changes']=changed
stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
(dest/f'{stamp}.json').write_text(json.dumps(current,indent=2))
(dest/'latest.json').write_text(json.dumps(current,indent=2))
print(json.dumps(dict(epoch=current['epoch'],queue=current['queue'],progress={k:v['progress'] for k,v in runs.items()},gpu=current['gpu']),indent=2))
