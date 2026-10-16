import os,json,time,subprocess
from pathlib import Path
from common import ROOT,write

def main():
    q=json.loads((ROOT/'receipts/queue.json').read_text());procs={}
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    for line in raw.splitlines():
        try:p,m=map(int,line.split(','));procs[p]=m
        except ValueError:pass
    rows=[]
    for job in q['active']:
        d=ROOT/'runs'/f"s{job['seed']}_{job['arm']}";r={k:job[k] for k in ('id','kind','pid','gpu')}
        r['gpu_memory_mib']=procs.get(job['pid'])
        stat=Path(f"/proc/{job['pid']}/status")
        r['alive']=stat.exists()
        if stat.exists():r['memory']=[l for l in stat.read_text().splitlines() if l.startswith(('VmRSS:','VmHWM:','Threads:'))]
        p=d/'progress.json'
        if p.exists():r['progress']=json.loads(p.read_text())
        rows.append(r)
    result=dict(time=time.time(),active=rows,pending=q['pending'],eval_pending=q['evaluation_pending'],failed=q['failed'],done=q['done'],gpus=q['gpus'])
    write(ROOT/'receipts/monitor_latest.json',result)
    with open(ROOT/'logs/resource_samples.jsonl','a') as f:f.write(json.dumps(result)+'\n')
    return result

if __name__=='__main__':
    import sys
    if '--watch' in sys.argv:
        while True:
            main()
            if (ROOT/'receipts/manager_complete.json').exists():break
            pid=json.loads((ROOT/'receipts/manager_started.json').read_text())['pid']
            if not Path(f'/proc/{pid}').exists():break
            time.sleep(30)
    else:print(json.dumps(main(),indent=2))
