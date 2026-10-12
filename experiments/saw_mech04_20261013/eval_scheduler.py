"""Work guards count evaluation work, not idle waiting for registered checkpoints."""
import concurrent.futures,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PY='/home/xushijie3/work/graph_saw_rd01_20261012/experiments/saw_rd01_20261012/vendor/venv/bin/python'
def main():
    jobs=[]
    for k,v in [('navigate','A'),('stitch','A'),('stitch','B')]:jobs.append([k,v,0,1000000,'full'])
    for k,v in [('navigate','A'),('stitch','A'),('stitch','B')]:
        for seed in (1,2):jobs.append([k,v,seed,1000000,'full'])
    for seed in (1,2):
        for step in (100000,300000,500000):jobs.append(['navigate','A',seed,step,'monitor'])
    pending=jobs.copy();running={};finished=[];available=['80-83','84-87','88-91']
    def write(name,x):
        p=ROOT/'receipts'/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
    write('EVALUATION_SOURCE.json',dict(files={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['eval_frozen.py','eval_scheduler.py','frozen.py','state_core.py']}))
    while pending or running:
        for label,(p,job,cores,log) in list(running.items()):
            if p.poll() is not None:
                rc=p.wait();log.close();finished.append(dict(label=label,job=job,exit_code=rc));available.append(cores);del running[label]
                print('EVAL_EXIT',label,rc,flush=True)
                if rc!=0:
                    write('EVALUATION_SCHEDULER_FAILURE.json',dict(label=label,exit_code=rc,note='No automatic retry; remaining independent jobs may proceed within unchanged package cap'))
        for job in pending.copy():
            if not available:break
            k,v,seed,step,scope=job;label=f'Ceval_{k}_{v}_s{seed}_{step}_{scope}'
            if seed:
                cps=ROOT/'results'/f'C_{k}_{v}_s{seed}_checkpoints.json'
                ready=cps.exists() and any(c['step']==step for c in json.loads(cps.read_text()))
                runexit=ROOT/'receipts'/f'C_{k}_{v}_s{seed}_exit.json'
                resource=ROOT/'receipts'/f'C_{k}_{v}_s{seed}_resource.json'
                if step==1000000:ready=ready and runexit.exists() and resource.exists() and json.loads(runexit.read_text())['status']=='COMPLETE_1M' and json.loads(resource.read_text())['exit_code']==0
                if not ready:
                    if resource.exists():
                        finished.append(dict(label=label,job=job,status='NOT_EXECUTED_MISSING_VALID_CHECKPOINT'));pending.remove(job)
                    continue
            cores=available.pop(0);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
            log=open(ROOT/'logs'/f'{label}_guard.log','x');cmd=[PY,str(ROOT/'guard.py'),'--label',label,'--group','C-EVAL','--seconds','2400','--','taskset','-c',cores,PY,str(ROOT/'eval_frozen.py'),'--kind',k,'--variant',v,'--seed',str(seed),'--step',str(step),'--scope',scope]
            p=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);running[label]=(p,job,cores,log);pending.remove(job)
        write('EVALUATION_SCHEDULER_PROGRESS.json',dict(pending=pending,running=list(running),finished=finished))
        if pending and not running and (ROOT/'receipts/SCHEDULER_EXIT.json').exists():
            for job in pending:finished.append(dict(job=job,status='NOT_EXECUTED_MISSING_VALID_CHECKPOINT'))
            pending=[]
        if pending or running:time.sleep(5)
    write('EVALUATION_SCHEDULER_EXIT.json',dict(finished=finished));print('ALL_EVALUATION_JOBS_ACCOUNTED',flush=True)
if __name__=='__main__':main()
