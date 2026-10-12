"""Separately authorized T training, two queues; no retries or early score decisions."""
import concurrent.futures,json,os,subprocess,hashlib,time,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PY='/home/xushijie3/work/graph_saw_rd01_20261012/experiments/saw_rd01_20261012/vendor/venv/bin/python'
def queue(gpu,cores,arms):
    out=[]
    for arm in arms:
        label=f'T{arm}';env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda,cpu');cmd=[PY,str(ROOT/'guard.py'),'--label',label,'--group','T','--seconds','2250','--','taskset','-c',cores,PY,str(ROOT/'t_train.py'),'--arm',str(arm)]
        with open(ROOT/'logs'/f'{label}_guard.log','x') as f:rc=subprocess.call(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        resource=json.loads((ROOT/'receipts'/f'{label}_resource.json').read_text());out.append(dict(arm=arm,exit_code=rc,GPU=gpu));print('T_ARM_EXIT',arm,rc,flush=True)
        reason=resource['stop_reason'] or ''
        if reason in ('AGGREGATE_VRAM_LIMIT','DISK_RESERVE') or reason.startswith('MONITOR_FAILURE'):break
    return out
def main():
    assert json.loads((ROOT/'receipts/T_AUTHORIZATION.json').read_text())['approved'] is True
    baseline=ROOT/'receipts/Teval_T-1_0_exit.json'
    resource=ROOT/'receipts/Teval_T-1_0_resource.json'
    while not (baseline.exists() and resource.exists()):
        if resource.exists() and not baseline.exists():
            (ROOT/'results/T_PRETRAIN_BEHAVIOR_GATE.json').write_text(json.dumps(dict(status='BASELINE_TECHNICAL_FAILED',training_updates=0),indent=2)+'\n')
            (ROOT/'receipts/T_SCHEDULER_EXIT.json').write_text(json.dumps(dict(status='NOT_EXECUTED_BASELINE_TECHNICAL_FAILED',arms=[]),indent=2)+'\n');return
        time.sleep(5)
    x=json.loads(baseline.read_text());rows=list(csv.DictReader(open(ROOT/'results/Teval_T-1_0_episodes.csv')))
    learned={t:sum(int(r['success']) for r in rows if r['task']==str(t) and r['complete']=='True') for t in (1,3)}
    passed=x['status']=='COMPLETE' and json.loads(resource.read_text())['exit_code']==0 and len(rows)==500 and all(learned[t]>0 for t in (1,3))
    (ROOT/'results/T_PRETRAIN_BEHAVIOR_GATE.json').write_text(json.dumps(dict(status='PASS' if passed else 'SCIENTIFIC_GATE_FAILED',baseline_episodes=len(rows),task1_task3_success_counts=learned,note='Fresh-scene baseline checked before all student updates; old source development100%/94% established ability; zero fresh successes in a main task pauses T, never changes seeds or source checkpoint'),indent=2)+'\n')
    if not passed:
        (ROOT/'receipts/T_SCHEDULER_EXIT.json').write_text(json.dumps(dict(status='NOT_EXECUTED_SCIENTIFIC_GATE_FAILED',arms=[]),indent=2)+'\n');return
    source=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['t_core.py','t_train.py','t_scheduler.py','common.py','guard.py']})
    (ROOT/'receipts/T_TRAINING_SOURCE.json').write_text(json.dumps(source,indent=2)+'\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(queue,1,'16-19',[0,2]),pool.submit(queue,7,'20-23',[1,3])];out=[x for f in fs for x in f.result()]
    (ROOT/'receipts/T_SCHEDULER_EXIT.json').write_text(json.dumps(dict(arms=out),indent=2)+'\n')
if __name__=='__main__':main()
