"""Separately authorized T training, two queues; no retries or early score decisions."""
import concurrent.futures,json,os,subprocess,hashlib
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
    source=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['t_core.py','t_train.py','t_scheduler.py','common.py','guard.py']})
    (ROOT/'receipts/T_TRAINING_SOURCE.json').write_text(json.dumps(source,indent=2)+'\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(queue,1,'16-19',[0,2]),pool.submit(queue,7,'20-23',[1,3])];out=[x for f in fs for x in f.result()]
    (ROOT/'receipts/T_SCHEDULER_EXIT.json').write_text(json.dumps(dict(arms=out),indent=2)+'\n')
if __name__=='__main__':main()
