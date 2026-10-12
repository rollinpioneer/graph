"""Separately account T CPU evaluation and verify actual four-arm start identity."""
import concurrent.futures,json,os,subprocess,time,pickle,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PY='/home/xushijie3/work/graph_saw_rd01_20261012/experiments/saw_rd01_20261012/vendor/venv/bin/python'
def main():
    assert json.loads((ROOT/'receipts/T_AUTHORIZATION.json').read_text())['approved']
    pending=[(-1,0)]+[(a,s) for a in range(4) for s in (25000,50000,100000)];running={};finished=[];cores=['92-95','96-99']
    def write(n,x):
        p=ROOT/'receipts'/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
    while pending or running:
        for label,(p,job,c,f) in list(running.items()):
            if p.poll() is not None:finished.append(dict(label=label,job=job,exit_code=p.wait()));f.close();cores.append(c);del running[label]
        paths=[ROOT/'results'/f'T{a}_checkpoints.json' for a in range(4)]
        if all(p.exists() for p in paths) and not (ROOT/'receipts/T_ACTUAL_UPDATE0.json').exists():
            hashes=[]
            for p in paths:
                c=next(x for x in json.loads(p.read_text()) if x['step']==0)
                with open(c['path'],'rb') as f:s=pickle.load(f)
                hashes.append(hashlib.sha256(pickle.dumps((s['agent'],s['actor_opt_state'],s['actor_rng'],s['numpy_rng'],s['python_rng'],s['stream_cursor'],s['stream_manifest_sha256']),4)).hexdigest())
            passed=len(set(hashes))==1;write('T_ACTUAL_UPDATE0.json',dict(status='PASS' if passed else 'FAILED',serialized_agent_actor_optimizer_rng_cursor_sha256=hashes))
            # Root must preserve and invalidate any failure; identity never drives model selection.
            assert passed,'T_ACTUAL_UPDATE0_MISMATCH'
        for job in pending.copy():
            if not cores:break
            arm,step=job
            if arm>=0:
                file=ROOT/'results'/f'T{arm}_checkpoints.json';resource=ROOT/'receipts'/f'T{arm}_resource.json'
                ready=file.exists() and any(x['step']==step for x in json.loads(file.read_text()))
                if step==100000:
                    status=ROOT/'results'/f'T{arm}_training_status.json';ready=ready and resource.exists() and status.exists() and json.loads(resource.read_text())['exit_code']==0 and json.loads(status.read_text())['status']=='COMPLETE_100K'
                if not ready:
                    if resource.exists():finished.append(dict(job=job,status='NOT_EXECUTED_MISSING_VALID_STAGE'));pending.remove(job)
                    continue
            label=f'Teval_T{arm}_{step}';c=cores.pop(0);f=open(ROOT/'logs'/f'{label}_guard.log','x');env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu');cmd=[PY,str(ROOT/'t_cpu_guard.py'),'--label',label,'--group','T-EVAL','--seconds','2400','--','taskset','-c',c,PY,str(ROOT/'t_eval.py'),'--arm',str(arm),'--step',str(step)]
            p=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);running[label]=(p,job,c,f);pending.remove(job)
        write('T_EVALUATION_PROGRESS.json',dict(pending=pending,running=list(running),finished=finished))
        if pending and not running and (ROOT/'receipts/T_SCHEDULER_EXIT.json').exists():
            for job in pending:finished.append(dict(job=job,status='NOT_EXECUTED_MISSING_VALID_STAGE'))
            pending=[]
        if pending or running:time.sleep(5)
    write('T_EVALUATION_EXIT.json',dict(finished=finished));print('T_EVALUATION_FINISHED',flush=True)
if __name__=='__main__':main()
