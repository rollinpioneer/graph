import os,sys,time,json,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/xushijie3/work/flow_baselines_preflight_20261013/envs/gfp312/bin/python'
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def main():
    lock=ROOT/'receipts/manager_started.json'
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),started=time.time()),f)
    jobs={};rows={};evaluated=set();evals={}
    for arm,gpu in [('L0','4'),('L1','7')]:
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',HOME='/home/xushijie3/work/gfp_eta0_ablation_20261015/home')
        log=open(ROOT/'logs'/f'{arm}.log','w')
        proc=subprocess.Popen([PYTHON,str(ROOT/'code/train.py'),arm],stdout=log,stderr=subprocess.STDOUT,env=env)
        jobs[arm]=(proc,env,log);rows[arm]=dict(pid=proc.pid,gpu=gpu,phase='training',started=time.time())
        write(ROOT/'receipts/queue.json',rows)
    while jobs or evals:
        for arm,(proc,env,log) in list(jobs.items()):
            code=proc.poll()
            if code is not None:
                log.close();rows[arm].update(exit_code=code,ended=time.time(),phase='trained' if code==0 else 'failed')
                write(ROOT/f'runs/s3_{arm}/exit.json',dict(exit_code=code,pid=proc.pid,time=time.time()))
                del jobs[arm]
                if code==0:
                    assert json.loads((ROOT/f'runs/s3_{arm}/complete.json').read_text())['updates']==50000
                    evals[arm]=dict(env=env,index=0,proc=None)
        for arm,e in list(evals.items()):
            if e['proc'] is not None:
                code=e['proc'].poll()
                if code is None:continue
                e['log'].close();it=(25000,50000)[e['index']]
                write(ROOT/f'runs/s3_{arm}/eval_{it}/exit.json',dict(exit_code=code,pid=e['proc'].pid,time=time.time()))
                if code!=0:
                    rows[arm].update(phase='evaluation_failed',evaluation_exit=code);del evals[arm];continue
                evaluated.add((arm,it));e['index']+=1;e['proc']=None
            if e['index']==2:
                rows[arm]['phase']='complete';del evals[arm];continue
            it=(25000,50000)[e['index']]
            e['log']=open(ROOT/'logs'/f'{arm}_eval_{it}.log','w')
            e['proc']=subprocess.Popen([PYTHON,str(ROOT/'code/evaluate.py'),'3',arm,str(it)],stdout=e['log'],stderr=subprocess.STDOUT,env=e['env'])
            rows[arm].update(phase='evaluation',eval_updates=it,eval_pid=e['proc'].pid)
        write(ROOT/'receipts/queue.json',rows)
        # Other-user resource information is aggregate only.
        gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
        with open(ROOT/'logs/resources.jsonl','a') as f:f.write(json.dumps(dict(time=time.time(),gpu=gpu.stdout))+'\n')
        time.sleep(10)
    write(ROOT/'receipts/manager_complete.json',dict(time=time.time(),jobs=rows,evaluations_completed=len(evaluated),auto_retries=0))
if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/'receipts/manager_exception.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
