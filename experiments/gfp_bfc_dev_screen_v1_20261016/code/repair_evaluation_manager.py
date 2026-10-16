import os,json,time,subprocess
from common import ROOT,PYTHON,FORMAL,write

def main():
    auth=json.loads((ROOT/'receipts/evaluation_repair_authorized.json').read_text());assert auth['approved'] and len(auth['jobs'])==9
    with open(ROOT/'receipts/repair_manager_started.json','x') as f:json.dump(dict(pid=os.getpid(),time=time.time()),f)
    pending=list(auth['jobs']);active=[];done=[];failed=[];dispatched=[]
    while pending or active:
        for j in list(active):
            rc=j['process'].poll()
            if rc is None:continue
            active.remove(j);j['log'].close()
            rec={k:v for k,v in j.items() if k not in ('process','log')};rec.update(exit_code=rc,ended=time.time())
            write(ROOT/'receipts'/f"exit_{j['id']}.json",rec)
            (done if rc==0 else failed).append(rec)
        if pending and len(active)<2:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            choices=[]
            for line in raw.splitlines():
                g,free=map(int,line.split(','));free-=4000*sum(j['gpu']==g and time.time()-j['started']<60 for j in active)
                if free>=8000:choices.append((free,g))
            if choices:
                _,gpu=max(choices);j=pending.pop(0);seed,arm,it=j['seed'],j['arm'],int(j['id'].rsplit('_',1)[1]);jid=j['id']+'_repair1'
                env=os.environ.copy();env.update(HOME=str(FORMAL/'home'),CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1')
                for k in ['LD_LIBRARY_PATH','JAX_DEFAULT_MATMUL_PRECISION','JAX_PLATFORMS']:env.pop(k,None)
                cpus=sorted(os.sched_getaffinity(0));offset=(len(dispatched)%2)*8
                cmd=['taskset','-c',','.join(map(str,cpus[-16+offset:-8+offset] if offset==0 else cpus[-8:])),PYTHON,str(ROOT/'code/evaluate.py'),str(seed),arm,str(it),'--repair']
                log=open(ROOT/'logs'/f'{jid}.log','x');p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                rec=dict(id=jid,seed=seed,arm=arm,updates=it,gpu=gpu,pid=p.pid,started=time.time(),command=cmd)
                dispatched.append(rec);active.append(dict(rec,process=p,log=log));write(ROOT/'receipts/repair_dispatch.json',dispatched)
        write(ROOT/'receipts/repair_queue.json',dict(time=time.time(),pending=[j['id'] for j in pending],active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],done=done,failed=failed))
        if pending or active:time.sleep(15)
    write(ROOT/'receipts/repair_complete.json',dict(time=time.time(),done=done,failed=failed))

if __name__=='__main__':main()
