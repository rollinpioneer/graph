import os,sys,json,time,subprocess
from common import ROOT,PYTHON,FORMAL,write

class AdoptedProcess:
    def __init__(self,pid):
        self.pid=pid
        self.start=self.stat()[19]
    def stat(self):
        return open(f'/proc/{self.pid}/stat').read().rsplit(')',1)[1].split()
    def poll(self):
        a=self.stat()
        assert a[19]==self.start, 'PID identity changed'
        return os.waitstatus_to_exitcode(int(a[49])) if a[0]=='Z' else None

def main():
    assert json.loads((ROOT/'receipts/zero_update_admission.json').read_text())['passed']
    old=json.loads((ROOT/'receipts/manager_started.json').read_text())
    old_pid=old['pid'];write(ROOT/'receipts/previous_manager_identity.json',old)
    assert open(f'/proc/{old_pid}/stat').read().rsplit(')',1)[1].split()[0]=='T'
    alljobs=json.loads((ROOT/'receipts/dispatch.json').read_text())
    active=[];done=[];failed=[]
    for j in alljobs:
        receipt=ROOT/'receipts'/f"exit_{j['id']}.json"
        if receipt.exists():
            rec=json.loads(receipt.read_text());(done if rec['exit_code']==0 else failed).append(rec)
        else:
            active.append(dict(j,process=AdoptedProcess(j['pid']),log=open(os.devnull,'w'),adopted=True))
    launched_train={(j['seed'],j['arm']) for j in alljobs if j['kind']=='train'}
    pending=[(seed,arm) for arm in ['R0','R2','R4','R5','R1','R3'] for seed in (0,3) if (seed,arm) not in launched_train]
    launched_eval={(j['seed'],j['arm'],int(j['id'].rsplit('_',1)[1])) for j in alljobs if j['kind']=='eval'}
    eval_pending=[];old_retired=False
    write(ROOT/'receipts/manager_started.json',dict(pid=os.getpid(),started=time.time(),adopted_from=old_pid))
    while pending or active or eval_pending:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        gpus=[tuple(map(int,l.split(','))) for l in raw.strip().splitlines()]
        gpus=[(g,free-9000*sum(j['gpu']==g and time.time()-j['started']<60 for j in active),util) for g,free,util in gpus]
        for job in list(active):
            rc=job['process'].poll()
            if rc is None:continue
            active.remove(job);job['log'].close()
            rec={k:v for k,v in job.items() if k not in ('process','log')};rec.update(exit_code=rc,ended=time.time())
            write(ROOT/'receipts'/f"exit_{job['id']}.json",rec)
            (done if rc==0 else failed).append(rec)
        # Queue each committed checkpoint exactly once. Evaluation never updates parameters.
        for seed in (0,3):
            for arm in [f'R{i}' for i in range(6)]:
                for it in (50000,100000):
                    key=(seed,arm,it)
                    if key not in launched_eval and (ROOT/'runs'/f's{seed}_{arm}'/f'checkpoint_{it}.json').exists():
                        eval_pending.append(key);launched_eval.add(key)
        if not old_retired and not any(j.get('adopted') for j in active):
            import signal
            os.kill(old_pid,signal.SIGTERM);os.kill(old_pid,signal.SIGCONT)
            old_retired=True
            write(ROOT/'receipts/old_coordinator_retired.json',dict(pid=old_pid,time=time.time(),all_child_exit_codes_collected=True))
        if len(active)<14:
            occupancy={g:sum(j['gpu']==g for j in active) for g,_,_ in gpus}
            choices=[(g,free,util) for g,free,util in gpus if free>=10000]
            if choices:
                gpu,free,util=sorted(choices,key=lambda x:(-x[1],x[2]))[0]
                # At most two evaluations, preserving CPU headroom.
                do_eval=not pending and bool(eval_pending) and sum(j['kind']=='eval' for j in active)<2
                if pending or do_eval:
                    if do_eval:
                        seed,arm,it=eval_pending.pop(0);kind='eval';script='evaluate.py';args=[str(seed),arm,str(it)];jid=f'eval_s{seed}_{arm}_{it}'
                    else:
                        seed,arm=pending.pop(0);kind='train';script='train.py';args=[str(seed),arm];jid=f'train_s{seed}_{arm}'
                    env=os.environ.copy();env.update(HOME=str(FORMAL/'home'),CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1',WANDB_MODE='disabled')
                    for k in ['LD_LIBRARY_PATH','JAX_DEFAULT_MATMUL_PRECISION','JAX_PLATFORMS']:env.pop(k,None)
                    log=open(ROOT/'logs'/f'{jid}.log','w')
                    available=sorted(os.sched_getaffinity(0));slot=len(alljobs)%max(1,len(available)//8)
                    cpus=available[slot*8:slot*8+8]
                    cmd=['taskset','-c',','.join(map(str,cpus)),PYTHON,str(ROOT/'code'/script)]+args
                    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    job=dict(id=jid,kind=kind,seed=seed,arm=arm,gpu=gpu,pid=proc.pid,started=time.time(),command=cmd,process=proc,log=log)
                    active.append(job);alljobs.append({k:v for k,v in job.items() if k not in ('process','log')})
                    write(ROOT/'receipts/dispatch.json',alljobs)
        write(ROOT/'receipts/queue.json',dict(time=time.time(),pending=pending,evaluation_pending=eval_pending,
          active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],done=done,failed=failed,gpus=gpus))
        if pending or active or eval_pending:time.sleep(2 if pending else 15)
    write(ROOT/'receipts/manager_complete.json',dict(time=time.time(),done=done,failed=failed))

if __name__=='__main__':main()
