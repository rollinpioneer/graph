"""Three paired GPU workers; own-tree aggregate monitoring, no time caps after amendment."""
import os,sys,json,time,subprocess,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert os.sched_getaffinity(0)=={75}
phase=sys.argv[1];assert phase in ('I','II')
auth=json.loads((ROOT/'receipts/AUTHORIZATION.json').read_text());assert auth['time_cost_limits_removed'] and auth[phase]
if phase=='II':assert json.loads((ROOT/'results/I_GATE.json').read_text())['H_R_pass']
ledger=ROOT/'receipts'/f'{phase}_WORKFLOW_MONITOR.json';assert not ledger.exists()
arms=[0,1] if phase=='I' else [2,3];slots={r:list(arms) for r in range(3)};jobs=[];running={};cpu=None;cpu_job=None;start=time.time();peakrss=peakgpu=0;reason=None
def save(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def launch(r,a):
    label=f'b{r}_a{a}';log=open(ROOT/'runs'/f'{label}_train.log','x')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.25',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
    p=subprocess.Popen(['taskset','-c',str(72+r),sys.executable,'train07.py','--block',str(r),'--arm',str(a)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    rec=dict(block=r,arm=a,pid=p.pid,start_epoch=time.time(),status='RUNNING',cpu=[72+r],gpu_index=0);jobs.append(rec);running[r]=(p,log,rec)
def descendants(roots):
    table={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            v=(p/'stat').read_text().split(') ',1)[1].split();table[int(p.name)]=(int(v[1]),int(v[21])*os.sysconf('SC_PAGE_SIZE'))
        except (OSError,IndexError,ValueError):pass
    ids=set(roots);changed=True
    while changed:
        before=len(ids);ids|={p for p,(parent,rss) in table.items() if parent in ids};changed=len(ids)>before
    return ids,sum(table.get(p,(0,0))[1] for p in ids)
def stop_all():
    for p,_,_ in running.values():
        if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
    if cpu and cpu.poll() is None:os.killpg(cpu.pid,signal.SIGTERM)
for r in range(3):launch(r,slots[r].pop(0))
while running:
    for r,(p,f,rec) in list(running.items()):
        rc=p.poll()
        if rc is None:continue
        f.close();rec.update(exit_code=rc,elapsed_seconds=time.time()-rec['start_epoch'],conservative_gpu_seconds=time.time()-rec['start_epoch'],child_exited=True)
        st=ROOT/'results'/f'b{r}_a{rec["arm"]}_status.json'
        success=rc==0 and st.exists() and json.loads(st.read_text())['status']=='COMPLETE_500K'
        rec['status']='COMPLETE_500K' if success else 'FAILED'
        del running[r]
        if not success:reason='FORMAL_JOB_FAILED_NO_AUTOMATIC_RESTART';stop_all()
        elif slots[r] and not reason:launch(r,slots[r].pop(0))
    # Evaluation code is separately committed and frozen before a ready receipt appears.
    ready=ROOT/'receipts/EVAL_CODE_READY.json'
    if phase=='I' and cpu is None and ready.exists() and not reason:
        info=json.loads(ready.read_text());assert hashlib.sha256((ROOT/'evaluate07.py').read_bytes()).hexdigest()==info['evaluate_sha256']
        f=open(ROOT/'runs/I_evaluate.log','x');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
        cpu=subprocess.Popen([sys.executable,'evaluate07.py','--phase','I','--watch'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        cpu_job=dict(pid=cpu.pid,start_epoch=time.time(),status='RUNNING',cpu=[75]);cpu_log=f
    if cpu and cpu.poll() is not None:
        cpu_log.close();cpu_job.update(exit_code=cpu.returncode,elapsed_seconds=time.time()-cpu_job['start_epoch'],child_exited=True,status='COMPLETE' if cpu.returncode==0 else 'FAILED')
        if cpu.returncode:reason='CPU_EVALUATION_FAILED';stop_all()
    roots=[p.pid for p,_,_ in running.values()]+([cpu.pid] if cpu and cpu.poll() is None else [])
    ids,rss=descendants(roots+[os.getpid()]);gpu=0
    try:
        out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
        for line in out.splitlines():
            pid,mem=line.split(',')
            if int(pid) in ids:gpu+=int(mem)*1024**2
    except Exception as exc:reason='GPU_MONITOR_UNAVAILABLE:'+repr(exc)
    peakrss=max(peakrss,rss);peakgpu=max(peakgpu,gpu)
    if gpu>12*1024**3:reason='OWNED_AGGREGATE_GPU_MEMORY_CAP'
    if rss>16*1024**3:reason='OWNED_AGGREGATE_RSS_CAP'
    if reason:stop_all()
    save(ledger,dict(phase=phase,status='STOPPING' if reason else 'RUNNING',reason=reason,start_epoch=start,elapsed_seconds=time.time()-start,jobs=jobs,cpu_evaluation=cpu_job,peak_owned_gpu_bytes=peakgpu,peak_owned_rss_bytes=peakrss,time_cost_limits_removed=True,conservative_gpu_seconds=sum(x.get('conservative_gpu_seconds',time.time()-x['start_epoch']) for x in jobs),CPU_union=[72,73,74,75]))
    time.sleep(5)
    if reason:
        for p,_,_ in running.values():
            if p.poll() is None and time.time()-start>30:
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
if cpu and cpu.poll() is None:
    # CPU worker can finish its queued committed evaluations; keep monitoring it.
    while cpu.poll() is None:
        ids,rss=descendants([cpu.pid,os.getpid()]);peakrss=max(peakrss,rss)
        if rss>16*1024**3:reason='OWNED_AGGREGATE_RSS_CAP';os.killpg(cpu.pid,signal.SIGTERM)
        save(ledger,dict(phase=phase,status='WAITING_CPU',reason=reason,start_epoch=start,elapsed_seconds=time.time()-start,jobs=jobs,cpu_evaluation=cpu_job,peak_owned_gpu_bytes=peakgpu,peak_owned_rss_bytes=peakrss,time_cost_limits_removed=True,conservative_gpu_seconds=sum(x['conservative_gpu_seconds'] for x in jobs),CPU_union=[72,73,74,75]))
        time.sleep(5)
    cpu_log.close();cpu_job.update(exit_code=cpu.returncode,elapsed_seconds=time.time()-cpu_job['start_epoch'],child_exited=True,status='COMPLETE' if cpu.returncode==0 else 'FAILED')
    if cpu.returncode:reason='CPU_EVALUATION_FAILED'
complete=not reason and len(jobs)==6 and all(x['status']=='COMPLETE_500K' for x in jobs)
save(ledger,dict(phase=phase,status='COMPLETE' if complete else 'FAILED',reason=reason,start_epoch=start,elapsed_seconds=time.time()-start,jobs=jobs,cpu_evaluation=cpu_job,all_owned_children_exited=True,peak_owned_gpu_bytes=peakgpu,peak_owned_rss_bytes=peakrss,time_cost_limits_removed=True,conservative_gpu_seconds=sum(x.get('conservative_gpu_seconds',0) for x in jobs),CPU_union=[72,73,74,75]))
print('WORKFLOW_DONE',phase,complete,flush=True);sys.exit(0 if complete else 1)
