"""Adopt live I workers after transport loss; never repeat a formal update/job."""
import os,sys,json,time,signal,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert os.sched_getaffinity(0)=={75}
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def identity(pid):
    p=Path('/proc',str(pid))
    try:
        s=(p/'stat').read_text().split(') ',1)[1].split()
        if s[0]=='Z':return None
        return dict(start_ticks=s[19],command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except (OSError,IndexError):return None
def alive(rec):
    x=identity(rec['pid']);return x is not None and x['start_ticks']==rec['process_start_ticks']
def completed(label):
    s=ROOT/'results'/f'{label}_status.json';c=ROOT/'results'/f'{label}_checkpoints.json'
    return s.exists() and read(s)['status']=='COMPLETE_500K' and read(s)['student_updates']==500000 and [x['step'] for x in read(c)]==[0,100000,200000,300000,400000,500000]
ledger=ROOT/'receipts/I_WORKFLOW_MONITOR.json';prior=read(ROOT/'receipts/I_WORKFLOW_MONITOR_BEFORE_TRANSPORT_LOSS.json')
assert read(ROOT/'receipts/AUTHORIZATION.json')['time_cost_limits_removed']
assert all(x['arm']==0 for x in prior['jobs']) and len(prior['jobs'])==3
jobs=prior['jobs'];active={};native={};logs={};start=prior['start_epoch'];peakgpu=prior['peak_owned_gpu_bytes'];peakrss=prior['peak_owned_rss_bytes'];reason=None
for rec in jobs:
    x=identity(rec['pid'])
    if x:
        assert 'train07.py' in x['command'] and f"--block {rec['block']} --arm 0" in x['command'];rec['process_start_ticks']=x['start_ticks'];active[rec['block']]=rec
    else:
        assert completed(f"b{rec['block']}_a0");rec.update(status='COMPLETE_500K',child_exited=True,exit_code=None,native_exit_code_unavailable_after_original_manager_loss=True,elapsed_seconds=time.time()-rec['start_epoch'],conservative_gpu_seconds=time.time()-rec['start_epoch'],process_start_ticks=None)
cpu=prior['cpu_evaluation'];x=identity(cpu['pid'])
if x:assert 'evaluate07.py --phase I --watch' in x['command'];cpu['process_start_ticks']=x['start_ticks']
else:assert read(ROOT/'receipts/I_EVALUATION_EXIT.json')['status']=='COMPLETE';cpu.update(status='COMPLETE',child_exited=True,exit_code=None,native_exit_code_unavailable_after_original_manager_loss=True,elapsed_seconds=time.time()-cpu['start_epoch'],process_start_ticks=None)

def launch(r):
    assert not (ROOT/'runs'/f'b{r}_a1').exists() and not (ROOT/'results'/f'b{r}_a1_status.json').exists()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.25')
    f=open(ROOT/'runs'/f'b{r}_a1_train.log','x');p=subprocess.Popen(['taskset','-c',str(72+r),sys.executable,'train07.py','--block',str(r),'--arm','1'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    x=identity(p.pid);assert x is not None
    rec=dict(block=r,arm=1,pid=p.pid,process_start_ticks=x['start_ticks'],start_epoch=time.time(),status='RUNNING',cpu=[72+r],gpu_index=0,first_and_only_formal_start=True)
    jobs.append(rec);active[r]=rec;native[r]=p;logs[r]=f
def owned(roots):
    table={}
    for q in Path('/proc').iterdir():
        if not q.name.isdigit():continue
        try:
            s=(q/'stat').read_text().split(') ',1)[1].split();table[int(q.name)]=(int(s[1]),int(s[21])*os.sysconf('SC_PAGE_SIZE'))
        except (OSError,IndexError,ValueError):pass
    ids=set(roots)
    while True:
        n=len(ids);ids|={pid for pid,(parent,rss) in table.items() if parent in ids}
        if len(ids)==n:break
    return ids,sum(table.get(pid,(0,0))[1] for pid in ids)
def stop_own():
    for rec in list(active.values())+[cpu]:
        if rec['status']=='RUNNING' and alive(rec):os.killpg(rec['pid'],signal.SIGTERM)
def record(status):
    save(ledger,dict(phase='I',status=status,reason=reason,start_epoch=start,elapsed_seconds=time.time()-start,jobs=jobs,cpu_evaluation=cpu,peak_owned_gpu_bytes=peakgpu,peak_owned_rss_bytes=peakrss,time_cost_limits_removed=True,conservative_gpu_seconds=sum(x.get('conservative_gpu_seconds',time.time()-x['start_epoch']) for x in jobs),CPU_union=[72,73,74,75],recovered_supervisor_pid=os.getpid(),original_supervisor_pid=1646513,formal_restarts=0,monitor_gap_start_epoch=prior['start_epoch']+prior['elapsed_seconds'],monitor_gap_end_epoch=recovery_epoch,unobserved_gap_peak_resource_usage_not_reconstructed=True,all_owned_children_exited=(status in ('COMPLETE','FAILED') and not active and cpu['status']!='RUNNING')))
recovery_epoch=time.time();lastbytes=0;size=0
for r in range(3):
    if r not in active:launch(r)
while active or cpu['status']=='RUNNING':
    for r,rec in list(active.items()):
        if r in native:
            rc=native[r].poll()
            if rc is None:continue
            logs[r].close();rec['exit_code']=rc
        else:
            if alive(rec):continue
            rc=None;rec.update(exit_code=None,native_exit_code_unavailable_after_original_manager_loss=True)
        label=f"b{r}_a{rec['arm']}";ok=completed(label) and (rc is None or rc==0)
        rec.update(status='COMPLETE_500K' if ok else 'FAILED',child_exited=True,elapsed_seconds=time.time()-rec['start_epoch'],conservative_gpu_seconds=time.time()-rec['start_epoch']);del active[r]
        if not ok:reason='FORMAL_JOB_FAILED_NO_AUTOMATIC_RESTART';stop_own()
        elif rec['arm']==0 and not reason:launch(r)
    if cpu['status']=='RUNNING' and not alive(cpu):
        ep=ROOT/'receipts/I_EVALUATION_EXIT.json';ok=ep.exists() and read(ep)['status']=='COMPLETE'
        cpu.update(status='COMPLETE' if ok else 'FAILED',child_exited=True,exit_code=None,native_exit_code_unavailable_after_original_manager_loss=True,elapsed_seconds=time.time()-cpu['start_epoch'])
        if not ok:reason='CPU_EVALUATION_FAILED_NO_AUTOMATIC_RERUN';stop_own()
    roots=[x['pid'] for x in active.values()]+([cpu['pid']] if cpu['status']=='RUNNING' else [])+[os.getpid()]
    ids,rss=owned(roots);gpu=0
    try:
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
        for line in raw.splitlines():
            pid,mem=line.split(',')
            if int(pid) in ids:gpu+=int(mem)*1024**2
    except Exception as exc:reason='GPU_MONITOR_UNAVAILABLE:'+repr(exc)
    peakgpu=max(peakgpu,gpu);peakrss=max(peakrss,rss)
    if time.time()-lastbytes>=30:size=sum(q.stat().st_size for q in ROOT.rglob('*') if q.is_file());lastbytes=time.time()
    if gpu>12*1024**3:reason='OWNED_GPU_MEMORY_CAP'
    if rss>16*1024**3:reason='OWNED_RSS_CAP'
    if size>50*1024**3:reason='NEW_ARTIFACT_BYTES_CAP'
    if reason:stop_own()
    record('STOPPING' if reason else ('RUNNING' if active else 'WAITING_CPU'));time.sleep(5)
ok=not reason and len(jobs)==6 and all(x['status']=='COMPLETE_500K' for x in jobs) and cpu['status']=='COMPLETE'
record('COMPLETE' if ok else 'FAILED');print('RECOVERED_I_WORKFLOW_DONE',ok,flush=True);sys.exit(0 if ok else 1)
