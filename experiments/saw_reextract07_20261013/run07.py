"""Monitor only this child tree, cap aggregate owned GPU/RSS and conservatively bill."""
import os, sys, time, json, subprocess, signal
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert len(os.sched_getaffinity(0))<=4
phase=sys.argv[1]; script=sys.argv[2]; args=sys.argv[3:]
assert phase in ('P','I','II','CPU_PREP','CPU_EVAL')
start=time.time(); cap={'P':1800,'I':1800,'II':2700,'CPU_PREP':14400,'CPU_EVAL':28800}[phase]
rec=ROOT/'receipts'/f'{phase}_{Path(script).stem}_MONITOR.json'
assert not rec.exists()
previous=0
if phase=='P':
    for q in (ROOT/'receipts').glob('P_*_MONITOR.json'):
        x=json.loads(q.read_text());assert x.get('child_exited'), 'PREVIOUS_P_STILL_RUNNING'
        previous+=x['conservative_gpu_seconds']
    cap-=previous;assert cap>0,'P_CUMULATIVE_BUDGET_EXHAUSTED'
def save(x):
    q=rec.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(rec)
def tree(pid):
    current={pid};changed=True;table={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            v=(p/'stat').read_text().split(') ',1)[1].split();table[int(p.name)]=(int(v[1]),int(v[21])*os.sysconf('SC_PAGE_SIZE'))
        except (OSError,IndexError,ValueError):pass
    while changed:
        before=len(current);current|={p for p,(parent,rss) in table.items() if parent in current};changed=len(current)>before
    return current,sum(table.get(p,(0,0))[1] for p in current)
env=os.environ.copy()
if phase in ('P','I','II'):
    env.update(CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.25')
else:env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
log=ROOT/'runs'/f'{phase}_{Path(script).stem}.log';f=open(log,'x')
p=subprocess.Popen([sys.executable,script,*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
status='RUNNING';peakrss=peakgpu=0;reason=None
while p.poll() is None:
    ids,rss=tree(p.pid);peakrss=max(peakrss,rss)
    gpu=0
    try:
        out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
        for line in out.splitlines():
            pid,mem=line.split(',');
            if int(pid) in ids:gpu+=int(mem)*1024**2
    except Exception as e:reason='GPU_MONITOR_UNAVAILABLE:'+str(e)
    peakgpu=max(peakgpu,gpu)
    if time.time()-start>=cap:reason='WALL_HARD_CAP'
    if gpu>12*1024**3:reason='OWNED_GPU_MEMORY_CAP'
    if phase.startswith('CPU') and gpu:reason='GPU_FORBIDDEN_IN_CPU_PHASE'
    if rss>16*1024**3:reason='OWNED_RSS_CAP'
    if reason:
        status='RESOURCE_STOP';os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
        break
    save(dict(phase=phase,status=status,pid=p.pid,start_epoch=start,elapsed_seconds=time.time()-start,peak_rss_bytes=peakrss,peak_owned_gpu_bytes=peakgpu,cpu_affinity=sorted(os.sched_getaffinity(0)),conservative_gpu_seconds=time.time()-start if phase in ('P','I','II') else 0))
    time.sleep(2)
rc=p.wait();f.close();elapsed=time.time()-start
save(dict(phase=phase,status=status if reason else ('COMPLETE' if rc==0 else 'FAILED'),reason=reason,exit_code=rc,pid=p.pid,child_exited=True,start_epoch=start,elapsed_seconds=elapsed,prior_same_phase_gpu_seconds=previous,peak_rss_bytes=peakrss,peak_owned_gpu_bytes=peakgpu,cpu_affinity=sorted(os.sched_getaffinity(0)),conservative_gpu_seconds=elapsed if phase in ('P','I','II') else 0))
print('MONITOR_DONE',phase,rc,elapsed,flush=True)
sys.exit(rc or (1 if reason else 0))
