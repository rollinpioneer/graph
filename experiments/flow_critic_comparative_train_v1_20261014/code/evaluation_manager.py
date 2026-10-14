"""One evaluation child at a time, no training calls and no failed-job retries."""
import os,sys,json,time,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p,default=None):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return default
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
lock=(ROOT/'receipts/evaluation_manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
parallel='--adopt-all-eight-parallel' in sys.argv
adopt='--adopt-authorized-startup-repair' in sys.argv or parallel
if (ROOT/'receipts/evaluation_manager_started.json').exists() and not adopt:raise RuntimeError('No duplicate evaluation manager')
if adopt:
    auth=read(ROOT/'receipts/all_eight_parallel_authorized.json',{}) if parallel else read(ROOT/'reports/gfp_startup_failure_and_fix.json',{})
    assert auth.get('authorized' if parallel else 'restart_authorized') is True
    marker=ROOT/'receipts'/('evaluation_manager_parallel_adopted.json' if parallel else 'evaluation_manager_adopted.json')
    if marker.exists():raise RuntimeError('Coordinator already adopted')
    previous=read(ROOT/'receipts/evaluation_queue.json',{})
    assert previous.get('active') is None,'Do not interrupt an evaluation child'
    write(marker,dict(epoch=time.time(),optimizer_updates=0))
write(ROOT/'receipts/evaluation_manager_started.json',dict(pid=os.getpid(),epoch=time.time(),optimizer_updates=0))
jobs=read(ROOT/'configs/jobs.json');done=set();failed=set();skipped=set();active=None
if parallel:
    done=set(previous.get('completed',[]));failed=set(previous.get('failed',[]));skipped=set(previous.get('skipped',[]))
while len(done|failed|skipped)<len(jobs):
    if active:
        proc,j,start,gpu=active
        if proc.poll() is not None:
            code=proc.wait();key=j['job']
            receipt=read(ROOT/'runs'/key/'matched_verified/completed.json',{})
            (done if code==0 and receipt.get('status')=='PASS' else failed).add(key)
            write(ROOT/'runs'/key/'matched_verified_exit.json',dict(exit_code=code,start_epoch=start,
                end_epoch=time.time(),gpu=gpu,gpu_reserved_hours=(time.time()-start)/3600,optimizer_updates=0,automatic_restart=False))
            active=None
    if active is None:
        for j in jobs:
            key=j['job']
            if key in done|failed|skipped:continue
            status=read(ROOT/'runs'/key/'status.json',{})
            if status.get('status') in ['FAIL_STOPPED','BLOCKED_BASELINE_FAILURE']:skipped.add(key);continue
            if status.get('status')!='COMPLETED':continue
            inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
            own=set()
            for line in apps.splitlines():
                uuid,pid=map(str.strip,line.split(','))
                try:
                    if Path('/proc/'+pid).stat().st_uid==os.getuid():own.add(uuid)
                except FileNotFoundError:pass
            candidates=[row.split(',') for row in inventory.splitlines() if int(row.split(',')[2])>=17408]
            unshared=[row for row in candidates if row[1].strip() not in own]
            if unshared:candidates=unshared
            if not candidates:
                write(ROOT/'receipts/evaluation_resource_block.json',dict(status='RESOURCE_BLOCKED',job=key,epoch=time.time()))
                failed.add(key);continue
            row=min(candidates,key=lambda x:(int(x[3]),-int(x[2])));gpu=int(row[0])
            e=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
                XLA_PYTHON_CLIENT_MEM_FRACTION='0.30',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                HOME=str(ROOT/'home'),MUJOCO_GL='egl')
            for k in ['JAX_PLATFORMS','JAX_DEFAULT_MATMUL_PRECISION','JAX_ENABLE_X64','LD_LIBRARY_PATH']:e.pop(k,None)
            log=open(ROOT/'logs'/f'{key}_matched_verified.log','x');start=time.time()
            proc=subprocess.Popen(['taskset','-c','48-55',str(ROOT/f"envs/{j['host']}/bin/python"),
                str(ROOT/'code/matched_verified.py'),key],env=e,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active=(proc,j,start,gpu);break
    write(ROOT/'receipts/evaluation_queue.json',dict(completed=sorted(done),failed=sorted(failed),skipped=sorted(skipped),
        active=active[1]['job'] if active else None,epoch=time.time(),optimizer_updates=0))
    time.sleep(15)
write(ROOT/'receipts/evaluation_manager_finished.json',dict(completed=sorted(done),failed=sorted(failed),skipped=sorted(skipped),epoch=time.time()))
subprocess.run(['python',str(ROOT/'code/summarize.py')],check=True)
