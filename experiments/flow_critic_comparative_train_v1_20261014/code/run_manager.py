"""Eight fixed jobs. One-shot native processes; successful T2 resumes once at 50k."""
import os,json,time,subprocess,sys,fcntl,resource,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,sort_keys=True));tmp.replace(p)
def read(p,default=None):
    try:return json.loads(Path(p).read_text())
    except FileNotFoundError:return default

def command(job,phase):
    host=job['host'];w=ROOT/f'workspaces/{host}_floq';py=str(ROOT/f'envs/{host}/bin/python')
    base=['taskset','-c',','.join(map(str,job['cpu'])),py,'main.py']
    out=ROOT/'runs'/job['job']/'output'
    gate=str(ROOT/'configs'/f"{job['job']}.gate.json")
    if host=='gfp':
        cmd=base+[f'wiring_gate={gate}',f"seed={job['seed']}",f"env_name={job['task']}",
            f'save_dir={out}',f"agent.critic_type={job['critic_type']}",'offline_steps=1000000','online_steps=0',
            'dataset_on_gpu=true','n_steps_fused=4','is_final=true','eval_episodes=100','n_eval_envs=10',
            'eval_interval=100000','save_interval=100000','save_last_checkpoint=true','log_interval=5000',
            'log_on_wandb=false','log_locally=true','video_episodes=0','hydra.run.dir=.','hydra.output_subdir=null']
    else:
        cmd=base+[f'--wiring_gate={gate}',f"--seed={job['seed']}",f"--env_name={job['task']}",f'--save_dir={out}',
            '--agent=agents/dfql.py',f"--agent.critic_type={job['critic_type']}",'--agent.alpha=2',
            '--agent.discount=0.99','--agent.q_agg=mean','--offline_steps=1000000','--online_steps=0',
            '--eval_episodes=50','--eval_interval=100000','--save_interval=100000','--log_interval=5000',
            '--wandb_mode=disabled','--video_episodes=0',f"--run_group={job['job']}"]
        if job['critic_type']=='floq':cmd+=['--agent.block_depth=2','--agent.block_width=512']
    if phase=='T3':
        admission=read(ROOT/'runs'/job['job']/'t2_admission.json')
        prog=read(ROOT/'runs'/job['job']/'training_progress.json')
        runtime=read(ROOT/'runs'/job['job']/'runtime.json')
        if not admission or admission['status']!='PASS' or prog['updates']!=50000:raise RuntimeError('T2 admission missing')
        source=runtime['output_dir'];key='restore_path' if host=='gfp' else '--restore_path'
        epoch='restore_epoch' if host=='gfp' else '--restore_epoch'
        cmd+=[f'{key}={source}',f'{epoch}=50000']
        # Gate stage is control metadata. Agent parameters/config/budget remain identical.
        gate_data=read(gate);gate_data['stage']='T3';write(gate,gate_data)
    env=dict(os.environ)
    for key in ['LD_LIBRARY_PATH','JAX_PLATFORMS','JAX_DEFAULT_MATMUL_PRECISION','JAX_ENABLE_X64']:env.pop(key,None)
    env.update(COMPARATIVE_ROOT=str(ROOT),COMPARATIVE_JOB=job['job'],COMPARATIVE_TASK=job['task'],
        COMPARATIVE_PHASE=phase,COMPARATIVE_PHASE_STOP=str(50000 if phase=='T2' else 1000000),
        CUDA_VISIBLE_DEVICES=str(job['gpu']),XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_PYTHON_CLIENT_MEM_FRACTION='0.30',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',HYDRA_FULL_ERROR='1',
        HOME=str(ROOT/'home'),MUJOCO_GL='egl')
    return cmd,w,env

def worker(job_path,phase):
    job=read(job_path);jobdir=ROOT/'runs'/job['job'];jobdir.mkdir(parents=True,exist_ok=True)
    cmd,w,env=command(job,phase)
    attempt_suffix='_attempt2' if job.get('attempt')==2 and phase=='T1' else ''
    start=time.time();log=ROOT/'logs'/f"{job['job']}_{phase}{attempt_suffix}.log"
    if log.exists():raise FileExistsError(log)
    write(jobdir/f'{phase}_command.json',dict(command=cmd,cwd=str(w),environment={k:v for k,v in env.items()
        if k.startswith(('COMPARATIVE_','CUDA_','XLA_')) or k in ['HOME','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MUJOCO_GL']}))
    write(jobdir/'status.json',dict(status='RUNNING',phase=phase,worker_pid=os.getpid(),start_epoch=start,gpu=job['gpu']))
    peaks=dict(rss_kb=0,gpu_memory_mib=0,cpu_seconds=0.)
    with log.open('x') as f:
        p=subprocess.Popen(cmd,cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write(jobdir/f'{phase}_pid.json',dict(native_pid=p.pid,worker_pid=os.getpid(),phase=phase,start_epoch=start))
        while p.poll() is None:
            try:
                status=Path(f'/proc/{p.pid}/status').read_text()
                rss=int(next(x.split()[1] for x in status.splitlines() if x.startswith('VmRSS:')))
                peaks['rss_kb']=max(peaks['rss_kb'],rss)
                st=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()
                peaks['cpu_seconds']=max(peaks['cpu_seconds'],sum(float(st[x]) for x in [11,12,13,14])/os.sysconf('SC_CLK_TCK'))
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
                for line in apps.splitlines():
                    pid,mem=line.split(',');
                    if int(pid)==p.pid:peaks['gpu_memory_mib']=max(peaks['gpu_memory_mib'],int(mem))
                if rss>48*1024*1024:
                    write(jobdir/f'{phase}_resource_stop.json',dict(reason='RSS exceeds frozen 48 GiB per job',rss_kb=rss))
                    p.terminate() # Only this explicitly approved child, never another experiment.
            except (OSError,subprocess.SubprocessError,StopIteration,ValueError):pass
            write(jobdir/f'{phase}_resources.json',dict(**peaks,epoch=time.time(),native_pid=p.pid,start_epoch=start))
            time.sleep(15)
        code=p.wait()
    end=time.time();prog=read(jobdir/'training_progress.json',{})
    expected=50000 if phase=='T2' else 1000000
    passed=code==0 and prog.get('updates')==expected and prog.get('stage')=='PHASE_COMPLETED'
    receipt=dict(exit_code=code,status='PASS' if passed else 'FAIL_STOPPED',phase=phase,native_pid=p.pid,
        start_epoch=start,end_epoch=end,gpu_resident_hours=(end-start)/3600,resources=peaks,
        final_progress=prog,automatic_restart=False)
    write(jobdir/f'{phase}_exit.json',receipt)
    write(jobdir/'status.json',dict(status='T2_PASS' if passed and phase=='T2' else 'COMPLETED' if passed else 'FAIL_STOPPED',
        phase=phase,exit_code=code,updates=prog.get('updates'),end_epoch=end,gpu=job['gpu']))
    return 0 if passed else 1

def manager():
    lock=(ROOT/'receipts/manager.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=read(ROOT/'configs/jobs.json');start=time.time();active={};failed=set();done=set()
    adopt='--adopt-authorized-startup-repair' in sys.argv
    original=read(ROOT/'receipts/manager_started.json')
    if original and not adopt:raise RuntimeError('Manager already dispatched; no duplicate/restart')
    if adopt:
        auth=read(ROOT/'reports/gfp_startup_failure_and_fix.json',{})
        assert auth.get('restart_authorized') is True and auth.get('restart_attempts_allowed')==1
        if (ROOT/'receipts/authorized_startup_repair_dispatched.json').exists():raise RuntimeError('Authorized retry already used')
        start=original['start_epoch']
    write(ROOT/'receipts/manager_started.json',dict(pid=os.getpid(),start_epoch=start,jobs=[j['job'] for j in jobs]))
    for j in jobs:write(ROOT/'configs'/f"{j['job']}.job.json",j)
    def launch(j,phase):
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
        if available<64*1024*1024:raise RuntimeError('RESOURCE_BLOCKED: less than 64 GiB available RAM')
        reserved={next(x['gpu'] for x in jobs if x['job']==key) for key in active}
        own_cards=set()
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        gpu_lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        inventory=[line.split(',') for line in gpu_lines.splitlines()]
        for line in apps.splitlines():
            uuid,pid=map(str.strip,line.split(','))
            try:
                if Path(f'/proc/{pid}').stat().st_uid==os.getuid():
                    own_cards.update(int(row[0]) for row in inventory if row[1].strip()==uuid)
            except FileNotFoundError:pass
        candidates=[row for row in inventory if int(row[0]) not in reserved|own_cards and int(row[2])>=16384]
        if not candidates:raise RuntimeError('RESOURCE_BLOCKED: no safe shared card; no wait/retry')
        chosen=next((row for row in candidates if int(row[0])==j['gpu']),None)
        if chosen is None:chosen=min(candidates,key=lambda row:(int(row[3]),-int(row[2])))
        j['gpu']=int(chosen[0]);write(ROOT/'configs'/f"{j['job']}.job.json",j)
        write(ROOT/'receipts'/f"{j['job']}_{phase}_placement.json",dict(gpu=j['gpu'],free_mib=int(chosen[2]),
            other_load_util=int(chosen[3]),available_ram_kb=available,epoch=time.time(),exclusive_gpu_required=False))
        key=j['job'];proc=subprocess.Popen([sys.executable,str(Path(__file__)), '--worker',
            str(ROOT/'configs'/f'{key}.job.json'),phase],stdin=subprocess.DEVNULL,
            stdout=open(ROOT/'logs'/f"{key}_{phase}{'_attempt2' if j.get('attempt')==2 and phase=='T1' else ''}_worker.log",'x'),stderr=subprocess.STDOUT,start_new_session=True)
        active[key]=(proc,phase);print('LAUNCH',key,phase,'GPU',j['gpu'],flush=True)
    class AdoptedWorker:
        def __init__(self,j):self.job=j
        def poll(self):
            receipt=read(ROOT/'runs'/self.job['job']/'T1_exit.json')
            if receipt:return 0 if receipt.get('status')=='PASS' else 1
            pid=read(ROOT/'runs'/self.job['job']/'T1_pid.json')['worker_pid']
            return None if Path(f'/proc/{pid}').exists() else 1
    # Adopt live DFQL workers without touching their processes or optimizer state.
    for j in jobs:
        if j['critic_type']!='scalar':continue
        if adopt and j['host']=='dfql':active[j['job']]=(AdoptedWorker(j),'T1')
        else:
            if adopt:j['attempt']=2
            launch(j,'T1')
    if adopt:write(ROOT/'receipts/authorized_startup_repair_dispatched.json',dict(epoch=time.time(),gfp_attempts=1,
        unchanged_dfql_workers=[j['job'] for j in jobs if j['host']=='dfql' and j['critic_type']=='scalar']))
    launched_hosts=set()
    while active or len(launched_hosts)<2:
        for key,(proc,phase) in list(active.items()):
            if proc.poll() is None:continue
            del active[key];j=next(x for x in jobs if x['job']==key)
            receipt=read(ROOT/'runs'/key/f'{phase}_exit.json',{})
            if receipt.get('status')!='PASS':failed.add(key)
            elif phase=='T2':launch(j,'T3')
            else:done.add(key)
        for host in ['gfp','dfql']:
            baselines=[j for j in jobs if j['critic_type']=='scalar' and j['host']==host]
            if host not in launched_hosts and all(j['job'] in done|failed for j in baselines):
                launched_hosts.add(host)
                for j in jobs:
                    if j['critic_type']=='scalar' or j['host']!=host:continue
                    if any(b['job'] in failed for b in baselines):
                        failed.add(j['job']);write(ROOT/'runs'/j['job']/'status.json',dict(status='BLOCKED_BASELINE_FAILURE',updates=0))
                    else:launch(j,'T2')
        write(ROOT/'receipts/queue.json',dict(active={k:phase for k,(p,phase) in active.items()},
            completed=sorted(done),failed=sorted(failed),epoch=time.time(),start_epoch=start,
            total_formal_update_cap=8000000,automatic_retry=False))
        if active:time.sleep(15)
    write(ROOT/'receipts/manager_finished.json',dict(epoch=time.time(),completed=sorted(done),failed=sorted(failed),
        wall_hours=(time.time()-start)/3600))
    subprocess.run([sys.executable,str(ROOT/'code/summarize.py')],check=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':sys.exit(worker(sys.argv[2],sys.argv[3]))
    manager()
