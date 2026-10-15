"""User-authorized simultaneous dispatch; adopt every live worker without restarting it."""
import os,sys,time,json,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from run_manager import read,write

def inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(a),uuid=b.strip(),free_mib=int(c),util=int(d)) for a,b,c,d in
        (line.split(',') for line in raw.splitlines())]

def main():
    lock=(ROOT/'receipts/manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    auth=read(ROOT/'receipts/all_eight_parallel_authorized.json',{})
    assert auth.get('authorized') is True and auth.get('formal_update_cap')==8000000
    marker=ROOT/'receipts/all_eight_parallel_dispatched.json'
    if marker.exists():raise RuntimeError('Parallel dispatcher already used; no duplicate/retry')
    jobs=read(ROOT/'configs/jobs.json');old=read(ROOT/'receipts/manager_started.json');start=old['start_epoch']
    active={};done=set();failed=set();pending=[];local_launches=[]
    write(ROOT/'receipts/manager_started.json',dict(pid=os.getpid(),start_epoch=start,
        jobs=[j['job'] for j in jobs],entrypoint='code/all_eight_manager.py',baseline_barrier=False))

    def adopt(j):
        key=j['job'];state=read(ROOT/'runs'/key/'status.json')
        if state is None:pending.append(j);return
        if state['status']=='COMPLETED':done.add(key);return
        if state['status'] in ['FAIL_STOPPED','BLOCKED_BASELINE_FAILURE']:
            failed.add(key);return # The all-start request does not authorize failed-job retries.
        phase=state['phase'];pid=read(ROOT/'runs'/key/f'{phase}_pid.json')
        wp=pid['worker_pid'];npid=pid['native_pid']
        for check in [wp,npid]:
            if Path(f'/proc/{check}').stat().st_uid!=os.getuid():raise RuntimeError('Worker UID mismatch')
        active[key]=dict(phase=phase,worker_pid=wp,adopted=True,gpu=state['gpu'])
        j['gpu']=state['gpu']

    for j in jobs:adopt(j)
    # Separate CPU affinity for new jobs, preserving 8 cores and thread limits per job.
    cpu_groups={'G1_seed0':list(range(32,40)),'G1_seed1':list(range(40,48)),
        'D1_seed0':list(range(96,104)),'D1_seed1':list(range(104,112))}
    for j in pending:
        if j['job'] in cpu_groups:j['cpu']=cpu_groups[j['job']]

    def launch(j,phase):
        key=j['job'];d=ROOT/'runs'/key
        if read(d/f'{phase}_pid.json') or (ROOT/'logs'/f'{key}_{phase}.log').exists():
            raise RuntimeError('Existing dispatch identity; refusing repeat '+key+' '+phase)
        mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
        if mem<64*1024*1024:raise RuntimeError('RESOURCE_BLOCKED: RAM floor64GiB')
        cards=inventory();planned={2 if x['job']=='G1_seed0' else 3 for x in pending
            if x['host']=='gfp' and x['job'] not in active}
        slots={'G1_seed0':2,'G1_seed1':3,'D1_seed0':5,'D1_seed1':7}
        preferred=slots.get(key,j['gpu'])
        required_free=21504 if j['host']=='gfp' else 17408
        active_combo_cards={x['gpu'] for k,x in active.items() if k.startswith(('G1','D1'))}
        candidates=[x for x in cards if x['free_mib']>=required_free and x['index'] not in active_combo_cards]
        if j['host']=='dfql':
            without_planned=[x for x in candidates if x['index'] not in planned]
            if without_planned:candidates=without_planned
        if not candidates:raise RuntimeError('RESOURCE_BLOCKED: no card with16GiB free; no wait/retry')
        preferred_card=next((x for x in candidates if x['index']==preferred),None)
        card=preferred_card if preferred_card else min(candidates,key=lambda x:(x['util'],-x['free_mib']))
        j['gpu']=card['index'];write(ROOT/'configs'/f'{key}.job.json',j)
        prior=[k for k,x in active.items() if x['gpu']==j['gpu']]
        write(ROOT/'receipts'/f'{key}_{phase}_placement.json',dict(gpu=j['gpu'],free_mib=card['free_mib'],
            utilization=card['util'],available_ram_kb=mem,shared_own_jobs=prior,epoch=time.time(),
            exclusive_gpu_required=False,baseline_barrier=False,allocator_fraction=.30,
            required_free_mib=required_free,gpu_headroom_mib=4096))
        log=ROOT/'logs'/f'{key}_{phase}_worker.log'
        if log.exists():raise FileExistsError(log)
        p=subprocess.Popen([sys.executable,str(ROOT/'code/run_manager.py'),'--worker',
            str(ROOT/'configs'/f'{key}.job.json'),phase],stdin=subprocess.DEVNULL,stdout=log.open('x'),
            stderr=subprocess.STDOUT,start_new_session=True)
        active[key]=dict(phase=phase,worker_pid=p.pid,adopted=False,gpu=j['gpu'],process=p)
        local_launches.append(dict(job=key,phase=phase,worker_pid=p.pid,gpu=j['gpu']))
        print('LAUNCH_NOW',key,phase,'GPU',j['gpu'],'shared',prior,flush=True)

    # Every previously unstarted combination starts now; no baseline dependency.
    for j in pending:
        try:launch(j,'T2' if j['critic_type']=='floq' else 'T1')
        except Exception as e:
            failed.add(j['job']);write(ROOT/'runs'/j['job']/'status.json',dict(status='DISPATCH_BLOCKED',
                reason=str(e),updates=0,automatic_retry=False));print('DISPATCH_BLOCKED',j['job'],str(e),flush=True)
    write(marker,dict(epoch=time.time(),new_launches=local_launches,
        adopted_jobs=[k for k,x in active.items() if x['adopted']],formal_update_cap=8000000,baseline_barrier=False))
    while active:
        for key,x in list(active.items()):
            phase=x['phase'];receipt=read(ROOT/'runs'/key/f'{phase}_exit.json')
            if not receipt:
                try:
                    path=Path('/proc/'+str(x['worker_pid']))
                    alive=path.exists() and path.stat().st_uid==os.getuid()
                    if alive:
                        stat=path.joinpath('stat').read_text().rsplit(')',1)[1].split();alive=stat[0]!='Z'
                except OSError:alive=False
                if alive:continue
                receipt=dict(status='FAIL_STOPPED',reason='Worker disappeared without an exit receipt')
                write(ROOT/'runs'/key/'coordinator_failure.json',receipt)
            del active[key];j=next(z for z in jobs if z['job']==key)
            if receipt.get('status')!='PASS':failed.add(key)
            elif phase=='T2':
                try:launch(j,'T3')
                except Exception as e:
                    failed.add(key);write(ROOT/'runs'/key/'t3_dispatch_blocked.json',dict(reason=str(e),
                        completed_updates=50000,automatic_retry=False))
            else:done.add(key)
        write(ROOT/'receipts/queue.json',dict(active={k:x['phase'] for k,x in active.items()},
            completed=sorted(done),failed=sorted(failed),epoch=time.time(),start_epoch=start,
            total_formal_update_cap=8000000,automatic_retry=False,baseline_barrier=False,
            coordinator_entrypoint='code/all_eight_manager.py'))
        if active:time.sleep(15)
    write(ROOT/'receipts/manager_finished.json',dict(epoch=time.time(),completed=sorted(done),failed=sorted(failed),
        wall_hours=(time.time()-start)/3600,baseline_barrier=False))
    subprocess.run([sys.executable,str(ROOT/'code/summarize.py')],check=True)

if __name__=='__main__':main()
