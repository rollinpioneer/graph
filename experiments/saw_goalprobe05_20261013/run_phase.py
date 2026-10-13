"""Bounded own-process CPU phase; no intervention in other users' jobs."""
import os, sys, time, json, subprocess, pathlib, signal
root=pathlib.Path(__file__).resolve().parent
phase=sys.argv[1];assert phase in ('execute','analyze','audit','finalize_analysis')
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('JAX_PLATFORMS')=='cpu'
assert len(os.sched_getaffinity(0))<=4
start=json.loads((root/'receipts/RESOURCE_START.json').read_text())['epoch']
assert time.time()-start<14400
path=root/'receipts'/f'{phase.upper()}_PROCESS_RESOURCE.json';assert not path.exists()
f=open(root/'logs'/f'{phase}.log','x');begin=time.time();timefile=root/'receipts'/f'{phase.upper()}_TIME.txt';p=subprocess.Popen(['/usr/bin/time','-v','-o',str(timefile),sys.executable,str(root/(phase+'.py'))],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
peak=0;peak_threads=0;cpu=0.;samples=0;GPU_matches=[];reason=None
def process_tree(pid):
    out=[];pending=[pid]
    while pending:
        current=pending.pop()
        try:
            line=pathlib.Path(f'/proc/{current}/stat').read_text();fields=line[line.rfind(')')+2:].split()
            children=pathlib.Path(f'/proc/{current}/task/{current}/children').read_text().split();pending.extend(map(int,children))
            out.append(dict(pid=current,rss=int(fields[21])*os.sysconf('SC_PAGE_SIZE'),threads=int(fields[17]),CPU_seconds=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),affinity=os.sched_getaffinity(current)))
        except (FileNotFoundError,ProcessLookupError):pass
    return out
while p.poll() is None:
    samples+=1;procs=process_tree(p.pid)
    peak=max(peak,sum(x['rss'] for x in procs));peak_threads=max(peak_threads,sum(x['threads'] for x in procs));cpu=max(cpu,sum(x['CPU_seconds'] for x in procs))
    for x in procs:assert set(x['affinity']).issubset(set(os.sched_getaffinity(0)))
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
    assert r.returncode==0
    ours={x['pid'] for x in procs}
    for line in r.stdout.splitlines():
        v=line.split(',')
        if v and v[0].strip().isdigit() and int(v[0].strip()) in ours: GPU_matches.append(line)
    if GPU_matches: reason='OWN_GPU_ALLOCATION_FORBIDDEN'
    if time.time()-start>=14400:reason='CPU_WALL_CAP'
    if reason:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        break
    time.sleep(2)
code=p.wait();f.close();out=dict(phase=phase,exit_code=code,stop_reason=reason,epoch_start=begin,epoch_end=time.time(),wall_seconds=time.time()-begin,observed_CPU_seconds=cpu,peak_RSS_bytes_observed=peak,peak_threads_observed=peak_threads,CPU_affinity=sorted(os.sched_getaffinity(0)),resource_samples=samples,own_GPU_process_matches=GPU_matches,GPU_hours=0,GPU_MiB=0,new_training_updates=0)
path.write_text(json.dumps(out,indent=2)+'\n');print('PHASE_EXIT',phase,code,flush=True);sys.exit(code if code>=0 else 1)
