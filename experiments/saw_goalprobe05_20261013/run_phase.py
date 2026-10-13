"""Bounded own-process CPU phase; no intervention in other users' jobs."""
import os, sys, time, json, subprocess, pathlib, signal, psutil
root=pathlib.Path(__file__).resolve().parent
phase=sys.argv[1];assert phase in ('execute','analyze','audit')
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('JAX_PLATFORMS')=='cpu'
assert len(os.sched_getaffinity(0))<=4
start=json.loads((root/'receipts/RESOURCE_START.json').read_text())['epoch']
assert time.time()-start<14400
path=root/'receipts'/f'{phase.upper()}_PROCESS_RESOURCE.json';assert not path.exists()
f=open(root/'logs'/f'{phase}.log','x');begin=time.time();p=subprocess.Popen([sys.executable,str(root/(phase+'.py'))],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
peak=0;peak_threads=0;cpu=0.;samples=0;GPU_matches=[];reason=None
while p.poll() is None:
    samples+=1;procs=[]
    try:
        parent=psutil.Process(p.pid);procs=[parent]+parent.children(recursive=True)
        peak=max(peak,sum(x.memory_info().rss for x in procs));peak_threads=max(peak_threads,sum(x.num_threads() for x in procs));cpu=max(cpu,sum(x.cpu_times().user+x.cpu_times().system for x in procs))
        for x in procs: assert set(x.cpu_affinity()).issubset(set(os.sched_getaffinity(0)))
    except psutil.NoSuchProcess:pass
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
    assert r.returncode==0
    ours={x.pid for x in procs}
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
