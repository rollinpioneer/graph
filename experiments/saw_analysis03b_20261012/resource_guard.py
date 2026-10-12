"""Independent job guard: summed residency, own PID memory, disk, graceful save."""
import argparse,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
root=Path(__file__).resolve().parent;receipt=root/'receipts'/f'{a.label}_resource.json';active=root/'receipts'/f'{a.label}_active.json'
command=a.command[1:] if a.command[0]=='--' else a.command
started=time.time();samples=[];peak=0;reason=None
def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
with open(root/'logs'/f'{a.label}.log','x') as log:
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(active,dict(pid=child.pid,start=started,label=a.label))
    while child.poll() is None:
        now=time.time();elapsed=now-started
        try:
            live=[json.loads(f.read_text()) for f in (root/'receipts').glob('*_active.json')]
            pids={v['pid'] for v in live}
            lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()
            mem=sum(int(l.split(',')[1]) for l in lines if int(l.split(',')[0]) in pids)
            done=sum(json.loads(f.read_text())['gpu_residency_upper_seconds'] for f in (root/'receipts').glob('*_resource.json'))
            total=done+sum(now-v['start'] for v in live)
            disk=shutil.disk_usage(root).free
            peak=max(peak,mem);samples.append([round(elapsed,3),mem,round(total,3),disk])
            if elapsed>=a.seconds-90:reason='JOB_TIME_LIMIT'
            elif total>=19800-180:reason='TOTAL_GPU_LIMIT'
            elif mem>=12288:reason='AGGREGATE_MEMORY_LIMIT'
            elif disk<10*1024**3:reason='DISK_RESERVE'
        except Exception as e:reason='MONITOR_FAILURE:'+str(e)
        if reason:
            os.killpg(child.pid,signal.SIGUSR1)
            try:child.wait(timeout=75)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            break
        time.sleep(2)
    child.wait()
elapsed=time.time()-started
write(receipt,dict(label=a.label,pid=child.pid,command=command,exit_code=child.returncode,
    gpu_residency_upper_seconds=elapsed,gpu_hours=elapsed/3600,peak_aggregate_mib=peak,stop_reason=reason,samples=samples))
active.unlink(missing_ok=True)
sys.exit(child.returncode if child.returncode>=0 else 1)
