"""Conservative child residency accounting and aggregate memory gate."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--label', required=True)
p.add_argument('--seconds', type=float, required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
root = Path(__file__).resolve().parent
out = root / 'receipts' / (a.label + '_resource.json')
log = root / 'logs' / (a.label + '.log')
out.parent.mkdir(exist_ok=True); log.parent.mkdir(exist_ok=True)
start = time.monotonic(); samples=[]; peak=0; stopped=None
command=a.command[1:] if a.command and a.command[0]=='--' else a.command
with open(log, 'w') as f:
    child = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    while child.poll() is None:
        elapsed = time.monotonic()-start
        try:
            lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'], text=True, timeout=5).splitlines()
            # Includes all instrumented task jobs, never other users' jobs.
            task_pids={child.pid}
            for fp in (root/'receipts').glob('*_active_pid.txt'):
                try: task_pids.add(int(fp.read_text()))
                except (ValueError, OSError): pass
            mem=sum(int(line.split(',')[1]) for line in lines if int(line.split(',')[0]) in task_pids)
            peak=max(peak,mem)
            samples.append([round(elapsed,3),mem])
        except Exception as e:
            stopped='MEMORY_MONITOR_FAILURE:'+str(e)
            mem=0
        if stopped is None and (elapsed >= a.seconds-90 or mem > 12288):
            stopped='TIME_BUDGET' if elapsed >= a.seconds-90 else 'AGGREGATE_MEMORY'
        if stopped:
            os.killpg(child.pid, signal.SIGUSR1)
            try: child.wait(timeout=75)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
            break
        active=root/'receipts'/(a.label+'_active_pid.txt')
        active.write_text(str(child.pid))
        time.sleep(2)
elapsed=time.monotonic()-start
(root/'receipts'/(a.label+'_active_pid.txt')).unlink(missing_ok=True)
out.write_text(json.dumps(dict(label=a.label,pid=child.pid,command=command,exit_code=child.returncode,
                              gpu_residency_upper_seconds=elapsed,gpu_hours=elapsed/3600,
                              peak_aggregate_mib=peak,stop_reason=stopped,samples=samples),indent=2)+'\n')
sys.exit(child.returncode if child.returncode is not None and child.returncode >= 0 else 1)
