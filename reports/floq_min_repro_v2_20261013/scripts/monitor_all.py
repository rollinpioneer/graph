"""Read-only inspection of training/evaluation; writes only monitoring receipts."""
import os
import pwd
import json
import time
import pathlib
import subprocess
import psutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
OLD = pathlib.Path('/home/xushijie3/work/flow_baselines_preflight_20261013')


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def cpu_process(p):
    try:
        t = p.cpu_times()
        return dict(pid=p.pid, create_epoch=p.create_time(), status=p.status(),
                    rss_bytes=p.memory_info().rss, cpu_seconds=t.user+t.system)
    except psutil.Error:
        return None


def main():
    now = time.time()
    username = pwd.getpwuid(os.getuid()).pw_name
    gpu_raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'],text=True)
    gpu = []
    for line in gpu_raw.splitlines():
        index, uuid, mem, util = [x.strip() for x in line.split(',')]
        gpu.append(dict(index=int(index), uuid=uuid, memory_mib=int(mem), utilization=int(util)))
    raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                   '--format=csv,noheader,nounits'],text=True)
    own_gpu_pids = {}
    for line in raw.splitlines():
        uuid, pid, mem = [x.strip() for x in line.split(',')]
        try:
            p = psutil.Process(int(pid))
            if p.username() == username:
                own_gpu_pids[int(pid)] = dict(gpu_uuid=uuid, memory_mib=int(mem))
        except psutil.Error:
            pass
    jobs, roots = [], []
    for alg in ['fql','gfp']:
        for seed in [0,1]:
            roots.append((alg.upper()+'-%d' % seed,OLD/'runs'/('formal_%s_seed%d' % (alg,seed))))
    roots.append(('floq-P1',ROOT/'runs/p1_seed0'))
    for seed in [0,1]:
        roots.append(('floq-%d' % seed,ROOT/'runs'/('formal_seed%d' % seed)))
    known_pids = set()
    for label, directory in roots:
        status = read(directory/'status.json')
        progress = read(directory/'training_progress.json')
        if not directory.exists():
            jobs.append(dict(job=label,status='NOT_STARTED',path=str(directory)))
            continue
        pids = set(status.get('pids',[]))
        for name in ['pid','python_pid','training_pid','supervisor_pid','process_pid']:
            if isinstance(status.get(name),int):
                pids.add(status[name])
        live = []
        for pid in sorted(pids):
            try:
                p = psutil.Process(pid)
                if p.username() == username:
                    x = cpu_process(p)
                    if x:
                        live.append(x)
                        known_pids.add(pid)
            except psutil.Error:
                pass
        key_fields = ['status','stage','updates','completed_updates','step','returncode','start_epoch',
                      'end_epoch','gpu','gpu_uuid','gpu_peak_mib','rss_peak_bytes','gpu_residence_hours']
        ev_lines = directory/'native_evaluations.jsonl'
        latest_eval = None
        if ev_lines.exists():
            lines = ev_lines.read_text().splitlines()
            if lines:
                try:
                    latest_eval=json.loads(lines[-1])
                except ValueError:
                    pass
        error = read(directory/'entry_failure.json') or read(directory/'numerical_failure.json')
        jobs.append(dict(job=label,path=str(directory),
                         status={k:status[k] for k in key_fields if k in status},
                         progress=progress, live_processes=live,
                         latest_native_evaluation=latest_eval,
                         failure=error if error else None))
    additional=[]
    for p in psutil.process_iter(['pid','username','name']):
        try:
            if p.info['username'] != username or p.pid == os.getpid():
                continue
            cmd = p.cmdline()
            programs = [s for s in cmd if s.endswith('.py')]
            appears_training = any(any(k in pathlib.Path(s).name.lower() for k in
                                     ['train','main','learn','eval']) for s in programs)
            if p.pid in own_gpu_pids or appears_training:
                association = next((label for label,dir in roots if any(str(dir) in s for s in cmd)),None)
                x = cpu_process(p)
                if x:
                    x.update(script=pathlib.Path(programs[0]).name if programs else p.info['name'],
                             associated_job=association, gpu=own_gpu_pids.get(p.pid))
                    additional.append(x)
        except (psutil.Error,OSError):
            pass
    result=dict(epoch=now,host='lab-xushijie3',username=username,jobs=jobs,
                own_training_evaluation_and_gpu_processes=additional,gpus=gpu,
                available_ram_bytes=psutil.virtual_memory().available,
                disk_free_bytes=psutil.disk_usage(str(ROOT)).free,
                floq_queue=read(ROOT/'receipts/queue.json'),
                floq_manager_failure=read(ROOT/'receipts/manager_failure.json'),
                qflow_formal_status='PAPER_REPRODUCTION_BLOCKED',
                inspection_only_no_process_or_config_changes=True)
    d=ROOT/'receipts/monitor'
    d.mkdir(exist_ok=True)
    stamp=time.strftime('%Y%m%d_%H%M%S',time.localtime(now))
    p=d/(stamp+'.json')
    p.write_text(json.dumps(result,indent=2))
    (d/'latest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
