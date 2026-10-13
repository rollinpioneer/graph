"""Durable P1 and two-seed P2 queue. No time or GPU-hour budget.

P2 requires a verified P0 archive plus successful isolated P1. Per the latest
user instruction, rank GPUs with no process owned by this account by current
load and memory use; sharing with other users is allowed. Keep memory headroom,
one own run per GPU, no automatic restart, and no parameter changes.
"""
import os
import sys
import json
import time
import pathlib
import subprocess
import hashlib
import psutil
import pwd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHA = '3d60e638b42bbef018c56cb1199ff37c3470520d'
PYTHON = ROOT / 'envs/floq/bin/python'
BRANCH = 'codex/floq-min-repro-v2-20261013'


def write(path, obj):
    path = pathlib.Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def read(path):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        return {}


def gpu_inventory():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    gpu = {}
    for line in out.splitlines():
        i, uuid, mem, total, util = [x.strip() for x in line.split(',')]
        gpu[int(i)] = dict(uuid=uuid, memory_mib=int(mem), total_mib=int(total), utilization=int(util))
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    for entry in gpu.values():
        entry['processes'] = []
    for line in out.splitlines():
        uuid, pid, mem = [x.strip() for x in line.split(',')]
        for entry in gpu.values():
            if entry['uuid'] == uuid:
                try:
                    owned = psutil.Process(int(pid)).username() == pwd.getpwuid(os.getuid()).pw_name
                except psutil.NoSuchProcess:
                    owned = False
                except psutil.AccessDenied:
                    owned = True  # uncertain ownership cannot satisfy no-own-process rule
                entry['processes'].append(dict(pid=int(pid), memory_mib=int(mem), owned=owned))
    return gpu


def eligible(gpu):
    return not any(p['owned'] for p in gpu['processes']) and gpu['total_mib']-gpu['memory_mib'] >= 8192


def ranked_candidates(inventory, occupied=()):
    return sorted([i for i, g in inventory.items() if i not in occupied and eligible(g)],
                  key=lambda i:(inventory[i]['utilization'], inventory[i]['memory_mib'], i))


def cpu_pool():
    allowed = psutil.Process().cpu_affinity()
    reserved = {104, 51, 99, 87, 105, 97, 85, 46, 33, 49, 53, 55, 89, 101, 110, 30}
    usage = psutil.cpu_percent(interval=2, percpu=True)
    return sorted([c for c in allowed if c not in reserved], key=lambda c: usage[c])[:8]


def command(seed, phase, run):
    short = phase == 'P1'
    return [str(PYTHON), str(ROOT / 'scripts/native_entry.py'),
            '--env_name=antmaze-large-navigate-singletask-v0', '--agent=agents/floq.py',
            '--agent.alpha=10', '--offline_steps=%d' % (5000 if short else 2000000),
            '--online_steps=0', '--seed=%d' % seed, '--save_dir=' + str(run / 'outputs'),
            '--project_name=CP_FLOQ', '--run_group=FLOQ_MIN_REPRO_V2_' + phase,
            '--eval_interval=20000', '--eval_episodes=50',
            '--save_interval=%d' % (5000 if short else 1000000)]


def launch(seed, phase, gpu, cpus):
    run = ROOT / 'runs' / ('p1_seed0' if phase == 'P1' else 'formal_seed%d' % seed)
    run.mkdir(exist_ok=False)
    cmd = command(seed, phase, run)
    env = os.environ.copy()
    env.update(HOME=str(ROOT / 'home'), CUDA_VISIBLE_DEVICES=str(gpu),
               XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl',
               WANDB_MODE='offline', PYTHONUNBUFFERED='1', MPLBACKEND='Agg',
               FLOQ_RUN_DIR=str(run), FLOQ_PHASE=phase, FLOQ_SOURCE_COMMIT=SHA)
    env.pop('JAX_DEFAULT_MATMUL_PRECISION', None)
    env.pop('JAX_ENABLE_X64', None)
    env['LD_LIBRARY_PATH'] = ':'.join(str(p) for p in
        (ROOT / 'envs/floq/lib/python3.8/site-packages/nvidia').glob('*/lib'))
    full = ['/usr/bin/time', '-v', '-o', str(run / 'cpu_time.txt'),
            'taskset', '-c', ','.join(map(str, cpus))] + cmd
    with (run / 'stdout.log').open('x') as out:
        p = subprocess.Popen(full, cwd=ROOT / 'source', env=env, stdout=out,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
    receipt = dict(phase=phase, seed=seed, status='STARTING', command=full, gpu=gpu,
                   gpu_uuid=gpu_inventory()[gpu]['uuid'], cpu_affinity=cpus,
                   supervisor_pid=p.pid, source_commit=SHA, start_epoch=time.time(),
                   prescribed_updates=5000 if phase == 'P1' else 2000000,
                   branch=BRANCH, environment_overrides={k: env[k] for k in [
                       'HOME', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH',
                       'XLA_PYTHON_CLIENT_PREALLOCATE', 'OMP_NUM_THREADS',
                       'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'WANDB_MODE']})
    write(run / 'launch.json', receipt)
    write(run / 'status.json', receipt)
    return dict(p=p, run=run, receipt=receipt, peak_gpu=0, peak_rss=0, cpu_seconds=0,
                first_gpu_epoch=None, last_gpu_epoch=None, gpu_sample_count=0)


def observe(job, inventory):
    r = job['receipt']
    tree = []
    try:
        parent = psutil.Process(job['p'].pid)
        tree = [parent] + parent.children(recursive=True)
    except psutil.Error:
        pass
    pids, rss, cpu = [], 0, 0
    for proc in tree:
        try:
            pids.append(proc.pid)
            rss += proc.memory_info().rss
            t = proc.cpu_times()
            cpu += t.user + t.system
        except psutil.Error:
            pass
    own_gpu = sum(p['memory_mib'] for p in inventory[r['gpu']]['processes'] if p['pid'] in pids)
    now = time.time()
    if own_gpu:
        job['first_gpu_epoch'] = job['first_gpu_epoch'] or now
        job['last_gpu_epoch'] = now
        job['gpu_sample_count'] += 1
    job['peak_gpu'] = max(job['peak_gpu'], own_gpu)
    job['peak_rss'] = max(job['peak_rss'], rss)
    job['cpu_seconds'] = max(job['cpu_seconds'], cpu)
    progress = read(job['run'] / 'training_progress.json')
    status = dict(r, status=progress.get('phase', 'STARTING'),
                  updates=progress.get('updates', 0), epoch=now, pids=pids,
                  gpu_memory_mib=own_gpu, gpu_peak_mib=job['peak_gpu'],
                  rss_peak_bytes=job['peak_rss'], cpu_seconds_observed=job['cpu_seconds'],
                  first_gpu_epoch=job['first_gpu_epoch'], last_gpu_epoch=job['last_gpu_epoch'],
                  gpu_residence_hours=(0 if job['first_gpu_epoch'] is None else
                                      (job['last_gpu_epoch'] - job['first_gpu_epoch']) / 3600),
                  process_lifetime_hours=(now-r['start_epoch'])/3600)
    write(job['run'] / 'status.json', status)
    with (job['run'] / 'resource_samples.jsonl').open('a') as f:
        f.write(json.dumps(dict(epoch=now, pids=pids, gpu_memory_mib=own_gpu,
                               process_rss_bytes=rss, process_cpu_seconds=cpu))+'\n')
    return status


def finish(job):
    code = job['p'].returncode
    run, r = job['run'], job['receipt']
    final = read(run / 'entry_final_state.json')
    complete = code == 0 and final.get('updates') == r['prescribed_updates']
    indices = []
    if (run / 'checkpoint_index.jsonl').exists():
        indices = [json.loads(x) for x in (run / 'checkpoint_index.jsonl').read_text().splitlines()]
    complete = complete and any(x['step'] == r['prescribed_updates'] for x in indices)
    if r['phase'] == 'P1':
        complete = complete and read(run / 'checkpoint_roundtrip.json').get('passed', False)
        complete = complete and bool(read(run / 'initialization_probes.json'))
    else:
        ev = run / 'native_evaluations.jsonl'
        complete = complete and ev.exists() and any(
            json.loads(x).get('step') == 2000000 and json.loads(x).get('episodes') == 50
            for x in ev.read_text().splitlines())
    status = read(run / 'status.json')
    status.update(status='P1_PASSED' if complete and r['phase']=='P1' else
                        ('COMPLETE' if complete else 'TECHNICAL_FAILURE'),
                  returncode=code, end_epoch=time.time(),
                  completed_updates=final.get('updates', 0), checkpoints=indices)
    write(run / 'status.json', status)
    return complete


def main():
    current = subprocess.check_output(['git', '-C', str(ROOT/'source'), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT/'source'), 'status', '--porcelain'], text=True).strip()
    assert current == SHA and not dirty
    assert read(ROOT/'receipts/gpu_probe.json').get('returncode') == 0
    cpus = cpu_pool()
    assert len(cpus) == 8 and psutil.virtual_memory().available > 32*1024**3
    write(ROOT/'receipts/manager.json', dict(pid=os.getpid(), start_epoch=time.time(),
                                          cpu_pool=cpus, no_time_budget=True,
                                          official_source=current, formal_updates=2000000))
    candidates = ranked_candidates(gpu_inventory())
    if not candidates:
        write(ROOT/'receipts/queue.json',dict(status='NO_SAFE_GPU_WITHOUT_OWN_PROCESS',epoch=time.time()))
        return 1
    p1 = launch(0, 'P1', candidates[0], cpus[:4])
    while p1['p'].poll() is None:
        observe(p1, gpu_inventory())
        time.sleep(2)
    observe(p1, gpu_inventory())
    if not finish(p1):
        write(ROOT/'receipts/queue.json', dict(status='P1_FAILED_P2_BLOCKED', epoch=time.time()))
        return 1
    while not read(ROOT/'receipts/p0_release.json').get('verified_remote_commit'):
        write(ROOT/'receipts/queue.json', dict(status='P1_PASSED_WAITING_P0_ARCHIVE', epoch=time.time()))
        time.sleep(15)
    pending, active, complete = [0, 1], [], []
    while pending or active:
        inv = gpu_inventory()
        for job in list(active):
            observe(job, inv)
            if job['p'].poll() is not None:
                ok = finish(job)
                complete.append(dict(seed=job['receipt']['seed'], passed=ok))
                active.remove(job)
                if not ok:
                    pending.clear()
        if pending:
            for gpu in ranked_candidates(inv, [j['receipt']['gpu'] for j in active]):
                slot = len(active)
                job = launch(pending.pop(0), 'P2', gpu, cpus[4*slot:4*slot+4])
                active.append(job)
                if not pending:
                    break
        write(ROOT/'receipts/queue.json', dict(status='RUNNING' if active else 'WAITING_RESOURCES',
                                              epoch=time.time(), pending_seeds=pending,
                                              active=[{'seed':j['receipt']['seed'], 'gpu':j['receipt']['gpu'],
                                                       'supervisor_pid':j['p'].pid} for j in active],
                                              completed=complete))
        time.sleep(5)
    write(ROOT/'receipts/queue.json', dict(status='ALL_COMPLETE' if all(c['passed'] for c in complete)
                                          and len(complete)==2 else 'TECHNICAL_FAILURE',
                                          epoch=time.time(), completed=complete))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except BaseException as e:
        if not isinstance(e, SystemExit):
            import traceback
            write(ROOT/'receipts/manager_failure.json', dict(error=str(e), traceback=traceback.format_exc()))
        raise
