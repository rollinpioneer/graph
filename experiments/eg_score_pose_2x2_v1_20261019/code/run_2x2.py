import os, sys, json, time, subprocess, concurrent.futures, resource
from pathlib import Path

R = Path(os.environ['P2_ROOT'])
cfg = json.loads((R / 'frozen_config.json').read_text())
micro = json.loads((R / 'evaluation/micro/0037/COMPLETE.json').read_text())
assert micro['status'] == 'PASS' and len(micro['official_micro_checks']) == 6
start = time.time()
def save(status, **extra):
    value = dict(status=status, new_trainings=0, gpu_jobs=0, workers=7, controller=1, seconds=time.time()-start, **extra)
    p = R / 'results/pipeline_status.json'
    p.write_text(json.dumps(value, indent=2)+'\n')
def one(scene):
    t = time.time()
    with open(R / f'logs/scene_{scene:04d}.log', 'w') as log:
        result = subprocess.run([sys.executable, str(R / 'code/evaluate_2x2.py'), '--scene', str(scene)], stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, (scene, result.returncode)
    return dict(scene=scene, seconds=time.time()-t)
done = []
save('RUNNING')
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        futures = [pool.submit(one, scene) for scene in cfg['development']]
        for future in concurrent.futures.as_completed(futures):
            done.append(future.result())
            save('RUNNING', done=done)
    save('EVALUATION_COMPLETE', done=done, child_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime)
except Exception as e:
    save('TECHNICAL_STOP', error=repr(e), done=done)
    raise
