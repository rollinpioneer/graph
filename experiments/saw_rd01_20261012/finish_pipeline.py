"""Wait for both credible frozen baselines, then run the CPU-only discovery stage."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parent
started=time.monotonic()
receipt=root/'receipts/CPU_PIPELINE.json'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',MUJOCO_GL='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
python=str(root/'vendor/venv/bin/python')
jobs=[]
while True:
    statuses=[]
    for kind in ('navigate','stitch'):
        fp=root/'results'/f'{kind}_training_status.json'
        if fp.exists(): statuses.append(json.loads(fp.read_text()))
    if len(statuses)==2:
        if not all(s['status']=='SINGLE_SEED_BASELINE' and s['step']==1000000 for s in statuses):
            receipt.write_text(json.dumps(dict(status='ABORTED_BASELINE_INCOMPLETE',baselines=statuses),indent=2)+'\n')
            raise SystemExit(2)
        if all((root/'receipts'/f'formal_{k}_resource.json').exists() for k in ('navigate','stitch')):break
    if time.monotonic()-started>14520:
        receipt.write_text(json.dumps(dict(status='ABORTED_BASELINE_WAIT_TIMEOUT'),indent=2)+'\n');raise SystemExit(3)
    time.sleep(2)


def launch(script,args,label):
    stream=open(root/'logs'/f'{label}.log','w')
    p=subprocess.Popen([python,str(root/script),*args],stdout=stream,stderr=subprocess.STDOUT,env=env)
    return p,stream,label,time.monotonic()


for kind in ('navigate','stitch'):
    jobs.append(launch('final_eval.py',['--kind',kind],f'final_{kind}'))
    jobs.append(launch('diagnose.py',['--kind',kind,'--phase','discovery'],f'discovery_{kind}'))
records=[]
for p,stream,label,start in jobs:
    try: code=p.wait(timeout=1200)
    except subprocess.TimeoutExpired:p.terminate();code=p.wait(timeout=30)
    stream.close();records.append(dict(label=label,pid=p.pid,exit_code=code,wall_upper_seconds=time.monotonic()-start,gpu_seconds=0))
if all(r['exit_code']==0 for r in records):
    for script in ('failure_map.py','input_coverage.py'):
        args=['--mode','build'] if script=='failure_map.py' else []
        p,stream,label,start=launch(script,args,script[:-3])
        code=p.wait(timeout=600);stream.close();records.append(dict(label=label,pid=p.pid,exit_code=code,wall_seconds=time.monotonic()-start,gpu_seconds=0))
receipt.write_text(json.dumps(dict(status='DISCOVERY_DONE' if all(r['exit_code']==0 for r in records) else 'CPU_ANALYSIS_FAILED',jobs=records,wall_seconds=time.monotonic()-started,confirmation='pending frozen intervention decision'),indent=2)+'\n')
print('CPU_PIPELINE_END',receipt.read_text(),flush=True)
