"""One-shot continuation of the finite experiment batch; no scheduled monitor."""
import subprocess,sys
from common import *
start=time.time()
while True:
    x=json.loads((R/'receipts/confirmation_queue.json').read_text())
    if x['status']=='PASS':break
    if x['status'] in ['FAILED_OR_BUDGET_STOP','DRAINING_AFTER_FAILURE']:
        write_json(R/'receipts/confirmation_postprocess.json',dict(status='NOT_RUN_QUEUE_FAILED',queue_status=x['status']));sys.exit(1)
    time.sleep(15)
steps=[]
for name in ['analyze_confirmation.py','diagnose_confirmation.py','finalize_confirmation.py']:
    t=time.time();rc=subprocess.run([sys.executable,str(R/'code'/name)]).returncode
    steps.append(dict(script=name,exit=rc,seconds=time.time()-t))
    if rc:
        write_json(R/'receipts/confirmation_postprocess.json',dict(status='FAILED',steps=steps));sys.exit(rc)
write_json(R/'receipts/confirmation_postprocess.json',dict(status='PASS',steps=steps,wall_seconds_including_dependency_wait=time.time()-start))
