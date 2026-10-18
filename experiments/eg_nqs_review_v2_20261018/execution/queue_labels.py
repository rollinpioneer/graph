"""Finite work queue with a process-occupancy budget; not a scheduled monitor."""
import os,subprocess,time,json,sys
from common import *
role=sys.argv[1] if len(sys.argv)>1 else 'subset'
pending=SPLIT['probe_fit']+SPLIT['probe_calibration'];active={};done=[];start=time.time()
while pending or active:
    for s,(p,t,log) in list(active.items()):
        rc=p.poll()
        if rc is not None:
            log.close();done.append(dict(scene=s,pid=p.pid,exit=rc,wall_seconds=time.time()-t));del active[s]
            if rc:write_json(R/f'receipts/label_queue_{role}.json',dict(status='FAILED',done=done));raise RuntimeError(('label failed',s,rc))
    hours=(sum(x['wall_seconds'] for x in done)+sum(time.time()-x[1] for x in active.values()))/3600
    if hours>25:
        # Stop launching; running own jobs are allowed to finish before reporting the budget block.
        pending=[]
        if not active:raise RuntimeError('LABEL_QUEUE_BUDGET_EXCEEDED')
    for s in pending.copy():
        if len(active)>=8:break
        if not all((R/f'exports/{role}/scene_{s:04d}/{f:04d}.npz').exists() for f in frames(s)):continue
        log=(R/f'logs/label_{role}_{s}.log').open('w')
        p=subprocess.Popen([sys.executable,str(R/'code/label.py'),'--scene',str(s),'--role',role],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[s]=(p,time.time(),log);pending.remove(s)
    write_json(R/f'receipts/label_queue_{role}.json',dict(status='RUNNING' if active or pending else 'PASS',pending=pending,active=[dict(scene=s,pid=x[0].pid) for s,x in active.items()],done=done,process_hours=hours,wall_seconds=time.time()-start))
    if pending or active:time.sleep(5)
