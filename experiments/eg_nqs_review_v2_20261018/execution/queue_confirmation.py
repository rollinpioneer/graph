"""Run the finite registered confirmation batch, at most 32 CPU workers; supports adopting existing workers."""
import subprocess,sys
from common import *
assert json.loads((R/'results/subset_development_decision.json').read_text())['continue_confirmation']
assert json.loads((R/'results/confirmation_duplicate_audit.json').read_text())['status']=='COMPLETE_BEFORE_PREDICTIONS'
for role in ['subset','official']:
    assert json.loads((R/f'receipts/eval_cache_validation_{role}.json').read_text())['status']=='PASS'
    for s in [20261018,20261019,20261020]:assert json.loads((R/f'receipts/fit_{role}_{s}.json').read_text())['status']=='PASS'
jobs=[(role,s) for scenes in SPLIT['official_confirmation'].values() for s in scenes for role in ['subset','official']]
pending=[(role,s) for role,s in jobs if not (R/f'evaluation/{role}/scene_{s:04d}/COMPLETE.json').exists()]
active={};done=[];start=time.time();failed=False
adopt_path=os.environ.get('NQS_ADOPT_QUEUE')
class Adopted:
    def __init__(self,pid,role,scene):
        self.pid=pid;self.role=role;self.scene=scene
    def poll(self):
        proc=Path(f'/proc/{self.pid}')
        try:
            stat=(proc/'stat').read_text().split()
            cmd=(proc/'cmdline').read_bytes().replace(b'\x00',b' ').decode()
            if stat[2]!='Z' and str(R/'code/evaluate.py') in cmd:return None
        except FileNotFoundError:pass
        return 0 if (R/f'evaluation/{self.role}/scene_{self.scene:04d}/COMPLETE.json').exists() else 1
if adopt_path:
    prior=json.loads(Path(adopt_path).read_text())
    pending=[tuple(x) for x in prior['pending']];done=prior['done']
    start=time.time()-prior['wall_seconds']
    uptime=float(Path('/proc/uptime').read_text().split()[0]);hz=os.sysconf('SC_CLK_TCK')
    for x in prior['active']:
        pid=x['pid'];role=x['role'];s=x['scene']
        try:
            stat=Path(f'/proc/{pid}/stat').read_text().split();t=time.time()-(uptime-int(stat[21])/hz)
        except FileNotFoundError:t=time.time()
        active[role,s]=(Adopted(pid,role,s),t,open(os.devnull,'w'))
baseline_hours=json.loads((R/'results/resource_summary.json').read_text())['total_worker_hours']
while pending or active:
    for key,(p,t,log) in list(active.items()):
        rc=p.poll()
        if rc is not None:
            log.close();done.append(dict(role=key[0],scene=key[1],pid=p.pid,exit=rc,wall_seconds=time.time()-t));del active[key]
            if rc:failed=True
    hours=(sum(x['wall_seconds'] for x in done)+sum(time.time()-v[1] for v in active.values()))/3600+baseline_hours
    if hours>=175:failed=True
    if failed and not active:
        write_json(R/'receipts/confirmation_queue.json',dict(status='FAILED_OR_BUDGET_STOP',pending=pending,done=done,process_hours=hours,wall_seconds=time.time()-start));sys.exit(1)
    if not failed:
        for role,s in pending.copy():
            if len(active)>=32:break
            if not all((R/f'exports/{role}/scene_{s:04d}/{f:04d}.npz').exists() for f in frames(s)):continue
            logfile=R/f'logs/confirmation_{role}_{s}.log';log=logfile.open('a');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
            p=subprocess.Popen([sys.executable,str(R/'code/evaluate.py'),'--role',role,'--scene',str(s)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[role,s]=(p,time.time(),log);pending.remove((role,s))
    write_json(R/'receipts/confirmation_queue.json',dict(status='DRAINING_AFTER_FAILURE' if failed else ('RUNNING' if pending or active else 'PASS'),pending=pending,active=[dict(role=k[0],scene=k[1],pid=v[0].pid) for k,v in active.items()],done=done,process_hours=hours,wall_seconds=time.time()-start))
    if pending or active:time.sleep(10)
