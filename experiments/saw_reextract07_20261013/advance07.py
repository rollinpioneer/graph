"""Advance only previously authorized stages after registered scientific gates."""
import os,sys,json,time,subprocess,hashlib,argparse,resource,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert os.sched_getaffinity(0)=={75}
parser=argparse.ArgumentParser();parser.add_argument('--i-pid',type=int,required=True);args=parser.parse_args()
start=time.time();ledger=ROOT/'receipts/ADVANCE_PROCESS.json';assert not ledger.exists()
jobs=[]
def read(p):return json.loads(Path(p).read_text())
def write(status,**more):
    x=dict(status=status,pid=os.getpid(),start_epoch=start,elapsed_seconds=time.time()-start,jobs=jobs,CPU_affinity=[75],phase_I_workflow_pid=args.i_pid,no_automatic_formal_restart=True,process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,**more)
    q=ledger.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(ledger)
def run(script,argv):
    log=ROOT/'runs'/('advance_'+script.replace('.py','')+'_'+('_'.join(argv) or 'only')+'.log')
    epoch=time.time()
    with open(log,'x') as f:p=subprocess.Popen([sys.executable,script]+argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    job=dict(script=script,arguments=argv,pid=p.pid,start_epoch=epoch,code_sha256=hashlib.sha256((ROOT/script).read_bytes()).hexdigest(),log=str(log));jobs.append(job);write('RUNNING_SUBPROCESS')
    rc=p.wait();job.update(exit_code=rc,elapsed_seconds=time.time()-epoch,child_exited=True)
    if rc:raise RuntimeError('NO_AUTOMATIC_RESTART:'+script+':'+str(rc))
    write('SUBPROCESS_COMPLETED')
try:
    auth=read(ROOT/'receipts/AUTHORIZATION.json');assert auth['I'] and auth['II'] and auth['time_cost_limits_removed'] and not auth['X']
    write('WAITING_ALL_I_TRAINING_AND_EVALUATION')
    while True:
        m=read(ROOT/'receipts/I_WORKFLOW_MONITOR.json')
        if m['status']=='FAILED':raise RuntimeError('I_WORKFLOW_FAILED_NO_FORMAL_RESTART')
        if m['status']=='COMPLETE' and not Path('/proc',str(args.i_pid)).exists():break
        time.sleep(5)
    assert m['all_owned_children_exited'] and len(m['jobs'])==6 and all(x['status']=='COMPLETE_500K' for x in m['jobs'])
    run('analyze07.py',['I'])
    gate=read(ROOT/'results/I_GATE.json');write('I_SCIENTIFIC_GATE_CALCULATED',H_R_pass=gate['H_R_pass'],raw_gap_pp=gate['raw_gap_pp'])
    if gate['H_R_pass']:
        adm=read(ROOT/'results/P_continuation/P_ADMISSION.json');assert adm['technical_gates_pass'] and adm['manipulation_all_blocks_pass']
        assert not (ROOT/'receipts/II_WORKFLOW_MONITOR.json').exists()
        run('workflow07.py',['II'])
        run('analyze07.py',['FINAL'])
    run('replay_logs07.py',[])
    run('finish07.py',[])
    write('COMPLETE',H_R_pass=gate['H_R_pass'],all_owned_children_exited=True)
    print('ADVANCE_COMPLETE',gate['H_R_pass'],flush=True)
except Exception as exc:
    write('FAILED',error=repr(exc),traceback=traceback.format_exc());raise
