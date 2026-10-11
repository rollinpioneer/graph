"""Launch one isolated bounded job; reserve before fork and select current least busy GPU."""
import datetime,json,pathlib,subprocess,sys,os,time
ROOT=pathlib.Path(__file__).resolve().parent
OLD=pathlib.Path('/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011')
name,seconds,*args=sys.argv[1:];seconds=float(seconds)
(ROOT/'logs').mkdir(exist_ok=True);(ROOT/'runs').mkdir(exist_ok=True)
used=sum(json.loads(f.read_text())['charged_gpu_seconds']+2 for f in (ROOT/'runs').glob('*/guard_result.json'))
active=[json.loads(f.read_text()) for f in (ROOT/'logs').glob('*_reservation.json') if not (ROOT/'runs'/f.name.replace('_reservation.json','')/'guard_result.json').exists()]
assert used+sum(x['seconds'] for x in active)+seconds<=6900,('reservation exceeds total budget',used,active)
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
rows=[list(map(int,line.split(','))) for line in raw.strip().splitlines()]
occupied={x['gpu'] for x in active}
available=[r for r in rows if r[0] not in occupied]
chosen=min(available or rows,key=lambda r:(r[1],r[2]))[0]
reservation={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'name':name,'seconds':seconds,'gpu':chosen,'snapshot':rows,
             'criterion':'least current memory, then utilization; separate primary GPU for independent active jobs'}
(ROOT/'logs'/(name+'_reservation.json')).write_text(json.dumps(reservation,indent=2))
cmd=[str(OLD/'env/bin/python'),str(ROOT/'guard.py'),'--out',str(ROOT/'runs'/name),'--seconds',str(seconds),'--gpu',str(chosen),'--','bash',str(ROOT/'run_job_validation.sh'),'--out',str(ROOT/'runs'/name/'data')]+args
with (ROOT/'logs'/(name+'_guard.log')).open('w') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({'name':name,'guard_pid':child.pid,'gpu':chosen,'reservation_seconds':seconds}),flush=True)
