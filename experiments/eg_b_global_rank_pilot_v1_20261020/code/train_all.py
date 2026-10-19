from common import *
import subprocess,concurrent.futures,threading,traceback
assert json.loads((R/'results/pair_coverage.json').read_text())['status']=='PASS'
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
eligible=sorted([tuple(map(int,v.split(','))) for v in raw.splitlines() if v.strip()],key=lambda v:(v[2],-v[1]))
eligible=[v for v in eligible if v[2]<85 and v[1]>=3072];assert eligible,'No currently eligible GPU'
slots=[]
for gpu,mem,util in eligible:
 slots.extend([str(gpu)]*min(3,max(0,(mem-1024)//2048)))
assert slots
assign={seed:slots[i%len(slots)] for i,seed in enumerate(SEEDS)}
write(R/'results/GPU_assignment_training.json',dict(snapshot=raw,paired_seed_GPU=assign,rule='paired M/R same GPU; reserve 1GiB/GPU and2GiB/pair; utilization<85%',maximum_training_processes=min(6,2*len(slots))))
mutex=threading.Lock();ledger=R/'results/resource_jobs.jsonl'
def run(family,seed):
 name=f'train_{family}_{seed}';env=dict(os.environ,CUDA_VISIBLE_DEVICES=assign[seed]);t=time.time()
 with (R/f'results/{name}.log').open('w') as out:
  p=subprocess.Popen([sys.executable,str(R/'code/train_head.py'),'--family',family,'--seed',str(seed)],env=env,stdout=out,stderr=subprocess.STDOUT)
  try:rc=p.wait(timeout=300)
  except subprocess.TimeoutExpired:p.terminate();p.wait();raise RuntimeError(name+' training timeout')
  finally:
   with mutex:
    with ledger.open('a') as f:f.write(json.dumps(dict(name=name,kind='GPU',pid=p.pid,seconds=time.time()-t,returncode=p.returncode,GPU=assign[seed]))+'\n')
 assert rc==0,(name,rc)
 return json.loads((R/f'results/{name}.json').read_text())
try:
 write(R/'results/pipeline_status.json',dict(status='RUNNING',stage='P1_FIRST_FORMAL_HEAD_COST_CHECK',controller_pid=os.getpid()))
 first=run('M',SEEDS[0]);projected=first['seconds']*6
 write(R/'results/training_cost_check.json',dict(first_formal_head_seconds=first['seconds'],six_head_projection=projected,GPU_budget=3600,additional_smoke_training=0))
 assert projected+300<3600,'Projected training/inference exceeds frozen GPU budget'
 tasks=[(f,s) for s in SEEDS for f in ['M','R'] if (f,s)!=('M',SEEDS[0])];n=min(5,2*len(slots))
 write(R/'results/pipeline_status.json',dict(status='RUNNING',stage='P1_REMAINING_FIVE_HEADS',workers=n,controller_pid=os.getpid()))
 with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:results=[first]+list(pool.map(lambda x:run(*x),tasks))
 for s in SEEDS:
  pair=[v for v in results if v['seed']==s];assert len(pair)==2
  for k in ['initial_state_sha256','sample_stream_sha256','schedule_sha256','GPU_slot','torch_version']:assert pair[0][k]==pair[1][k],(s,k)
 assert len(results)==6 and all(v['step']==1000 for v in results)
 write(R/'results/training_complete.json',dict(status='PASS',heads=results,paired_initialization_and_frame_stream=True,training_updates=6000,old_heads_unchanged=True))
 write(R/'results/pipeline_status.json',dict(status='P1_TRAINING_COMPLETE',heads=6,P2_not_started=True))
 print('SIX_HEADS_COMPLETE',flush=True)
except Exception as e:
 write(R/'results/pipeline_status.json',dict(status='TECHNICAL_STOP',error=repr(e),traceback=traceback.format_exc()));raise
