"""Only the reset flow differs; no training and no formal score peeking."""
import argparse
from eval_core import *
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['navigate','stitch'],required=True);a=p.parse_args()
assert json.load(open(ROOT/'results/A_AUDIT.json'))['status']=='PASS'
size=json.load(open(ROOT/'SAMPLE_SIZE.json'));start=time.monotonic();cpu=time.process_time();agent,identity=load(a.kind)
paths=[]
for base in size['streams']:
    for condition in ('O','U'):
        env=env_new(a.kind);setup_stream(env,base);rows=[];raw=[];t=time.monotonic()
        for task in range(1,6):
            for ep in range(50):
                row,trajectory=episode(agent,env,base,task,ep,condition);row['model']=a.kind;rows.append(row);raw.append(trajectory)
            write(ROOT/'receipts'/f'B_{a.kind}_progress.json',dict(stream=base,condition=condition,task_completed=task,episodes_completed=len(rows),wall_seconds=time.monotonic()-start,gpu_seconds=0))
            print('B_PROGRESS',a.kind,base,condition,task,len(rows),flush=True)
        env.close();name=f'B_{a.kind}_{condition}_{base}';csv_rows(ROOT/'results'/f'{name}.csv',rows)
        dump(ROOT/'runs'/f'{name}.pkl',dict(checkpoint=identity,rows=rows,trajectories=raw))
        paths.append(dict(condition=condition,stream=base,episodes=len(rows),wall_seconds=time.monotonic()-t,raw_path=str(ROOT/'runs'/f'{name}.pkl')))
assert sha(identity['path'])==identity['sha256']
write(ROOT/'receipts'/f'B_{a.kind}_DONE.json',dict(status='COMPLETED',identity=identity,conditions=paths,formal_episodes=sum(x['episodes'] for x in paths),
       wall_seconds=time.monotonic()-start,cpu_process_seconds=time.process_time()-cpu,gpu_seconds=0,training_updates=0))
print('B_MODEL_DONE',a.kind,flush=True)
