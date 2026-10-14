"""Results remain per host/task; never rank incompatible tasks against each other."""
import json,csv,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
rows=[];cost=[]
for j in json.loads((ROOT/'configs/jobs.json').read_text()):
    d=ROOT/'runs'/j['job'];result=dict(job=j['job'],host=j['host'],task=j['task'],seed=j['seed'],model=j['job'][:2])
    statuses=json.loads((d/'status.json').read_text()) if (d/'status.json').exists() else {'status':'NOT_EXECUTED'}
    result['status']=statuses['status'];result['updates']=statuses.get('updates',0)
    matched=[];native=[]
    for step in [800000,900000,1000000]:
        p=d/'matched_verified'/f'{step}.json'
        if p.exists():
            x=json.loads(p.read_text());result[f'matched_success_{step}']=x['success_rate'];matched.append(x['success_rate'])
            result[f'matched_return_{step}']=x['mean_return']
        p=d/'native'/f'{step}_Actor.json'
        if p.exists():
            x=json.loads(p.read_text());value=x['stats'].get('success')
            result[f'native_success_{step}']=value
            if value is not None:native.append(value)
    result['matched_last3_mean']=sum(matched)/3 if len(matched)==3 else None
    result['native_last3_mean']=sum(native)/3 if len(native)==3 else None
    result['two_seed_preliminary_only']=True
    rows.append(result)
    exits=[json.loads((d/f'{ph}_exit.json').read_text()) for ph in ['T1','T2','T3'] if (d/f'{ph}_exit.json').exists()]
    cost.append(dict(job=j['job'],gpu_resident_hours=sum(x['gpu_resident_hours'] for x in exits),
        cpu_seconds=sum(x['resources']['cpu_seconds'] for x in exits),
        gpu_peak_mib=max([x['resources']['gpu_memory_mib'] for x in exits] or [0]),
        rss_peak_kb=max([x['resources']['rss_kb'] for x in exits] or [0]),exits=[x['exit_code'] for x in exits]))
for name,data in [('results',rows),('costs',cost)]:
    (ROOT/'reports'/f'{name}.json').write_text(json.dumps(data,indent=2))
    keys=sorted(set(k for x in data for k in x))
    with (ROOT/'reports'/f'{name}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
finished=(ROOT/'receipts/manager_finished.json').exists() and (ROOT/'receipts/evaluation_manager_finished.json').exists()
(ROOT/'reports/final_review.md').write_text('# Comparative batch review\n\nStatus: '+('ALL_REGISTERED_JOBS_ENDED' if finished else 'RUNNING')+
    '\n\nCompare G0/G1 only on HumanoidMaze and D0/D1 only on Cube-double. Two seeds are preliminary evidence. '
    'Native and matched evaluations are separate. No best-checkpoint selection, no tuning, no expansion. '
    'Author multi-task aggregates cannot be equated with these single-task results.\n')
print('SUMMARY_WRITTEN',finished,flush=True)
