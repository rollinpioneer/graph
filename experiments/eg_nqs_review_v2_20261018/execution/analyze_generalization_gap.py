"""Post-hoc descriptive checks on frozen results; no model or metric changes."""
from common import *
import csv
out=R/'results';seeds=[20261018,20261019,20261020]
rows=[]
def comparisons(data,role,scope):
    scenes=sorted(set(int(x['scene']) for x in data))
    for metric in ['ap','fixed_pool_ap','top1_valid','top1_collision']:
        groups={}
        for x in data:
            val=x[metric];val=float(val) if val not in ['True','False'] else float(val=='True')
            groups.setdefault((int(x['scene']),x['configuration']),[]).append(val)
        diff=np.array([[np.mean(groups[i,f'A_{seed_}'])-np.mean(groups[i,f'B6_{seed_}']) for i in scenes] for seed_ in seeds])
        rows.append(dict(role=role,scope=scope,metric=metric,mean=float(diff.mean()),seed_mean=diff.mean(1).tolist(),positive_scenes=int((diff.mean(0)>0).sum()),scene_mean={str(i):float(v) for i,v in zip(scenes,diff.mean(0))}))
for role in ['subset','official']:
    data=list(csv.DictReader(open(out/f'{role}_confirmation_per_frame.csv')))
    comparisons(data,role,'all256')
    comparisons([x for x in data if int(x['frame']) in SPLIT['development_frames']],role,'matched32')
comparisons(list(csv.DictReader(open(out/'subset_per_scene.csv'))),'subset','development32')
fit=[];support=[]
for role in ['subset','official']:
    for seed_ in seeds:
        receipt=json.loads((R/f'receipts/fit_{role}_{seed_}.json').read_text())
        for v in receipt['runs']:
            first,last=v['history'][0],v['history'][-1]
            fit.append(dict(role=role,arm=v['arm'],seed=seed_,first_step=first['step'],first_train_mse=first['train_mse'],first_cal_mse=first['cal_mse'],final_train_mse=last['train_mse'],final_cal_mse=last['cal_mse']))
            if seed_==seeds[0] and v['arm']=='A':
                for group,key in [('fit','fit_scene_counts'),('calibration','cal_scene_counts')]:
                    counts=sorted([x['n'] for x in v[key]],reverse=True)
                    support.append(dict(role=role,group=group,total=sum(counts),scenes=len(counts),zero_scenes=counts.count(0),top4_fraction=sum(counts[:4])/sum(counts)))
write_json(out/'generalization_gap_diagnostics.json',dict(status='COMPLETE',post_hoc=True,metric_units='AP and fixed_pool_ap in percentage points; top1 metrics in fractions',comparisons=rows,fit_history=fit,support_concentration=support))
print(json.dumps(dict(comparisons=[{k:v for k,v in x.items() if k!='scene_mean'} for x in rows],fit_history=fit,support=support),indent=2))
