"""Summarize all frozen development configurations and apply registered gates."""
import csv, argparse
from common import *
p=argparse.ArgumentParser();p.add_argument('--role',default='subset');a=p.parse_args()
out=R/'results';out.mkdir(exist_ok=True)
records=[]
for s in SPLIT['probe_development']:
    folder=R/f'evaluation/{a.role}/scene_{s:04d}'
    assert json.loads((folder/'COMPLETE.json').read_text())['status']=='PASS'
    for f in frames(s):
        rows=json.loads((folder/f'{f:04d}.json').read_text())
        assert len(rows)==20 and len({v['configuration'] for v in rows})==20
        records.extend(rows)
assert len(records)==5120
def csvout(name,rows):
    with open(out/name,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
csvout(f'{a.role}_per_frame.csv',records)
scenes=SPLIT['probe_development'];names=list(dict.fromkeys(v['configuration'] for v in records))
metrics=['ap','ap04','ap08','fixed_pool_ap','n','top1_valid','top1_collision','top1_empty']
per_scene=[];lookup={}
for s in scenes:
    for name in names:
        rows=[r for r in records if r['scene']==s and r['configuration']==name]
        assert len(rows)==32
        v=dict(scene=s,configuration=name,**{m:float(np.mean([r[m] for r in rows])) for m in metrics})
        corrs=[r['spearman'] for r in rows if r['spearman'] is not None]
        v['spearman_mean']=float(np.mean(corrs)) if corrs else None;v['spearman_valid_frames']=len(corrs)
        lookup[s,name]=v;per_scene.append(v)
csvout(f'{a.role}_per_scene.csv',per_scene)
summary=[]
for name in names:
    summary.append(dict(configuration=name,**{m:float(np.mean([lookup[s,name][m] for s in scenes])) for m in metrics}))
csvout(f'{a.role}_configuration_summary.csv',summary)
seeds=[20261018,20261019,20261020]
def difference(left,right,suffix=''):
    arr=np.array([[lookup[s,f'{left}_{seed_}{suffix}']['ap']-lookup[s,(f'{right}_{seed_}' if right!='R0' else 'R0')+suffix]['ap'] for s in scenes] for seed_ in seeds])
    sm=arr.mean(0);seedmeans=arr.mean(1)
    # Exploratory scene bootstrap, not confirmation or a frame-level test.
    rng=np.random.default_rng(20261018);bs=sm[rng.integers(0,len(sm),(10000,len(sm)))].mean(1)
    return dict(mean_pp=float(sm.mean()),seed_mean_pp=seedmeans.tolist(),scene_mean_pp={str(s):float(d) for s,d in zip(scenes,sm)},positive_seeds=int((seedmeans>0).sum()),positive_scenes=int((sm>0).sum()),exploratory_scene_ci95_pp=np.quantile(bs,[.025,.975]).tolist(),pass_gate=bool(sm.mean()>=1 and (seedmeans>0).all() and (sm>0).sum()>=6))
effects={}
for suffix in ['', '_CD']:
    for left,right in [('A','B6'),('A','E'),('A','R0'),('B6','R0'),('E','R0')]:effects[left+'-'+right+suffix]=difference(left,right,suffix)
gate=effects['A-B6']['pass_gate'] or effects['A-R0_CD']['pass_gate']
write_json(out/f'{a.role}_development_decision.json',dict(status='COMPLETE',frames=256,scenes=scenes,configurations=20,probe_seeds=seeds,protocol='sampled-development-native-AP; not full official AP',effects=effects,continue_confirmation=gate,decision='CONTINUE_FROZEN_CONFIRMATION' if gate else 'STOP_EXPLORATION_NO_GATE',summary=summary))
counts=[];targets=[]
for group in ['probe_fit','probe_calibration']:
    for s in SPLIT[group]:
        n=0;sampled=0;allmatch=0;gaps=[];ys=[]
        for f in frames(s):
            with np.load(R/f'labels/{a.role}/scene_{s:04d}/{f:04d}.npz') as z,np.load(R/f'exports/{a.role}/scene_{s:04d}/{f:04d}.npz') as e:
                n+=len(z['ids']);sampled+=len(e['sample_ids']);allmatch+=int(e['M'].sum())
                gaps.extend((z['B6']-z['A']).tolist());ys.extend(np.column_stack([z[k] for k in ['E','B6','A']]).tolist())
        ar=np.array(ys).reshape(-1,3);gap=np.array(gaps)
        counts.append(dict(group=group,scene=s,sampled=sampled,matched=n,coverage=n/sampled,all_1024_matched=allmatch,B6_gt_A=int((gap>0).sum()),mean_B6_minus_A=float(gap.mean()) if n else None,E_mean=float(ar[:,0].mean()) if n else None,B6_mean=float(ar[:,1].mean()) if n else None,A_mean=float(ar[:,2].mean()) if n else None))
csvout(f'{a.role}_label_coverage.csv',counts)
fits=[]
for seed_ in seeds:
    receipt=json.loads((R/f'receipts/fit_{a.role}_{seed_}.json').read_text())
    assert receipt['status']=='PASS'
    assert len({v['initial_hash'] for v in receipt['runs']})==1 and len({v['batch_stream_hash'] for v in receipt['runs']})==1
    for run in receipt['runs']:
        last=run['history'][-1]
        fits.append({k:run[k] for k in ['arm','seed','fit_samples','cal_samples','updates','initial_hash','batch_stream_hash','checkpoint_sha256','checkpoint_reload_exact','wall_seconds','constant_cal_mse']}|dict(train_mse=last['train_mse'],cal_mse=last['cal_mse'],cal_beats_constant=last['cal_mse']<run['constant_cal_mse']))
csvout(f'{a.role}_fit_summary.csv',fits)
print(json.dumps(dict(continue_confirmation=gate,primary=effects['A-B6'],practical=effects['A-R0_CD']),indent=2))
