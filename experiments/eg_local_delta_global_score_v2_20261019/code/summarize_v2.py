from common_v2 import *
import argparse,csv
p=argparse.ArgumentParser();p.add_argument('--stage',default='development');args=p.parse_args();scenes=CFG['development'] if args.stage=='development' else sum(CFG['transfer_development'].values(),[]);records=[]
for s in scenes:
 folder=R/f'evaluation/{args.stage}/{s:04d}';assert json.loads((folder/'COMPLETE.json').read_text())['status']=='PASS'
 for f in CFG['dev_frames']:
  v=json.loads((folder/f'{f:04d}.json').read_text());assert len(v)==20;records+=v
names=sorted(set(x['configuration'] for x in records));rows=[];table={}
for s in scenes:
 for name in names:
  v=[x for x in records if x['scene']==s and x['configuration']==name];assert len(v)==32
  x=dict(scene=s,configuration=name,N=len(v),**{k:float(np.mean([z[k] for z in v])) for k in ['ap','ap04','ap08','top1_valid','top1_collision','top1_empty','probe_regret','probe_utility_gain']})
  for k in ['R','D','original_top_success','modified','candidates','probe_n','probe_improved','probe_degraded']:x[k]=int(sum(z[k] for z in v))
  x.update(Net=(x['R']-x['D'])/x['N'],Damage=x['D']/x['original_top_success'] if x['original_top_success'] else None);rows.append(x);table[s,name]=x
with open(R/f'results/{args.stage}_per_scene.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
diff=np.array([[table[s,f'Gdelta_{seed}_CD']['ap']-table[s,'R0_CD']['ap'] for s in scenes] for seed in SEEDS]);score_delta=np.array([[table[s,f'Gdelta_{seed}_CD']['ap']-table[s,f'G0_{seed}_CD']['ap'] for s in scenes] for seed in SEEDS]);rng=np.random.default_rng(20261019);ix=rng.integers(0,len(scenes),(10000,len(scenes)));boot=diff.mean(0)[ix].mean(1);local=[]
for seed in SEEDS:
 rr=np.array([table[s,f'Gdelta_{seed}_CD']['R'] for s in scenes]);dd=np.array([table[s,f'Gdelta_{seed}_CD']['D'] for s in scenes]);ns=np.array([table[s,f'Gdelta_{seed}_CD']['original_top_success'] for s in scenes]);n=np.array([table[s,f'Gdelta_{seed}_CD']['N'] for s in scenes]);net=(rr.sum()-dd.sum())/n.sum();damage=dd.sum()/ns.sum() if ns.sum() else None;den=ns[ix].sum(1)
 local.append(dict(seed=seed,N=int(n.sum()),original_success=int(ns.sum()),R=int(rr.sum()),D=int(dd.sum()),Net=float(net),Damage=float(damage) if damage is not None else None,Net_ci95=np.quantile((rr-dd)[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist(),Damage_ci95=np.nanquantile(dd[ix].sum(1)/np.where(den>0,den,np.nan),[.025,.975]).tolist(),pass_local=bool(net>0 and damage is not None and damage<=.02)))
criteria=dict(mean_ap_at_least_1=float(diff.mean())>=1,seeds_all_positive=bool((diff.mean(1)>0).all()),positive_scenes=int((diff.mean(0)>0).sum()),positive_scene_requirement=6 if args.stage=='development' else None,all_seed_local_pass=all(x['pass_local'] for x in local));passed=(criteria['mean_ap_at_least_1'] and criteria['seeds_all_positive'] and criteria['positive_scenes']>=6 and criteria['all_seed_local_pass']) if args.stage=='development' else None
write(R/f'results/{args.stage}_decision.json',dict(status='COMPLETE',pass_stage=bool(passed) if passed is not None else None,mean_ap_difference_pp=float(diff.mean()),seed_ap_difference_pp=diff.mean(1).tolist(),scene_ap_difference_pp=dict(zip(map(str,scenes),diff.mean(0).tolist())),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),global_score_effect_Gdelta_minus_G0=float(score_delta.mean()),global_score_effect_ci95=np.quantile(score_delta.mean(0)[ix].mean(1),[.025,.975]).tolist(),local=local,criteria=criteria,statistics='scene-cluster bootstrap 10000; paired mean of seeds; exploratory observed scenes',confirmation='NOT_OPENED'))
print('STAGE_RESULT',args.stage,passed,float(diff.mean()),local,flush=True)
