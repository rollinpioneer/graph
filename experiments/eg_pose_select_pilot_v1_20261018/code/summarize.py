from core import *
import argparse,csv
p=argparse.ArgumentParser();p.add_argument('--stage',default='development');a=p.parse_args();scenes=CFG['development'] if a.stage=='development' else sum(CFG['transfer_development'].values(),[]) if a.stage=='transfer' else sum(CFG['confirmation'].values(),[])
rows=[]
for s in scenes:
 folder=R/f'evaluation/{a.stage}/{s:04d}';assert json.loads((folder/'COMPLETE.json').read_text())['status']=='PASS'
 for f in CFG['dev_frames'] if a.stage!='confirmation' else CFG['confirmation_frames']:rows+=json.loads((folder/f'{f:04d}.json').read_text())
names=sorted(set(x['configuration'] for x in rows));summary=[];table={}
for s in scenes:
 for name in names:
  v=[x for x in rows if x['scene']==s and x['configuration']==name];assert len(v)==(32 if a.stage!='confirmation' else 256)
  y=dict(scene=s,configuration=name,frames=len(v),**{k:float(np.mean([x[k] for x in v])) for k in ['ap','ap04','ap08','top1_valid','top1_collision','top1_empty','probe_regret','probe_oracle_coverage']})
  for k in ['R','D','original_top_success','modified','candidates','probe_changed','probe_improved','probe_degraded','probe_success_degraded','probe_original_success']:y[k]=int(sum(x[k] for x in v))
  y['Net']=(y['R']-y['D'])/len(v);y['Damage']=y['D']/y['original_top_success'] if y['original_top_success'] else None;summary.append(y);table[s,name]=y
with open(R/f'results/{a.stage}_per_scene.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
diff=np.array([[table[s,f'S4_{seed}_CD']['ap']-table[s,'R0_CD']['ap'] for s in scenes] for seed in SEEDS]);top=np.array([[table[s,f'S4_{seed}_CD']['top1_valid']-table[s,'R0_CD']['top1_valid'] for s in scenes] for seed in SEEDS]);rr=np.array([[table[s,f'S4_{seed}_CD']['R'] for s in scenes] for seed in SEEDS]);dd=np.array([[table[s,f'S4_{seed}_CD']['D'] for s in scenes] for seed in SEEDS]);ns=np.array([table[s,'R0_CD']['original_top_success'] for s in scenes]);n=np.array([table[s,'R0_CD']['frames'] for s in scenes]);net=float((rr-dd).sum()/(3*n.sum()));damage=float(dd.sum()/(3*ns.sum())) if ns.sum() else None
rng=np.random.default_rng(20261018);ix=rng.integers(0,len(scenes),size=(10000,len(scenes)))
if a.stage=='confirmation':ix=np.concatenate([rng.integers(k*4,(k+1)*4,size=(10000,4)) for k in range(3)],axis=1)
boot=diff.mean(0)[ix].mean(1);netboot=(rr-dd).mean(0)[ix].sum(1)/n[ix].sum(1);den=ns[ix].sum(1);damageboot=dd.mean(0)[ix].sum(1)/np.where(den>0,den,np.nan)
passes=dict(ap_mean=float(diff.mean())>=(1. if a.stage!='transfer' else .5),seeds_all_positive=bool((diff.mean(1)>0).all()),positive_scene_count=int((diff.mean(0)>0).sum()),net_positive=net>0,damage_le_2percent=damage is not None and damage<=.02,application_top1_not_decreased=float(top.mean())>=0)
passed=passes['ap_mean'] and passes['seeds_all_positive'] and passes['net_positive']
if a.stage!='transfer':passed=passed and passes['positive_scene_count']>=(6 if a.stage=='development' else 8) and passes['damage_le_2percent'] and passes['application_top1_not_decreased']
if a.stage=='confirmation':passed=passed and float(np.quantile(boot,.025))>0
write(R/f'results/{a.stage}_decision.json',dict(status='COMPLETE',pass_stage=bool(passed),mean_ap_difference_pp=float(diff.mean()),seed_ap_difference_pp=diff.mean(1).tolist(),scene_ap_difference_pp=dict(zip(map(str,scenes),diff.mean(0).tolist())),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),Net=net,Net_ci95=np.quantile(netboot,[.025,.975]).tolist(),Damage=damage,Damage_ci95=np.nanquantile(damageboot,[.025,.975]).tolist() if ns.sum() else None,R_by_seed=rr.sum(1).tolist(),D_by_seed=dd.sum(1).tolist(),N_per_seed=int(n.sum()),original_success_per_seed=int(ns.sum()),application_top1_difference=float(top.mean()),criteria=passes,statistical_unit='scene; paired seed means; exploratory development material'))
# Fixed first seed, earliest scene/frame; record both positive and negative paired outcomes.
case_rows=[]
for cat in ['rescued','damaged','unchanged_failure']:
 candidates=[x for x in rows if x['configuration']==f'S4_{SEEDS[0]}_CD' and (x['R'] if cat=='rescued' else x['D'] if cat=='damaged' else not x['original_top_success'] and x['selected_action']==0)]
 if candidates:case_rows.append(dict(category=cat,**min(candidates,key=lambda x:(x['scene'],x['frame']))))
write(R/f'results/{a.stage}_bad_cases.json',dict(cases=case_rows,missing=[k for k in ['rescued','damaged','unchanged_failure'] if k not in [x['category'] for x in case_rows]],exploratory=True))
print('DECISION',a.stage,passed,float(diff.mean()),net,damage,flush=True)
