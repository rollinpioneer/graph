"""Scene-level fixed-protocol analysis; does not tune or execute detectors."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;P=json.loads((R/'protocol.json').read_text());C=json.loads((R/'frozen_config.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writecsv(name,rows):
 with (R/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mask(a,v,op,method):
 if method=='EG':return a['EG_cd'].ravel()
 if method=='EG_full':return a['full_EG'][:int(a['variant_count'])]
 cfg=C['workpoints'][op]['methods'][method];key=cfg['key']
 if key.startswith(('ref_','nv_')):
  fam,k=key.split('_');z=a if fam=='ref' else v;sc=z[fam+'_occ']+float(k)*z[fam+'_unk']
 else:sc=a[key]
 return sc[:int(a['variant_count'])]>cfg['threshold']
def bootstrap_indices(scenes,split):
 rng=np.random.default_rng(20261024)
 strata=[np.arange(len(scenes))] if split=='train' else [np.flatnonzero(np.array(scenes)//30==k) for k in []]
 if split!='train':strata=[np.flatnonzero(np.array(scenes)<130),np.flatnonzero((np.array(scenes)>=130)&(np.array(scenes)<160)),np.flatnonzero(np.array(scenes)>=160)]
 return np.concatenate([rng.choice(st,(10000,len(st)),replace=True) for st in strata],1)
def estimate(values,indices):
 a=np.asarray(values,float)
 if not np.isfinite(a).any():return dict(mean=None,low=None,high=None)
 boot=np.nanmean(a[indices],axis=1)
 return dict(mean=float(np.nanmean(a)),low=float(np.nanquantile(boot,.025)),high=float(np.nanquantile(boot,.975)))
methods=['EG','EG_full','R1','R2','R3','R4','N_ref','N_ref_free','nvblox','nvblox_free']
counts=defaultdict(lambda:np.zeros(8,dtype=int));panel={};light=R/'per_pose';light.mkdir(exist_ok=True);dense_counts=defaultdict(Counter);mechanism_rows=[];eff=[]
for split in ['train','C3','P2']:
 scenes=P['training_scenes' if split=='train' else 'C3_scenes'];frames=P['training_frames' if split=='train' else 'P2_frames' if split=='P2' else 'C3_frames'];panel[split]=dict(frames=0,candidates=0,poses=0)
 for s in scenes:
  for f in frames:
   ap=R/f'cpu/{split}_{s:04d}_{f:04d}.npz';vp=R/f'nv/{split}_{s:04d}_{f:04d}.npz';a=np.load(ap);v=np.load(vp);n=int(a['variant_count']);co=a['collision'].ravel();em=a['empty'].ravel();mu=a['mu'].ravel();pos=co&~em;neg=~co&~em;feas=neg&(mu>0)&(mu<=.4)
   panel[split]['frames']+=1;panel[split]['candidates']+=len(a['candidate_id']);panel[split]['poses']+=n
   export={k:a[k] for k in ['candidate_id','original_angle','object_id','collision','empty','mu']}
   for k in ['R1','R2','R3','R4','ref_occ','ref_unk']:export[k]=a[k][:n].reshape(-1,12)
   for k in ['nv_occ','nv_unk']:export[k]=v[k][:n].reshape(-1,12)
   masks={}
   if split=='C3':
    dense=np.load(R/f'dense/{s:04d}_{f:04d}.npz');robust=dense['robust_collision'].ravel();dc=dense['dense_collision'].ravel();export.update(robust_collision=robust.reshape(-1,12),dense_collision=dc.reshape(-1,12))
   for op in ['OP1','OP2']:
    for method in methods:
     m=mask(a,v,op,method);masks[(op,method)]=m;counts[(split,op,method,s)]+=np.array([pos.sum(),(pos&m).sum(),neg.sum(),(neg&m).sum(),feas.sum(),(feas&m).sum(),em.sum(),(em&m).sum()])
     export[op+'_'+method]=m.reshape(-1,12)
     if split=='C3' and method in ['N_ref','nvblox',C['workpoints'][op]['best_simple']]:
      new=m&~mask(a,v,op,'EG')&pos;lost=~m&mask(a,v,op,'EG')&pos
      dense_counts[(op,method)].update(new_hits=int(new.sum()),new_robust=int((new&robust).sum()),new_dense=int((new&dc).sum()),lost_hits=int(lost.sum()),lost_robust=int((lost&robust).sum()))
   np.savez_compressed(light/f'{split}_{s:04d}_{f:04d}.npz',**export)
   if split=='C3':
    old=json.loads((R.parent/f'eg_i1_collision_controls_20261024/mechanism/{s:04d}_{f:04d}.json').read_text())
    lookup={int(x):j for j,x in enumerate(a['candidate_id'])}
    for case in old['rows']:
     j=lookup[case['candidate_id']];k=case['angle_class'];i=j*12+k
     category='feasible_false_filter' if case['scope']=='feasible_false_filter' else 'missing_observation' if case['observed_points_in_GT_collision_solid']==0 else 'geometry_mismatch' if case['observed_GT_points_also_counted_by_EG']==0 else 'below_threshold'
     for op in ['OP1','OP2']:
      for method in ['N_ref','nvblox',C['workpoints'][op]['best_simple']]:mechanism_rows.append(dict(scene=s,frame=f,candidate_id=case['candidate_id'],angle=k,scope=case['scope'],category=category,op=op,method=method,filtered=int(masks[(op,method)][i]),dense=int(dc[i]),robust=int(robust[i])))
   cr=json.loads(ap.with_suffix('.json').read_text());nr=json.loads(vp.with_suffix('.json').read_text())
   eff.append(dict(split=split,scene=s,frame=f,cpu_all_detectors_seconds=cr['detector_seconds'],nv_integration_seconds=nr['integration_seconds'],nv_query_seconds=nr['query_seconds'],nv_sampling_seconds=nr['sampling_seconds'],nv_process_GPU_MiB=max(nr['process_GPU_MiB'],default=0)))
scene_rows=[];metric_rows=[];comparison_rows=[];look={};metric_names=['recall','false_filter_clean','false_filter_feasible']
for split in ['train','C3','P2']:
 scenes=sorted(P['training_scenes' if split=='train' else 'C3_scenes']);indices=bootstrap_indices(scenes,split)
 for op in ['OP1','OP2']:
  for method in methods:
   values=[]
   for s in scenes:
    p,tp,n,fp,q,fq,e,ef=counts[(split,op,method,s)];arr=[tp/p if p else np.nan,fp/n if n else np.nan,fq/q if q else np.nan];values.append(arr)
    scene_rows.append(dict(split=split,op=op,method=method,scene=s,collision=int(p),collision_detected=int(tp),clean=int(n),clean_filtered=int(fp),feasible=int(q),feasible_filtered=int(fq),empty=int(e),empty_filtered=int(ef),**dict(zip(metric_names,arr))))
   values=np.array(values);look[(split,op,method)]=values
   for j,name in enumerate(metric_names):metric_rows.append(dict(split=split,op=op,method=method,metric=name,**estimate(values[:,j],indices)))
  for first,second in [('N_ref','EG'),('N_ref',C['workpoints'][op]['best_simple']),('N_ref','nvblox'),('nvblox','EG')]:
   for j,name in enumerate(metric_names):comparison_rows.append(dict(split=split,op=op,first=first,second=second,metric=name,**estimate(look[(split,op,first)][:,j]-look[(split,op,second)][:,j],indices)))
writecsv('scene_detector_statistics.csv',scene_rows);writecsv('detector_metrics.csv',metric_rows);writecsv('paired_comparisons.csv',comparison_rows);writecsv('mechanism_transitions.csv',mechanism_rows);writecsv('efficiency.csv',eff)
joint_rows=[];selected=[];top_rows=[];jc=defaultdict(Counter);scene_joint=defaultdict(Counter);extra_fc=0
for s in P['C3_scenes']:
 for f in P['C3_frames']:
  a=np.load(R/f'cpu/C3_{s:04d}_{f:04d}.npz');data=json.loads((R/f'joint/{s:04d}_{f:04d}.json').read_text());extra_fc+=data['new_FC_count'];nom=a['original_angle'];n=len(nom);ix=np.arange(n);oldmu=a['mu'][ix,nom];oldgood=(oldmu>0)&(oldmu<=.4);oldweak=oldmu>.4
  for g in data['groups']:
   ang=np.array(g['angles']);mu=a['mu'][ix,ang];co=a['collision'][ix,ang];em=a['empty'][ix,ang];good=~co&~em&(mu>0)&(mu<=.4);weak=~co&~em&(mu>.4);nofc=~co&~em&(mu<=0);valid=(mu>0)&(oldmu>0)&~co&~em
   row=dict(scene=s,frame=f,op=g['op'],group=g['group'],n=n,original_good=int(oldgood.sum()),original_weak=int(oldweak.sum()),feasible=int(good.sum()),recovery=int((oldweak&good).sum()),harm=int((oldgood&~good).sum()),collision=int((co&~em).sum()),empty=int(em.sum()),no_FC=int(nofc.sum()),fallback=sum(g['fallback']),mu_pair_n=int(valid.sum()),mu_delta_sum=float((mu[valid]-oldmu[valid]).sum()),top1_feasible=int(g['top1']['status']=='feasible'),top1_collision=int(g['top1']['status']=='collision'),top1_unknown=int(g['top1']['status']=='unknown'))
   joint_rows.append(row);jc[(g['op'],g['group'])].update({k:v for k,v in row.items() if k not in ['scene','frame','op','group']});scene_joint[(g['op'],g['group'],s)].update({k:v for k,v in row.items() if k not in ['scene','frame','op','group']})
   for j,cid in enumerate(a['candidate_id']):selected.append(dict(scene=s,frame=f,op=g['op'],group=g['group'],candidate_id=int(cid),object_id=int(a['object_id'][j]),original_angle=int(nom[j]),angle=int(ang[j]),original_mu=float(oldmu[j]),mu=float(mu[j]),collision=int(co[j]),empty=int(em[j]),feasible=int(good[j]),recovery=int(oldweak[j]&good[j]),harm=int(oldgood[j]&~good[j]),fallback=int(g['fallback'][j])))
   top_rows.append(dict(scene=s,frame=f,op=g['op'],group=g['group'],top1_status=g['top1']['status'],fixed_top1_status=g['fixed_top1']['status'],final_count=len(g['final_ids']),final_unknown=sum(x['status']=='unknown' for x in g['final_results']),post_cd=g['post_cd'],nms_count=len(g['nms_ids']),object10_count=len(g['object10_ids'])))
writecsv('joint_frame_statistics.csv',joint_rows);writecsv('selected_candidates.csv',selected);writecsv('top1_and_postprocess.csv',top_rows)
joint_summary=[dict(op=op,group=g,**dict(v)) for (op,g),v in jc.items()];writecsv('joint_summary.csv',joint_summary)
joint_ci=[];scenes=sorted(P['C3_scenes']);indices=bootstrap_indices(scenes,'C3')
for op in ['OP1','OP2']:
 for g in ['B','N','N+','N_rule','E','O_N']:
  for baseline in ['O','B']:
   for metric,den in [('feasible','n'),('recovery','original_weak'),('harm','original_good'),('collision','n'),('top1_feasible',None)]:
    vals=[]
    for s in scenes:
     x=scene_joint[(op,g,s)];y=scene_joint[(op,baseline,s)];vals.append((x[metric]-y[metric])/(x[den] if den else 4))
    joint_ci.append(dict(op=op,group=g,baseline=baseline,metric=metric,**estimate(vals,indices)))
writecsv('joint_paired_intervals.csv',joint_ci)
mechanism_summary=[]
for category in ['below_threshold','missing_observation','geometry_mismatch','feasible_false_filter']:
 for op in ['OP1','OP2']:
  for method in ['N_ref','nvblox',C['workpoints'][op]['best_simple']]:
   rr=[r for r in mechanism_rows if r['category']==category and r['op']==op and r['method']==method]
   mechanism_summary.append(dict(category=category,op=op,method=method,n=len(rr),filtered=sum(r['filtered'] for r in rr),passed=sum(1-r['filtered'] for r in rr)))
writecsv('mechanism_summary.csv',mechanism_summary)
dense_summary=[dict(op=k[0],method=k[1],**dict(v)) for k,v in dense_counts.items()];writecsv('dense_audit_summary.csv',dense_summary)
def contrast(split,op,second,metric):return next(x for x in comparison_rows if x['split']==split and x['op']==op and x['first']=='N_ref' and x['second']==second and x['metric']==metric)
gates={}
c=contrast('C3','OP1','EG','recall');gates['recall_vs_EG_5pp_CI_positive']=c['mean']>=.05 and c['low']>0
gates['best_simple_both_OP_2pp']=all(contrast('C3',op,C['workpoints'][op]['best_simple'],'recall')['mean']>=.02 for op in ['OP1','OP2'])
gates['OP1_feasible_FPR_no_more_than_1p5pp_extra']=contrast('C3','OP1','EG','false_filter_feasible')['mean']<=.015
ns=jc[('OP1','N')];os=jc[('OP1','O')];on=jc[('OP1','O_N')]
gates.update(joint_collision_at_most_88=ns['collision']<=88,joint_recovery_at_least_173=ns['recovery']>=173,joint_harm_at_most_59=ns['harm']<=59,O_N_top1_no_loss=on['top1_feasible']>=os['top1_feasible'] and on['top1_collision']<=os['top1_collision'],P2_positive_recall=contrast('P2','OP1','EG','recall')['mean']>0)
den=next(x for x in dense_summary if x['op']=='OP1' and x['method']=='N_ref');gates['majority_new_hits_robust']=den['new_hits']>0 and den['new_robust']/den['new_hits']>.5
decision='GO-N' if all(gates.values()) else 'STOP' if not gates['recall_vs_EG_5pp_CI_positive'] or not gates['best_simple_both_OP_2pp'] else 'HOLD'
result=dict(decision=decision,gates=gates,panels=panel,calibration_kappa=C['kappa'],metrics=metric_rows,comparisons=comparison_rows,joint=joint_summary,dense=dense_summary,mechanism=mechanism_summary,new_direct_FC_candidates=extra_fc,frozen_config_sha256=sha(R/'frozen_config.json'),training=False,C4_C5=False)
(R/'statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,op in zip(axs,['OP1','OP2']):
 for name in ['EG','EG_full','R1','R2','R3','R4','N_ref','nvblox']:
  rows=[x for x in metric_rows if x['split']=='C3' and x['op']==op and x['method']==name];xx=next(x for x in rows if x['metric']=='false_filter_feasible');yy=next(x for x in rows if x['metric']=='recall')
  ax.errorbar(xx['mean']*100,yy['mean']*100,xerr=[[100*(xx['mean']-xx['low'])],[100*(xx['high']-xx['mean'])]],yerr=[[100*(yy['mean']-yy['low'])],[100*(yy['high']-yy['mean'])]],fmt='o',alpha=.65);ax.annotate(name,(xx['mean']*100,yy['mean']*100),fontsize=8)
 ax.set_title(op+' frozen on training scenes');ax.set_xlabel('Feasible grasp false filtering (%)');ax.set_ylabel('Collision recall (%)');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(R/'C3_tradeoff.png',dpi=160);plt.close(fig)
curves=json.loads((R/'calibration_curves.json').read_text());fig,ax=plt.subplots(figsize=(7,5))
for name in ['R1','R2','R3','R4','N_ref','nvblox']:
 z=curves['OP1_'+name];ax.plot(np.array(z['false_filter'])*100,np.array(z['recall'])*100,label=name)
ax.set_xlim(0,30);ax.set_xlabel('Training clean-grasp false filtering (%)');ax.set_ylabel('Training collision recall (%)');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(R/'calibration_tradeoff.png',dpi=160);plt.close(fig)
print(json.dumps(dict(decision=decision,gates=gates,panels=panel,joint=joint_summary,new_direct_FC=extra_fc),indent=2),flush=True)
