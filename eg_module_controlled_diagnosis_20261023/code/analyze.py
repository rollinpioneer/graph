import numpy as np,pandas as pd,json,math,hashlib
from pathlib import Path
O=Path(r'C:\Users\jackx\Documents\Codex\2026-10-21\ssh-aita-xushijie-172-19-164\outputs\eg_module_controlled_diagnosis_20261023')
OLD=O.parent/'EG_pose_quality_20261022/data';R=O/'data';seed=20261023
def good(x):return np.isfinite(x)&(x>0)&(x<=.4000001)
def dump(name,x):
 (O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist()),encoding='utf-8')
def ci(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)]
 if not len(x):return None
 b=np.random.default_rng(seed).choice(x,(10000,len(x))).mean(1);return {'scene_mean':float(x.mean()),'CI95':[float(t) for t in np.quantile(b,[.025,.975])],'scenes':len(x)}
def invrate(scores,mu,obj):
 vals=[]
 for o in np.unique(obj):
  g=(obj==o)&good(mu);b=(obj==o)&np.isfinite(mu)&~good(mu)
  if g.any() and b.any():vals.append(float(((scores[b,None]>scores[g][None,:])+0.5*(scores[b,None]==scores[g][None,:])).mean()))
 return float(np.mean(vals)) if vals else np.nan
rows=[];comps=[];aux=[];grid=[];feat=[];qstats=[]
for p in sorted((R/'evaluation').glob('*.npz')):
 tag=p.stem;split,s,f=tag.split('_');s=int(s);f=int(f);e=np.load(p);z=np.load(R/f'cache/{tag}.npz');a=np.load(OLD/f'cache/{tag}.npz');old=np.load(OLD/f'evaluation/{tag}.npz');ids=e['mechanism_ids'];ob=old['object_id'][ids];mu0=e['H0_mu'];pred0=z['H0_pred'];sc0=pred0[ids,0];g0=good(mu0)
 feat.append(dict(split=split,scene=s,frame=f,N=float(z['N_F_relative_L2']),E1=float(z['E1_F_relative_L2']) if 'E1_F_relative_L2' in z.files else np.nan))
 for v in ['H0','H1_half','H1_off','H2_half','H2_off','H3','H4','N']+(['E1'] if split=='train' else []):
  pr=z[v+'_pred'];mu=e[v+'_mu'];gv=good(mu);scores=pr[ids,0];oldr=pred0[ids,4:13].reshape(-1,3,3);newr=pr[ids,4:13].reshape(-1,3,3);cos=(np.einsum('nij,nij->n',oldr,newr)-1)/2;cos2=(np.einsum('nij,nij->n',oldr,newr*np.array([1,-1,-1])[None,None,:])-1)/2;rotdeg=np.rad2deg(np.arccos(np.clip(np.maximum(cos,cos2),-1,1)))
  for j,cid in enumerate(ids):
   rows.append(dict(split=split,scene=s,frame=f,candidate_id=int(cid),object_id=int(ob[j]),variant=v,baseline_mu=mu0[j],mu=mu[j],baseline_good=bool(g0[j]),good=bool(gv[j]),collision=bool(e[v+'_collision'][j]),empty=bool(e[v+'_empty'][j]),unknown_reason=str(e[v+'_unknown_reason'][j]),baseline_score=sc0[j],score=scores[j],rotation_delta_deg=rotdeg[j],depth_delta_mm=1000*(pr[cid,3]-pred0[cid,3]),width_delta_mm=1000*(pr[cid,1]-pred0[cid,1]),valid_match=bool(a['batch_valid_mask'][cid]) if split=='train' else None,qref=float(a['batch_grasp_score'][cid]) if split=='train' else None))
  for geo,score,mm,ss in [('original','original',mu0,sc0),('new','original',mu,sc0),('original','new',mu0,scores),('new','new',mu,scores)]:
   top=int(np.argmax(ss));comps.append(dict(split=split,scene=s,frame=f,variant=v,geometry=geo,score=score,n=len(ids),top1_id=int(ids[top]),top1_good=int(good(mm)[top]),feasible_rate=float(good(mm).mean()),inversion=invrate(ss,mm,ob)))
 if split=='C3':
  for v in ['H0','H1_off','H2_off']:
   fi=old['final_ids'] if v=='H0' else e['aux_'+v+'_ids'];mu=old['P0_mu'][fi] if v=='H0' else e['aux_'+v+'_mu'];ss=z[v+'_pred'][fi,0];i=np.argmax(ss)
   aux.append(dict(scene=s,frame=f,variant=v,n=len(fi),top1_id=int(fi[i]),top1_good=int(good(mu)[i]),feasible=int(good(mu).sum()),unknown=int((~np.isfinite(mu)).sum())))
 else:
  gi=e['grid_ids'];gm=e['grid_mu'].reshape(-1,12,4)
  for j,cid in enumerate(gi):
   ai=int(a['angle_index'][cid]);di=int(a['depth_index'][cid]);aa=np.argsort(-a['angle_logits'][:12,cid],kind='stable')[:2];dd=np.argsort(-a['depth_logits'][:4,cid],kind='stable')[:2];g=good(gm[j]);b=bool(g[ai,di]);top2=bool(g[np.ix_(aa,dd)].any());anyg=bool(g.any())
   for ang in range(12):
    for dep in range(4):grid.append(dict(scene=s,frame=f,candidate_id=int(cid),query_stratum='raw_top1' if j==0 else 'rank_stratified' if j<8 else 'random',angle=ang,depth=dep,mu=gm[j,ang,dep],collision=bool(e['grid_collision'].reshape(-1,12,4)[j,ang,dep]),empty=bool(e['grid_empty'].reshape(-1,12,4)[j,ang,dep]),baseline_cell=ang==ai and dep==di,top2_cell=ang in aa and dep in dd,baseline_good=b,top2_any=top2,grid_any=anyg))
  valid=a['batch_valid_mask'];q0=(a['batch_grasp_score'].astype(np.float32)*10/2).astype(int);mus=[old['R_labelpoint_mu'],old['P0_mu']];targets=[q0]+[(np.where(m>0,np.clip(np.round(1.1-m,6),0,1),0).astype(np.float32)*10/2).astype(int) for m in mus];log=z['H0_quality_logits'].T.astype(float);exp=np.exp(log-log.max(1,keepdims=True));prob=exp/exp.sum(1,keepdims=True);H=z['quality_features'];high=np.isin(np.arange(1024),old['final_ids'])
  for group,mask in [('valid',valid),('qref_zero',valid&(a['batch_grasp_score']==0)),('qref_positive',valid&(a['batch_grasp_score']>0)),('high_fail',valid&high&~good(old['P0_mu'])),('original_feasible',valid&good(old['P0_mu']))]:
   if not mask.any():continue
   ggs=[]
   for lab,yy in zip(['Q0','Q1','Q2'],targets):
    y=np.eye(6)[yy[mask]];pp=prob[mask];grad=(pp-y).T/len(pp);gg=np.concatenate([(grad@H[mask]).ravel(),grad.sum(1)]);ggs.append(gg)
    qstats.append(dict(scene=s,frame=f,group=group,target=lab,n=int(mask.sum()),CE=float(-np.log(pp[np.arange(len(pp)),yy[mask]]).mean()),Brier=float(((pp-y)**2).sum(1).mean()),score_head_gradient_norm=float(np.linalg.norm(gg)),target_disagree_Q0=float((yy[mask]!=targets[0][mask]).mean()),target_disagree_Q1=float((yy[mask]!=targets[1][mask]).mean())))
df=pd.DataFrame(rows);df.to_csv(O/'intervention_results.csv',index=False);cc=pd.DataFrame(comps);cc.to_csv(O/'counterfactual_composition.csv',index=False);ax=pd.DataFrame(aux);ax.to_csv(O/'pipeline_auxiliary.csv',index=False);gr=pd.DataFrame(grid);gr.to_csv(O/'local_AD_grid.csv',index=False);pd.DataFrame(feat).to_csv(O/'feature_changes.csv',index=False);pd.DataFrame(qstats).to_csv(O/'quality_target_groups.csv',index=False)
stats={};comp={}
for (split,v),d in df.groupby(['split','variant']):
 scene=d.groupby('scene').agg(base=('baseline_good','mean'),new=('good','mean'));g=d.good.astype(bool);b=d.baseline_good.astype(bool)
 stats[f'{split}/{v}']=dict(n=len(d),baseline=int(b.sum()),good=int(g.sum()),recovery=int((g&~b).sum()),harm=int((b&~g).sum()),unknown=int(d.mu.isna().sum()),scene_delta=ci(scene.new-scene.base),collision=int(d.collision.sum()),empty=int(d['empty'].sum()),rotation_median_deg=float(d.rotation_delta_deg.median()),depth_mean_abs_mm=float(d.depth_delta_mm.abs().mean()),width_mean_abs_mm=float(d.width_delta_mm.abs().mean()))
 for (geo,score),c in cc[(cc.split==split)&(cc.variant==v)].groupby(['geometry','score']):
  base=cc[(cc.split==split)&(cc.variant=='H0')&(cc.geometry=='original')&(cc.score=='original')].set_index(['scene','frame']);cur=c.set_index(['scene','frame']);delta=(cur.top1_good-base.top1_good).groupby('scene').mean();invd=(cur.inversion-base.inversion).groupby('scene').mean();comp[f'{split}/{v}/{geo}/{score}']=dict(top1=int(c.top1_good.sum()),frames=len(c),top1_scene_delta=ci(delta),inversion_scene_mean=ci(c.groupby('scene').inversion.mean()),inversion_scene_delta=ci(invd))
dump('counterfactual_composition.json',comp)
gg=gr.drop_duplicates(['scene','frame','candidate_id']);bad=gg[~gg.baseline_good]
summary=dict(interventions=stats,grid=dict(queries=len(gg),scenes=int(gg.scene.nunique()),baseline_good=int(gg.baseline_good.sum()),top2_any=int(gg.top2_any.sum()),grid_any=int(gg.grid_any.sum()),baseline_failed=len(bad),failed_top2_reachable=int(bad.top2_any.sum()),failed_grid_only=int((bad.grid_any&~bad.top2_any).sum()),failed_grid_none=int((~bad.grid_any).sum()),unknown=int(gr.mu.isna().sum()),scene_top2_gain=ci(gg.assign(delta=gg.top2_any.astype(int)-gg.baseline_good.astype(int)).groupby('scene').delta.mean()),scene_grid_gain=ci(gg.assign(delta=gg.grid_any.astype(int)-gg.baseline_good.astype(int)).groupby('scene').delta.mean())),auxiliary={v:dict(top1=int(d.top1_good.sum()),frames=len(d),candidates=int(d.n.sum()),feasible=int(d.feasible.sum()),unknown=int(d.unknown.sum())) for v,d in ax.groupby('variant')})
for name in ['routes','population','cosines']:
 frames=[pd.read_csv(p) for p in sorted((R/'gradient').glob('*_'+name+'.csv'))];pd.concat(frames,ignore_index=True).to_csv(O/({'routes':'gradient_routes.csv','population':'loss_population.csv','cosines':'gradient_cosines.csv'}[name]),index=False)
rt=pd.read_csv(O/'gradient_routes.csv');lp=pd.read_csv(O/'loss_population.csv');co=pd.read_csv(O/'gradient_cosines.csv')
summary['gradient_routes']=[dict(loss=l,block=b,frames=len(d),nonzero=int((d['norm']>1e-12).sum()),median_norm=float(d['norm'].median()),median_weighted_norm=float(d.weighted_norm.median())) for (l,b),d in rt.groupby(['loss','block'])]
summary['gradient_cosines']=[dict(loss1=l1,loss2=l2,block=b,defined=int(d.cosine.notna().sum()),negative=int((d.cosine<0).sum()),median=float(d.cosine.median()) if d.cosine.notna().any() else None) for (l1,l2,b),d in co.groupby(['loss1','loss2','block'])]
summary['zero_loss']={l:dict(zero_samples=int(d[d.population=='zero'].n.sum()),valid_samples=int(d[d.population=='zero'].valid_denominator.sum()),zero_contribution_mean=float(d[d.population=='zero'].weighted_contribution.mean()),positive_contribution_mean=float(d[d.population=='positive'].weighted_contribution.mean())) for l,d in lp.groupby('loss')}
summary['E1_valid']={}
for label,d in [('valid',df[(df.variant=='E1')&(df.valid_match==True)]),('valid_positive',df[(df.variant=='E1')&(df.valid_match==True)&(df.qref>0)])]:
 g=d.good.astype(bool);b=d.baseline_good.astype(bool);delta=d.assign(delta=g.astype(int)-b.astype(int)).groupby('scene').delta.mean();summary['E1_valid'][label]=dict(n=len(d),baseline=int(b.sum()),good=int(g.sum()),recovery=int((g&~b).sum()),harm=int((b&~g).sum()),scene_delta=ci(delta))
receipts=[json.loads(p.read_text()) for p in sorted((R/'receipts').glob('*.json'))];evals=[json.loads(p.read_text()) for p in sorted((R/'evaluation').glob('*.json'))]
summary['replay']=dict(frames=len(receipts),historical_geometry_exact=all(x['historical_geometry_exact'] for x in receipts),max_score_error=max(x['historical_score_max_error'] for x in receipts),max_gradient_sum_error=max(x['gradient']['gradient_sum_relative_L2'] for x in receipts if 'gradient' in x),max_roundtrip_error=max(x['geometry_roundtrip_max_error'] for x in receipts if 'gradient' in x),mask_replay_differences=sum(x.get('label_replay_differences',{}).get('batch_valid_mask',0) for x in receipts),target_replay_differences=sum(x.get('label_replay_differences',{}).get('batch_grasp_score',0) for x in receipts))
summary['resources']=dict(analytic_calls=sum(x['calls'] for x in evals),analytic_CPU_hours=sum(x['cpu_seconds'] for x in evals)/3600,forward_probe_seconds=sum(x['seconds'] for x in receipts),updates=0)
summary['extra_classes']={k:sum(x['gradient'][k] for x in receipts if 'gradient' in x) for k in ['valid_angle13','valid_depth5','angle13_argmax','depth5_argmax','valid','zero']}
dump('statistics.json',summary);dump('baseline_replay.json',dict(summary=summary['replay'],receipts=receipts));dump('training_receipts.json',dict(updates=0,optimizer_created=False,checkpoint_modified=False,model_unchanged=json.loads((R/'all_complete.json').read_text())['model_unchanged'],unexecuted=['E3-2','E5','E6'],reason='parameter updates not authorized'))
print(json.dumps(summary,ensure_ascii=False,indent=2))
