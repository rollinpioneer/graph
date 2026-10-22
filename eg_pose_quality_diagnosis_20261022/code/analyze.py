import os
from pathlib import Path
import json,hashlib,gzip,collections
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
from scipy.special import softmax
from scipy.stats import spearmanr

O=Path(os.environ.get('EG_POSE_OUTPUT_DIR', './outputs/EG_pose_quality_20261022')).resolve()
D=O/'data';plan=json.loads((D/'study_plan.json').read_text());rng=np.random.default_rng(20261022)
SC={'C3':plan['C3_scenes'],'train':plan['training_scenes']}
BOOT={k:np.concatenate([rng.integers(i,i+w,(10000,w)) for i in range(0,len(ss),w)],axis=1) for k,ss,w in [('C3',SC['C3'],4),('train',SC['train'],2)]}
def good(q):return np.isfinite(q)&(q>0)&(q<=.4000001)
def utility(q):return ((q[:,None]>0)&(q[:,None]<=np.array([.2,.4,.6,.8,1,1.2])[None,:]+1e-7)).mean(1)
def macro(rows,key,split):
 vals=np.array([np.mean([r[key] for r in rows if r['scene']==s and r.get(key) is not None and np.isfinite(r[key])]) if any(r['scene']==s and r.get(key) is not None and np.isfinite(r[key]) for r in rows) else np.nan for s in SC[split]])
 if not np.isfinite(vals).any():return dict(mean=None,ci95=[None,None],scene_values=vals.tolist(),eligible_scenes=0)
 with np.errstate(invalid='ignore'):samples=np.nanmean(vals[BOOT[split]],axis=1)
 return dict(mean=float(np.nanmean(vals)),ci95=np.nanquantile(samples,[.025,.975]).tolist(),scene_values=vals.tolist(),eligible_scenes=int(np.isfinite(vals).sum()))
def count_rate(df,event,mask,split='train'):
 z=df.loc[mask];rows=[]
 for s,x in z.groupby('scene'):rows.append(dict(scene=int(s),value=float(x[event].mean())))
 return dict(numerator=int(z[event].sum()),denominator=len(z),aggregate=macro(rows,'value',split))
def category(q,c,e):
 return np.where(~np.isfinite(q),'unknown',np.where(e,'empty',np.where(c,'collision',np.where(good(q),'feasible',np.where(q>0,'weak_quality','no_force_closure')))))

frames=[];objects=[];evidence=[];metas=[];verifications=[]
vars=['P0','P1','P2','P3','P4','P5','P5_aligned','R_predpoint','R_labelpoint','P1_aligned','P4_aligned','width_no_scale']
old=json.loads((Path(__file__).parent/'historical_EG_B_48frames.json').read_text());oldframes={(r['scene'],r['frame']):r for r in old['frames']}
for split in ['C3','train']:
 for s in SC[split]:
  for f in plan['C3_frames' if split=='C3' else 'training_frames']:
   name=f'{split}_{s:04d}_{f:04d}'
   with np.load(D/'cache'/f'{name}.npz') as z:a={k:z[k] for k in z.files}
   ep=D/'evaluation'/f'{name}.npz'
   with np.load(ep) as z:e={k:z[k] for k in z.files}
   receipt=json.loads(ep.with_suffix('.json').read_text());assert hashlib.sha256(ep.read_bytes()).hexdigest()==receipt['result_sha256']
   assert str(e['geometry_sha256'])==str(a['geometry_sha256'])==hashlib.sha256(np.ascontiguousarray(a['pred'][:,1:]).tobytes()).hexdigest()
   meta=json.loads((D/'cache'/f'{name}.json').read_text());metas.append(meta)
   pred=a['pred'];score=pred[:,0];q=e['P0_mu'];feasible=good(q);ids=e['final_ids'];noCD=e['final_noCD'];top=int(ids[0]);rawtop=int(np.argmax(score))
   assert np.isfinite(q[ids]).all() and np.isfinite(q[noCD]).all()
   assert not ((q>0)&(e['P0_collision']|e['P0_empty'])).any()
   cats=category(q,e['P0_collision'],e['P0_empty']);obj=e['object_id'];utility0=utility(q)
   logits=a['logits'].astype(float);pr=softmax(logits,axis=1)@np.arange(6)*.2;centered=logits@np.arange(6)*.2-3*logits.mean(1)
   fr=dict(scene=s,frame=f,split=split,top=top,top_object=int(obj[top]),top_feasible=bool(feasible[top]),top_mu=float(q[top]),top_category=str(cats[top]),final_count=len(ids),top_failure_has_good_final=bool(feasible[ids].any()),top_failure_has_same_object_good_final=bool(np.any(feasible[ids]&(obj[ids]==obj[top]))),noCD_top=int(noCD[0]),noCD_feasible=bool(feasible[noCD[0]]),noCD_category=str(cats[noCD[0]]),postprocessing_top_changed=int(np.flatnonzero(~a['cd_mask'])[np.argmax(score[~a['cd_mask']])])!=top)
   for key,sc in [('centered',centered),('softmax',pr)]:
    ix=int(ids[np.argmax(sc[ids])]);fr[key+'_top']=ix;fr[key+'_feasible']=bool(feasible[ix]);fr[key+'_delta']=float(feasible[ix])-float(feasible[top])
   fr['width_no_scale_feasible']=bool(good(e['width_no_scale_mu'][top]));fr['width_no_scale_delta']=float(fr['width_no_scale_feasible'])-float(feasible[top])
   if split=='C3':
    fr['EG_B_feasible']=oldframes[s,f]['feasible'];fr['full_minus_EGB']=float(feasible[top])-float(fr['EG_B_feasible'])
   frames.append(fr)
   for oo in np.unique(obj[ids]):
    ii=ids[obj[ids]==oo];gg=ii[feasible[ii]];bb=ii[~feasible[ii]];row=dict(scene=s,frame=f,split=split,object_id=int(oo),n=len(ii),good=len(gg),bad=len(bb))
    if len(gg) and len(bb):
     diff=score[bb,None]-score[gg][None,:];row['inversion_rate']=float(((diff>0)+.5*(diff==0)).mean());row['has_inversion']=bool((diff>0).any());row['pairs']=diff.size
    if len(ii)>=4 and np.ptp(utility0[ii])>0 and np.ptp(score[ii])>0:row['rho']=float(spearmanr(score[ii],utility0[ii]).statistic)
    objects.append(row)
   keep=np.arange(1024) if split=='train' else np.flatnonzero(np.isfinite(q))
   ranks=np.zeros(1024,int);ranks[ids]=np.arange(1,len(ids)+1)
   rawranks=np.zeros(1024,int);rawranks[np.argsort(-score,kind='stable')]=np.arange(1,1025)
   df=pd.DataFrame(dict(split=split,scene=s,frame=f,candidate_id=keep,object_id=obj[keep],final_rank=ranks[keep],raw_rank=rawranks[keep],original_score=score[keep],softmax_expected=pr[keep],centered_score=centered[keep],cd_filtered=a['cd_mask'][keep],pred_angle_class=a['angle_index'][keep],pred_angle_rad=a['angle_index'][keep]*np.pi/12,pred_depth_m=pred[keep,3],pred_width_m=pred[keep,1],raw_width_m=a['raw_width'][keep]/10,pred_view_index=a['view_index'][keep]))
   for j in range(6):df[f'quality_logit_{j}']=logits[keep,j]
   for j,k in enumerate('xyz'):df['pred_'+k+'_m']=pred[keep,13+j];df['approach_'+k]=pred[keep,4:13].reshape(-1,3,3)[:,j,0]
   for j in range(9):df[f'pred_rotation_{j}']=pred[keep,4+j]
   for v in vars:
    if v+'_mu' not in e:continue
    df[v+'_mu']=e[v+'_mu'][keep];df[v+'_collision']=e[v+'_collision'][keep];df[v+'_empty']=e[v+'_empty'][keep];df[v+'_good']=good(e[v+'_mu'][keep]);df[v+'_unknown_reason']=e[v+'_unknown_reason'][keep]
    if v in ['P0','P5','P5_aligned','R_labelpoint']:
     df[v+'_contact1_obs_dist_m']=e[v+'_contact_observed_distance'][keep,0];df[v+'_contact2_obs_dist_m']=e[v+'_contact_observed_distance'][keep,1]
   df['P0_category']=cats[keep]
   if split=='train':
    v=a['batch_valid_mask'];assert np.isfinite(q).all()
    for key in vars[1:-1]:assert np.isfinite(e[key+'_mu'][v]).all() and np.isnan(e[key+'_mu'][~v]).all()
    assert np.array_equal(e['P0_owner'][v],e['P5_owner'][v])
    df['valid_match']=v;df['valid_point']=a['valid_point_mask'];df['valid_view']=a['valid_view_mask'];df['match_distance_m']=a['dist'];df['label_point_index']=a['label_point_index'];df['label_source_view']=a['label_source_view'];df['matched_object_id']=a['matched_object_id'];df['owner_consistent']=a['matched_object_id']==obj
    df['target_angle_class']=np.where(v,a['batch_grasp_rotations'],np.nan);df['target_depth_class']=np.where(v,a['batch_grasp_depth'],np.nan);df['target_depth_m']=np.where(v,(a['batch_grasp_depth']+1)*.01,np.nan);df['target_width_m']=np.where(v,a['batch_grasp_width'],np.nan);df['target_score']=np.where(v,a['batch_grasp_score'],np.nan)
    df['target_quality_class']=np.where(v,(a['batch_grasp_score']*10/2).astype(int),np.nan);df['pred_quality_class']=logits.argmax(1);df['quality_class_correct']=df['pred_quality_class']==df['target_quality_class']
    df['angle_class_correct']=a['angle_index']==a['batch_grasp_rotations'];df['depth_class_correct']=a['depth_index']==a['batch_grasp_depth'];df['AD_correct']=df['angle_class_correct']&df['depth_class_correct'];df['AD_error']=~df['AD_correct']
    da=(a['angle_index']-a['batch_grasp_rotations'])*np.pi/12;df['angle_error_deg']=np.where(v,np.rad2deg(abs((da+np.pi/2)%np.pi-np.pi/2)),np.nan);df['depth_error_mm']=np.where(v,1000*abs(pred[:,3]-(a['batch_grasp_depth']+1)*.01),np.nan);df['width_error_mm']=np.where(v,1000*abs(pred[:,1]-a['batch_grasp_width']),np.nan)
    df['view_error_deg']=a['view_rotation_error_deg'];df['probe_tf32_valid_changed']=a['probe_tf32_valid_mask']!=v;df['probe_tf32_target_changed']=a['probe_tf32_score_target']!=a['batch_grasp_score']
    rr=a['top_grasp_views_rot'].astype(float);angle=a['batch_grasp_rotations']*np.pi/12
    rx=np.zeros((1024,3,3));rx[:,0,0]=1;rx[:,1,1]=np.cos(angle);rx[:,2,2]=np.cos(angle);rx[:,2,1]=np.sin(angle);rx[:,1,2]=-np.sin(angle);rr=rr@rx
    rb=a['view_rotation'].astype(float);theta=np.arctan2((rr[:,:,1]*rb[:,:,2]).sum(1),(rr[:,:,1]*rb[:,:,1]).sum(1));roll=abs((theta-a['angle_index']*np.pi/12+np.pi/2)%np.pi-np.pi/2)
    df['reference_roll_in_prediction_frame_rad']=np.where(v,theta,np.nan);df['physical_roll_error_deg']=np.where(v,np.rad2deg(roll),np.nan)
    for j,k in enumerate('xyz'):df['ref_'+k+'_m']=np.where(v,a['batch_grasp_point'][:,j],np.nan)
    for j in range(9):df[f'ref_rotation_{j}']=np.where(v,rr.reshape(-1,9)[:,j],np.nan)
    df['P0_failed']=~df['P0_good'];df['nominal_high_target']=v&(a['batch_grasp_score']>=.699999)
    df['reference_good_actual_bad']=v&df['R_labelpoint_good']&~df['P0_good']
    for key in vars[1:]:
     df[key+'_rescues']=v&~df['P0_good']&df[key+'_good'];df[key+'_harms']=v&df['P0_good']&~df[key+'_good']
    df['any_P1_P5_rescues']=df[[x+'_rescues' for x in ['P1','P2','P3','P4','P5']]].any(axis=1)
    df['H1_P5_witness']=df['reference_good_actual_bad']&df['P5_good']
    df['H1_P5_aligned_witness']=df['reference_good_actual_bad']&df['P5_aligned_good']
    df['H1_head_correct_witness']=df['H1_P5_witness']&df['nominal_high_target']&df['quality_class_correct']
    df['H1_aligned_head_correct_witness']=df['H1_P5_aligned_witness']&df['nominal_high_target']&df['quality_class_correct']
    df['invalid_reason']=np.where(v,'valid',np.where(~df['valid_point']&~df['valid_view'],'point_and_view',np.where(~df['valid_point'],'point_distance','view_unmatched')))
   evidence.append(df)

df=pd.concat(evidence,ignore_index=True);train=df[df['split']=='train'].copy().infer_objects();Aframes=[r for r in frames if r['split']=='C3'];Aobj=[r for r in objects if r['split']=='C3']
A=dict(frames=48,raw_candidates=49152,known_quality_candidates=int((df['split']=='C3').sum()),final_candidates=sum(r['final_count'] for r in Aframes),top1=macro(Aframes,'top_feasible','C3'),top1_count=sum(r['top_feasible'] for r in Aframes),top1_categories=dict(collections.Counter(r['top_category'] for r in Aframes)),noCD_top1=macro(Aframes,'noCD_feasible','C3'),noCD_top1_count=sum(r['noCD_feasible'] for r in Aframes),noCD_categories=dict(collections.Counter(r['noCD_category'] for r in Aframes)),failed_with_good_final=sum(not r['top_feasible'] and r['top_failure_has_good_final'] for r in Aframes),failed_with_same_object_good_final=sum(not r['top_feasible'] and r['top_failure_has_same_object_good_final'] for r in Aframes),postprocessing_changed_top=sum(r['postprocessing_top_changed'] for r in Aframes),inversion=macro(Aobj,'inversion_rate','C3'),score_quality_rho=macro(Aobj,'rho','C3'),mixed_object_frames=sum('inversion_rate' in r for r in Aobj),EG_B_top1=macro(Aframes,'EG_B_feasible','C3'),full_minus_EGB=macro(Aframes,'full_minus_EGB','C3'))
A['controls']={k:dict(success_count=sum(r[k+'_feasible'] for r in Aframes),delta=macro(Aframes,k+'_delta','C3')) for k in ['centered','softmax','width_no_scale']}
valid=train['valid_match'].astype(bool);high=train['final_rank'].gt(0);failed=~train['P0_good'];same=train['owner_consistent'].astype(bool)
train['always']=True
Bstat=dict(raw_candidates=len(train),valid_matches=int(valid.sum()),coverage=count_rate(train,'valid_match',train['always']),invalid_reasons=train.loc[~valid,'invalid_reason'].value_counts().to_dict(),high_rank_candidates=int(high.sum()),valid_high_rank=int((valid&high).sum()),all_high_rank_failures=int((high&failed).sum()),valid_high_rank_failures=int((valid&high&failed).sum()),valid_top1=int((valid&(train.final_rank==1)).sum()),owner_mismatch_valid=int((valid&~same).sum()))
Bstat['rates']={}
for scope,mask in [('all_valid',valid),('high_rank_valid',valid&high),('all_high_failures',high&failed),('valid_high_failures',valid&high&failed),('valid_high_failures_same_object',valid&high&failed&same),('valid_all_failures',valid&failed)]:
 Bstat['rates'][scope]={event:count_rate(train,event,mask) for event in ['reference_good_actual_bad','P5_rescues','P5_aligned_rescues','any_P1_P5_rescues','H1_P5_witness','H1_P5_aligned_witness','H1_head_correct_witness']}
Bstat['parameter_deviation']={k:dict(median=float(train.loc[valid,k].median()),q25=float(train.loc[valid,k].quantile(.25)),q75=float(train.loc[valid,k].quantile(.75))) for k in ['angle_error_deg','physical_roll_error_deg','depth_error_mm','width_error_mm','view_error_deg','match_distance_m']}
Bstat['counterfactuals']={}
for name in vars[1:]:
 Bstat['counterfactuals'][name]=dict(recovery_high_failures=count_rate(train,name+'_rescues',valid&high&failed),recovery_all_failures=count_rate(train,name+'_rescues',valid&failed),harm_high_success=count_rate(train,name+'_harms',valid&high&~failed),quality_good=count_rate(train,name+'_good',valid&high))
groups=[]
for key,g in train[valid].groupby(['scene','frame','object_id','target_score']):
 correct=g[g.AD_correct];wrong=g[g.AD_error]
 if len(correct) and len(wrong):groups.append(dict(scene=int(key[0]),failure_risk_difference=float(wrong.P0_failed.mean()-correct.P0_failed.mean()),n_correct=len(correct),n_wrong=len(wrong)))
Bstat['AD_error_controlled']=dict(groups=len(groups),n_correct=sum(x['n_correct'] for x in groups),n_wrong=sum(x['n_wrong'] for x in groups),risk_difference=macro(groups,'failure_risk_difference','train'),strata='scene/frame/object/exact nominal target score; wrong angle or depth versus both correct')
Bstat['AD_strata']={}
for label,m in [('AD_correct',valid&high&train.AD_correct),('AD_error',valid&high&train.AD_error),('AD_correct_width_within_2mm',valid&high&train.AD_correct&(train.width_error_mm<=2)),('AD_correct_width_within_5mm',valid&high&train.AD_correct&(train.width_error_mm<=5))]:
 Bstat['AD_strata'][label]=dict(failure=count_rate(train,'P0_failed',m),P5_recovery=count_rate(train,'P5_rescues',m&failed),aligned_recovery=count_rate(train,'P5_aligned_rescues',m&failed))
cor=[]
for key,g in train[valid].groupby(['scene','frame','object_id']):
 if len(g)<4:continue
 u0=utility(g.P0_mu.to_numpy());ur=utility(g.R_labelpoint_mu.to_numpy());sc=g.original_score.to_numpy()
 if np.ptp(u0)>0 and np.ptp(ur)>0 and np.ptp(sc)>0:
  r0=float(spearmanr(sc,u0).statistic);rr=float(spearmanr(sc,ur).statistic);cor.append(dict(scene=int(key[0]),actual_rho=r0,reference_rho=rr,delta=rr-r0))
Bstat['score_target_vs_actual']={k:macro(cor,k,'train') for k in ['actual_rho','reference_rho','delta']};Bstat['score_target_vs_actual']['groups']=len(cor)
Bstat['head_accuracy']=dict(all_valid=count_rate(train,'quality_class_correct',valid),high_rank_valid=count_rate(train,'quality_class_correct',valid&high))
obs={}
for name,event in [('P5','P5_rescues'),('P5_aligned','P5_aligned_rescues'),('R_labelpoint','reference_good_actual_bad')]:
 g=train.loc[valid&high&train[event]];dist=g[[name+'_contact1_obs_dist_m',name+'_contact2_obs_dist_m']].max(axis=1,skipna=False)
 obs[name]=dict(candidates=len(g),contact_pairs_available=int(dist.notna().sum()),both_contacts_within_5mm=int((dist<=.005).sum()),both_contacts_within_10mm=int((dist<=.010).sum()),at_least_one_farther_than_10mm=int((dist>.010).sum()),limitation='Nearest distance to exact 20k input points, including other objects; geometric support proxy, not proof of visibility or learnability')
Bstat['observability']=obs
scene_rows=[]
for s,g in train.groupby('scene'):
 vh=g.valid_match&g.final_rank.gt(0);bad=~g.P0_good;scene_rows.append(dict(scene=int(s),raw=len(g),valid=int(g.valid_match.sum()),high=int(g.final_rank.gt(0).sum()),high_fail=int((g.final_rank.gt(0)&bad).sum()),valid_high_fail=int((vh&bad).sum()),reference_good_actual_bad=int((vh&g.reference_good_actual_bad).sum()),P5_rescues=int((vh&g.P5_rescues).sum()),aligned_rescues=int((vh&g.P5_aligned_rescues).sum()),H1_P5_witness=int((vh&g.H1_P5_witness).sum()),H1_aligned_witness=int((vh&g.H1_P5_aligned_witness).sum())))
Bstat['by_scene']=scene_rows
Bstat['replicated_P5_scenes']=sum(x['H1_P5_witness']>0 for x in scene_rows);Bstat['replicated_P5_objects']=int(train.loc[valid&high&train.H1_P5_witness,'object_id'].nunique())
supp={}
supp['coverage_high_rank']=count_rate(train,'valid_match',high)
supp['coverage_high_failures']=count_rate(train,'valid_match',high&failed)
supp['coverage_high_success']=count_rate(train,'valid_match',high&~failed)
supp['actual_success_valid']=count_rate(train,'P0_good',valid)
supp['actual_success_invalid']=count_rate(train,'P0_good',~valid)
supp['nominal_high_reference_recheck']=count_rate(train,'R_labelpoint_good',valid&train.nominal_high_target)
supp['nominal_high_reference_high_rank_recheck']=count_rate(train,'R_labelpoint_good',valid&high&train.nominal_high_target)
supp['aligned_head_correct_witness_all_high_failures']=count_rate(train,'H1_aligned_head_correct_witness',high&failed)
supp['aligned_head_correct_witness_valid_high_failures']=count_rate(train,'H1_aligned_head_correct_witness',valid&high&failed)
supp['aligned_witness_scenes']=int(train.loc[valid&high&train.H1_P5_aligned_witness,'scene'].nunique())
supp['aligned_witness_objects']=int(train.loc[valid&high&train.H1_P5_aligned_witness,'object_id'].nunique())
supp['physical_roll_strata']={}
for lo,hi in [(0,7.5),(7.5,15),(15,30),(30,90.0001)]:
 m=valid&high&train.physical_roll_error_deg.ge(lo)&train.physical_roll_error_deg.lt(hi)
 supp['physical_roll_strata'][f'{lo:g}-{hi:g}deg']={'failure':count_rate(train,'P0_failed',m),'aligned_recovery':count_rate(train,'P5_aligned_rescues',m&failed)}
supp['recovery_overlap']={}
for event in ['P5_rescues','P5_aligned_rescues','H1_P5_aligned_witness']:
 m=valid&high&train[event]
 supp['recovery_overlap'][event]={'n':int(m.sum()),'also_width_only':int((m&train.P3_good).sum()),'also_remove_width_scale':int((m&train.width_no_scale_good).sum()),'also_physical_AD_only':int((m&train.P4_aligned_good).sum()),'also_class_AD_only':int((m&train.P4_good).sum()),'also_physical_angle_only':int((m&train.P1_aligned_good).sum())}
supp['physical_counterfactual_union']={}
train['any_physical_rescue']=train[['P1_aligned_rescues','P2_rescues','P3_rescues','P4_aligned_rescues','P5_aligned_rescues']].any(axis=1)
for label,m in [('valid_high_failures',valid&high&failed),('all_high_failures',high&failed)]:supp['physical_counterfactual_union'][label]=count_rate(train,'any_physical_rescue',m)
supp['near_view_witness']=count_rate(train,'H1_P5_aligned_witness',valid&high&failed&train.view_error_deg.le(5))
supp['P5_aligned_witness_categories']=train.loc[valid&high&train.H1_P5_aligned_witness,'P0_category'].value_counts().to_dict()
supp['top1_training']={'all':int((train.final_rank==1).sum()),'feasible':int(((train.final_rank==1)&train.P0_good).sum()),'failed':int(((train.final_rank==1)&failed).sum()),'valid_failed':int(((train.final_rank==1)&failed&valid).sum()),'aligned_rescued':int(((train.final_rank==1)&train.P5_aligned_rescues).sum()),'H1_aligned_witness':int(((train.final_rank==1)&train.H1_P5_aligned_witness).sum())}
supp['nominal_target_reference_rho']={}
supp['notes']='Supplementary descriptive decomposition on frozen samples; physical-angle controls added after identifying basis mismatch, not substituted for pre-specified class-index primary. All-high-failure witness counts are verified lower bounds; invalid references remain unknown.'
Bstat['supplementary']=supp
casecols=['scene','frame','candidate_id','object_id','final_rank','original_score','target_score','quality_class_correct','angle_error_deg','physical_roll_error_deg','depth_error_mm','width_error_mm','view_error_deg','P0_mu','P1_mu','P2_mu','P3_mu','P4_mu','P5_mu','P5_aligned_mu','R_labelpoint_mu','width_no_scale_mu','P0_category','match_distance_m']
cases={}
for name,m in [('rescued',valid&high&train.H1_P5_witness),('unrescued',valid&high&train.reference_good_actual_bad&~train.P5_good),('harmed',valid&high&train.P5_harms),('physical_rescued',valid&high&train.H1_P5_aligned_witness),('physical_unrescued',valid&high&train.reference_good_actual_bad&~train.P5_aligned_good),('physical_harmed',valid&high&train.P5_aligned_harms)]:
 cases[name]=train.loc[m].sort_values(['final_rank','scene','frame','candidate_id']).drop_duplicates('scene').head(3)[casecols].to_dict('records')
verification=dict(frame_count=len(frames),all_final_labels_known=True,all_B_actual_labels_known=True,all_valid_counterfactuals_known=True,geometry_hashes_match=True,positive_quality_collision_conflicts=0,probe_valid_changes=sum(x.get('probe_valid_mask_changes',0) for x in metas),probe_target_changes=sum(x.get('probe_target_changes',0) for x in metas),probe_matched_point_changes=sum(x.get('probe_matched_point_changes',0) for x in metas),probe_reference_orth_error=max(x.get('reference_rotation_orthogonality_error',0) for x in metas),extension_triggered=False)
stats=dict(configuration=plan,A=A,B=Bstat,verification=verification,frames=frames,object_groups=objects,cases=cases,statistical_protocol='Scene-macro estimates; 10,000 stratified scene bootstrap resamples; C3 seen/similar/novel strata, training scene quartiles. Candidate totals descriptive only. Unknowns excluded, invalid references never labelled failures.')
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
stats=clean(stats)
(O/'EG_pose_quality_statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
df.to_csv(O/'EG_pose_quality_evidence.csv',index=False,encoding='utf-8-sig',float_format='%.9g',na_rep='')
with open(O/'EG_pose_quality_evidence.csv','rb') as src,gzip.open(O/'EG_pose_quality_evidence.csv.gz','wb') as dest:dest.write(src.read())
print(json.dumps(dict(A=A,B={k:v for k,v in Bstat.items() if k not in ['counterfactuals','rates','by_scene']},high_failure_rates=Bstat['rates']['all_high_failures'],verification=verification),ensure_ascii=False,indent=2,default=str))
