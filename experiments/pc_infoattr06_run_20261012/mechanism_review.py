"""Event contrasts frozen before opening new normal endpoints; no causal labels."""
import os,pathlib,json,time
import numpy as np
R=pathlib.Path(os.environ['R']);P=pathlib.Path(os.environ['P']);freeze=json.loads((R/'receipts/mechanism_review_freeze.json').read_text())
names=[('validate6_permuted',False),('validate6_blocked',False),('validate3_permuted',False),('validate3_blocked',False),('supplement3',False),('supplement6',False)]
names += [(f.parent.parent.name,True) for f in sorted((R/'runs').glob('normal*/data/completed.json'))]
records=[];summary=[];examples=[]
for name,new in names:
 root=R if new else P
 with np.load(root/'runs'/name/'data/trajectory.npz') as f:z={k:f[k] for k in f.files}
 st=z['states'];ee=z['eef'];dd=z['distances'];cg=z['controller_goal'];dist=np.linalg.norm(ee[:,None]-st[:,:,:3],axis=-1);fall=(st[:,:,2]<.975).any(1);unit=st[:-1,:,:3]-ee[:-1,None];unit/=np.maximum(np.linalg.norm(unit,axis=-1,keepdims=True),1e-9);req=cg-ee[:-1];act=np.diff(ee,axis=0);requested=np.einsum('tj,tij->ti',req,unit);actual=np.einsum('tj,tij->ti',act,unit)
 cos=np.einsum('ti,ti->t',req,act)/np.maximum(np.linalg.norm(req,axis=1)*np.linalg.norm(act,axis=1),1e-9);one=(dd>=.04).sum(1)==1
 for j in range(dd.shape[1]):
  candidate=[];execsus=[]
  for t in range(181):
   end=t+20
   if dd[t,j]<.04 or fall[t:end+1].any() or dist[t,j]>=.075:continue
   progress=float(dd[t,j]-dd[t:end+1,j].min());leave=float(dist[end,j]-dist[t,j]);rs=float(np.maximum(-requested[t:end,j],0).sum());ac=float(np.maximum(-actual[t:end,j],0).sum());moving=(np.linalg.norm(req[t:end],axis=1)>.003)&(np.linalg.norm(act[t:end],axis=1)>.001);agree=float((cos[t:end][moving]>.8).mean()) if moving.any() else None
   opposite=((requested[t:end,j]>.002)&(actual[t:end,j]<-.001));row={'run':name,'normal_sampling':new or name.startswith('supplement'),'object':j,'start':t,'end':end,'pending_at_start':True,'only_pending_at_start':bool(one[t]),'eef_distance_before_m':float(dist[t,j]),'eef_distance_after_m':float(dist[end,j]),'max_goal_improvement_m':progress,'eef_distance_increase_m':leave,'requested_retreat_positive_path_m':rs,'actual_retreat_positive_path_m':ac,'direction_agreement_moving_fraction':agree,'approach_request_depart_response_steps':int(opposite.sum()),'moving_steps':int(moving.sum()),'final_object_success':bool(dd[-1,j]<.04),'final_object_error_m':float(dd[-1,j])}
   cmd=progress<.002 and leave>=.04 and rs>=.04 and ac>=.04 and agree is not None and agree>=.8
   execution=progress<.002 and leave>=.04 and opposite.sum()>=3
   row.update(command_expressed_retreat_candidate=bool(cmd),execution_opposition_candidate=bool(execution));records.append(row)
   if cmd:candidate.append(row)
   if execution:execsus.append(row)
  summary.append({'run':name,'object':j,'normal_sampling':new or name.startswith('supplement'),'final_success':bool(dd[-1,j]<.04),'final_error_m':float(dd[-1,j]),'command_candidate_windows':len(candidate),'execution_candidate_windows':len(execsus),'first_command_candidate':candidate[0] if candidate else None,'first_execution_candidate':execsus[0] if execsus else None,'scope':'overlapping20step windows; not independent observations; retreat may be legitimate repositioning'})
  examples.extend(candidate[:1]);examples.extend(execsus[:1])
(R/'results/P2_mechanism_candidates.json').write_text(json.dumps(records,indent=2));(R/'results/P2_mechanism_review_summary.json').write_text(json.dumps(summary,indent=2));(R/'results/P2_first_candidate_per_object.json').write_text(json.dumps(examples,indent=2))
print('MECHANISM_REVIEW',[(q['run'],q['object'],q['command_candidate_windows'],q['execution_candidate_windows'],q['final_success']) for q in summary if q['command_candidate_windows'] or q['execution_candidate_windows']],flush=True)
