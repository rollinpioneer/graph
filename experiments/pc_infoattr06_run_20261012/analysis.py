"""CPU-only frozen prototype readout and input-command-response attribution."""
import os,pathlib,json,time,itertools,hashlib,csv
import numpy as np
from scipy.optimize import linear_sum_assignment
R=pathlib.Path(os.environ['R']);P=pathlib.Path(os.environ['P'])
start=time.monotonic()
def save(name,x):
 (R/'results'/name).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def table(name,rows):
 if rows:
  with (R/'results'/name).open('w',newline='') as f:
   w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
palette=np.array([[.6,.1,0],[0,.6,.1],[0,.1,.8],[.7,.7,0],[.5,0,.5],[0,.9,.9]])
palette/=palette.max(1)[:,None]
cameras=[np.array([.8,0,1.8]),np.array([0,-.8,1.65])];targets=[np.array([.12,0,1.025]),np.array([0,-.12,1.025])]
corners=np.array(list(itertools.product([-1,1],repeat=3)))*.0175
def project(x,v):
 f=targets[v]-cameras[v];f/=np.linalg.norm(f);right=np.cross(f,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,f);d=x-cameras[v];dep=d@f
 return np.stack([64+64/np.tan(np.deg2rad(35)/2)*(d@right)/dep,64-64/np.tan(np.deg2rad(35)/2)*(d@up)/dep],-1)
def classes(im):
 a=np.moveaxis(im,0,-1).astype(float);mx=a.max(-1);norm=a/np.maximum(mx[...,None],1);d=np.linalg.norm(norm[:,:,None,:]-palette,axis=-1);best=d.argmin(-1);best[(mx-a.min(-1)<=40)|(mx<=40)|(d.min(-1)>.4)]=-1;return best
def refs(im,xyz,v):
 cl=classes(im);rows=[]
 for point in xyz:
  pp=project(point+corners,v);lo=np.floor(pp.min(0)-2).astype(int).clip(0,127);hi=np.ceil(pp.max(0)+2).astype(int).clip(0,127)
  sl=(slice(lo[1],hi[1]+1),slice(lo[0],hi[0]+1));q=cl[sl];rows.append((lo,q))
 return rows
def identity(z):
 xyz=np.c_[z['goal'],np.full(len(z['goal']),1.0425)];score=np.zeros((len(xyz),6))
 for v in range(2):
  for j,(_,q) in enumerate(refs(z['goal_image'],xyz,v) if False else refs(z['goal_image'][v],xyz,v)):
   score[j]+=[(q==c).sum() for c in range(6)]
 row,col=linear_sum_assignment(-score);ans=np.zeros(len(xyz),int);ans[row]=col;return ans,score
cache={};sources=[]
def load(name,new=False):
 key=(name,new)
 if key not in cache:
  root=R if new else P;path=root/'runs'/name/'data';file=path/'trajectory.npz'
  with np.load(file) as z:out={k:z[k] for k in z.files}
  with np.load(path/'initial.npz') as z:out['goal_image']=z['goal_image']
  cache[key]=out;sources.append({'run':name,'path':str(file),'sha256':sha(file),'new_normal':new})
 return cache[key]
freeze=json.loads((R/'receipts/preanalysis_freeze.json').read_text())
cal=[];labels=[];calrows=[]
for name in freeze['calibration']:
 z=load(name);ids,scores=identity(z)
 for t in freeze['calibration_frames']+[None]:
  goal=t is None;xyz=np.c_[z['goal'],np.full(len(ids),1.0425)] if goal else z['states'][t,:,:3];im=z['goal_image'] if goal else z['RGB'][t];rep=z['used_goal_particles'][0] if goal else z['used_current_particles'][t]
  for v in range(2):
   xy=np.c_[(rep[v,:,1]+1)*127/2,(rep[v,:,0]+1)*127/2];d=np.linalg.norm(xy[:,None]-project(xyz,v)[None],axis=-1);j=d.argmin(1);cl=classes(im[v])
   for k in range(len(xy)):
    x,y=np.rint(xy[k]).astype(int).clip(0,127);patch=cl[max(0,y-2):min(128,y+3),max(0,x-2):min(128,x+3)];ok=(patch==ids[j[k]]).sum()>=3
    if rep[v,k,-1]>.1 and d[k,j[k]]<=8 and ok:
     cal.append(rep[v,k,5:9]);labels.append(int(ids[j[k]]));calrows.append({'run':name,'step':t,'view':v,'particle':k,'color':int(ids[j[k]]),'projection_error_px':float(d[k,j[k]])})
cal=np.array(cal);labels=np.array(labels);scale=np.maximum(cal.std(0),.1)
protos=np.array([np.median(cal[labels==c],axis=0) if (labels==c).any() else np.zeros(4) for c in range(6)])
present=np.array([(labels==c).any() for c in range(6)])
save('P1_calibration_labels.json',calrows)
save('P1_frozen_rule.json',{'prototypes':protos.tolist(),'feature_scale':scale.tolist(),'per_color_counts':[(labels==c).sum().item() for c in range(6)],'uses':['input particle fields5:9','input x/y fields0:2','input obj_on'],'truth_or_RGB_at_inference':False,'fit':'one median appearance prototype per color; pooled standardization; no outcome labels','tolerance_px':10,'max_feature_distance':3,'min_gap':.15,'source_sha':sha(__file__)})
def read(rep,v):
 features=rep[:,5:9];d=np.linalg.norm((features[:,None]-protos[None])/scale,axis=-1);d[:,~present]=1e9;order=d.argsort(1);best=order[:,0];dist=d[np.arange(len(rep)),best];gap=d[np.arange(len(rep)),order[:,1]]-dist
 valid=(rep[:,-1]>.1)&(dist<=3)&(gap>=.15);xy=np.c_[(rep[:,1]+1)*127/2,(rep[:,0]+1)*127/2];out={}
 for c in range(6):
  ix=np.flatnonzero(valid&(best==c))
  if len(ix):
   weights=rep[ix,-1]/np.maximum(dist[ix],.2);center=np.average(xy[ix],axis=0,weights=weights)
   out[c]={'xy_px':center.tolist(),'particles':ix.tolist(),'spread_px':float(np.sqrt(np.average(np.square(xy[ix]-center).sum(1),weights=weights))),'appearance_distance':float(dist[ix].min())}
 return out
rows=[];predictions=[]
def score(name,steps,role,new=False):
 z=load(name,new);ids,identityscores=identity(z)
 for t in steps+[None]:
  goal=t is None;xyz=np.c_[z['goal'],np.full(len(ids),1.0425)] if goal else z['states'][t,:,:3];im=z['goal_image'] if goal else z['RGB'][t];rep=z['used_goal_particles'][0] if goal else z['used_current_particles'][t]
  # Predictions are computed and sealed before truth/reference scoring.
  pred=[read(rep[v],v) for v in range(2)];predictions.append({'run':name,'step':t,'role':role,'predictions':pred})
  for v in range(2):
   for j,(lo,q) in enumerate(refs(im[v],xyz,v)):
    c=int(ids[j]);yy,xx=np.where(q==c);ref=np.array([xx.mean()+lo[0],yy.mean()+lo[1]]) if len(xx)>=5 else None;est=pred[v].get(c);err=float(np.linalg.norm(np.array(est['xy_px'])-ref)) if est is not None and ref is not None else None
    rows.append({'run':name,'role':role,'step':t,'goal':goal,'n':len(ids),'view':v,'object':j,'color':c,'visible_pixels':len(xx),'visible':ref is not None,'readable':est is not None,'error_visible_centroid_px':err,'error_projection_px':float(np.linalg.norm(np.array(est['xy_px'])-project(xyz[j:j+1],v)[0])) if est else None,'within_10px':bool(err is not None and err<=10),'spread_px':est['spread_px'] if est else None,'pending':bool(not goal and z['distances'][t,j]>=.04),'identity_reference_combined_pixels':float(identityscores[j,c])})
for name in freeze['holdout']:score(name,freeze['holdout_frames'],'independent_holdout')
hold=[q for q in rows if q['role']=='independent_holdout' and q['visible']];health=sum(q['within_10px'] for q in hold)/max(1,len(hold));healthy=health>=.8 and present.all()
save('P1_health_gate.json',{'visible_object_view_frames':len(hold),'within_10px_fraction':health,'threshold':.8,'all_colors_calibrated':bool(present.all()),'gate_passed':bool(healthy),'decision':'diagnostic errors eligible as reliability clues only' if healthy else 'stop strong encoding attribution; reader selfcheck failed; preserve descriptive outputs','not_information_theoretic_absence_test':True})
for name in freeze['diagnostic']:score(name,freeze['diagnostic_fixed_frames'],'discovery')
newnames=[]
for f in sorted((R/'runs').glob('normal*/data/completed.json')):
 name=f.parent.parent.name;newnames.append(name);score(name,[0,50,100,150,199],'new_normal_review',True)
save('P1_readout_predictions.json',predictions);save('P1_scored_object_rows.json',rows);table('P1_scored_object_rows.csv',rows)
joined=[]
for name in sorted(set(q['run'] for q in rows)):
 for t in sorted(set(-1 if q['step'] is None else q['step'] for q in rows if q['run']==name)):
  for j in sorted(set(q['object'] for q in rows if q['run']==name)):
   rr=[q for q in rows if q['run']==name and (-1 if q['step'] is None else q['step'])==t and q['object']==j]
   joined.append({'run':name,'step':None if t==-1 else t,'object':j,'role':rr[0]['role'],'any_view_visible':any(q['visible'] for q in rr),'any_view_within_10px':any(q['within_10px'] for q in rr),'view_errors_px':[q['error_visible_centroid_px'] for q in rr],'scope':'multiview availability only; no metrically triangulated XY claim'})
save('P1_multiview.json',joined)
eventrows=[];summary=[]
def p2(name,new=False):
 z=load(name,new);st=z['states'];ee=z['eef'];dd=z['distances'];a=z['actions'];cg=z['controller_goal'];q=z['q'];qd=z['qd'];xyz=st[:,:,:3];dist=np.linalg.norm(ee[:,None,:]-xyz,axis=-1)
 expected=np.clip(ee[:-1]+np.clip(a,-1,1)*.125,[-.2675,-.3175,1.045],[.2675,.3175,1.06]);transform=float(np.abs(expected-cg).max());turns=[]
 for j in range(len(z['goal'])):
  local=[]
  for t in range(len(a)):
   vec=xyz[t,j]-ee[t];unit=vec/max(np.linalg.norm(vec),1e-9);request=cg[t]-ee[t];actual=ee[t+1]-ee[t];req=float(request@unit);act=float(actual@unit);prog=float(dd[t,j]-dd[t+1,j]);pending=bool(dd[t,j]>=.04);delta=float(dist[t+1,j]-dist[t,j]);cos=float(request@actual/max(np.linalg.norm(request)*np.linalg.norm(actual),1e-9))
   row={'run':name,'normal_sampling':new or name.startswith('supplement'),'t_input':t,'t_result':t+1,'object':j,'pending':pending,'goal_distance_before_m':float(dd[t,j]),'goal_distance_after_m':float(dd[t+1,j]),'goal_improvement_m':prog,'eef_object_before_m':float(dist[t,j]),'eef_object_after_m':float(dist[t+1,j]),'eef_distance_change_m':delta,'requested_toward_object_m':req,'actual_toward_object_m':act,'request_actual_cosine':cos,'raw_action_x':float(a[t,0]),'raw_action_y':float(a[t,1]),'raw_action_z':float(a[t,2]),'controller_goal_x':float(cg[t,0]),'controller_goal_y':float(cg[t,1]),'controller_goal_z':float(cg[t,2]),'actual_eef_delta_x':float(actual[0]),'actual_eef_delta_y':float(actual[1]),'actual_eef_delta_z':float(actual[2]),'q_delta_norm_rad':float(np.linalg.norm(q[t+1]-q[t])),'qd_before_norm':float(np.linalg.norm(qd[t])),'q_before':q[t].tolist(),'q_after':q[t+1].tolist(),'qd_before':qd[t].tolist(),'eef_before':ee[t].tolist(),'eef_after':ee[t+1].tolist(),'action_H5':z['generated_actions'][t].tolist(),'controller_transform_error_m':float(np.abs(expected[t]-cg[t]).max()),'fall_before':bool((st[t,:,2]<.975).any()),'classification':'requested_departure_followed' if req<-.002 and act<-.001 else 'requested_approach_actual_departure' if req>.002 and act<-.001 else 'approach_followed' if req>.002 and act>.001 else 'small_or_other'}
   eventrows.append(row);local.append(row)
  # Freeze event definitions; no claiming retreat is necessarily wrong.
  for t in range(5,len(a)-5):
   if dd[t,j]>=.04 and dist[t-5,j]-dist[t,j]>.015 and dist[t+5,j]-dist[t,j]>.015 and dist[t,j]<.15:
    turns.append({'object':j,'step':t,'before_distance_m':float(dist[t-5,j]),'near_distance_m':float(dist[t,j]),'after_distance_m':float(dist[t+5,j]),'before_improvement_m':float(dd[t-5,j]-dd[t,j]),'after_improvement_m':float(dd[t,j]-dd[t+5,j]),'post_requested_departure_steps':sum(x['requested_toward_object_m']<-.002 for x in local[t:t+5]),'post_actual_departure_steps':sum(x['actual_toward_object_m']<-.001 for x in local[t:t+5]),'post_approach_actual_departure_steps':sum(x['classification']=='requested_approach_actual_departure' for x in local[t:t+5])})
  last=np.flatnonzero(dd[:-1,j]-dd[1:,j]>.002);lastpush=int(last[-1]) if len(last) else None
  eligible=[x for x in local if x['pending'] and not x['fall_before']]
  summary.append({'run':name,'object':j,'normal_sampling':new or name.startswith('supplement'),'final_success':bool(dd[-1,j]<.04),'final_distance_m':float(dd[-1,j]),'last_effective_2mm_progress_action':lastpush,'minimum_eef_distance_m':float(dist[:,j].min()),'pending_prefall_steps':len(eligible),'requested_departure_followed':sum(x['classification']=='requested_departure_followed' for x in eligible),'requested_approach_actual_departure':sum(x['classification']=='requested_approach_actual_departure' for x in eligible),'controller_transform_max_error_m':transform})
 save('P2_events_'+name+'.json',{'turn_events':turns,'all_initial_to_terminal':True,'scene_scope':'new normal independent scene' if new else 'historical discovery only','definition':'5-step approach>15mm then leave>15mm; near<150mm; pending; this is event location, not bad-action label','index_audit':'input[t], action[t], controller_goal after step=t requested goal, states/eef[t+1] result; transform checked numerically'})
for name in ['validate6_permuted','validate6_blocked','validate3_permuted','validate3_blocked','supplement3','supplement6']:p2(name)
for name in newnames:p2(name,True)
table('P2_step_evidence.csv',eventrows);save('P2_object_summaries.json',summary);save('source_inputs.json',sources)
save('analysis_receipt.json',{'CPU_wall_seconds':time.monotonic()-start,'GPU_calls':0,'new_normal_runs':newnames,'reader_healthy':bool(healthy),'health_fraction':health,'source_sha':sha(__file__),'P1_fit_separate_scene':True,'no_policy_or_encoder_training':True})
print('CPU_ANALYSIS_COMPLETE',len(rows),len(eventrows),'HEALTH',health,healthy,flush=True)
