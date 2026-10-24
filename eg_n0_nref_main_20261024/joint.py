"""Frozen detectors, I1/PCR angle selection and full original-pool postprocessing."""
import sys,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pipeline as P
R=P.R
CFG=json.loads((R/'frozen_config.json').read_text()) if (R/'frozen_config.json').exists() else None
def choose(risk,valid,k):
 v=valid&np.isfinite(risk)
 if not v.any():return int(k),True
 ix=np.flatnonzero(v);ties=ix[np.isclose(risk[ix],risk[ix].min(),rtol=0,atol=1e-7)];dist=np.minimum((ties-k)%12,(k-ties)%12)
 return int(k if k in ties else ties[np.lexsort((ties,dist))][0]),False
def status(mu,c,e):
 if e:return 'empty'
 if c:return 'collision'
 if not np.isfinite(mu):return 'unknown'
 if 0<mu<=.4:return 'feasible'
 if mu>0:return 'weak'
 return 'no_FC'
def mask(a,nv,op,method):
 if method=='EG':return np.r_[a['EG_cd'].ravel(),a['raw_EG']]
 config=CFG['workpoints'][op]['methods'][method];key=config['key']
 if key.startswith(('ref_','nv_')):
  family,k=key.split('_');z=a if family=='ref' else nv;score=z[family+'_occ']+float(k)*z[family+'_unk']
 else:score=a[key]
 return score>config['threshold']
def run(task):
 _,s,f=task;out=R/f'joint/{s:04d}_{f:04d}.json'
 if out.exists():return
 start=time.time();a=np.load(R/f'cpu/C3_{s:04d}_{f:04d}.npz');nv=np.load(R/f'nv/C3_{s:04d}_{f:04d}.npz');v=np.load(P.ANGLE/f'evaluation/{s:04d}_{f:04d}.npz');ctrl=np.load(P.CONTROL/f'results/{s:04d}_{f:04d}.npz');ev=np.load(P.OLD/f'evaluation/C3_{s:04d}_{f:04d}.npz')
 n=len(a['candidate_id']);count=n*12;ids=a['candidate_id'];raw=a['raw'];nom=a['original_angle'];labels={int(cid):j for j,cid in enumerate(ids)};owner=a['owner']
 direct=P.E.Direct(s);direct.frame(f);rawcoll,rawempty=P.official_labels(direct,raw,owner)
 mu=ev['P0_mu'].copy();groups=[];pending=set()
 for op in ['OP1','OP2']:
  best=CFG['workpoints'][op]['best_simple']
  for group,detector in [('O','EG'),('B','EG'),('N','N_ref'),('N+','N_ref'),('N_rule',best),('E','GT'),('O_N','N_ref')]:
   allmask=np.r_[v['collision'].ravel(),rawcoll] if detector=='GT' else mask(a,nv,op,detector)
   variants=allmask[:count].reshape(n,12);rawmask=allmask[count:].copy();angles=nom.copy();fallback=np.zeros(n,bool)
   if group not in ['O','O_N']:
    objective=v['I1']+ctrl['RC']+ctrl['RS'] if group=='N+' else v['I1']
    for j,k in enumerate(nom):angles[j],fallback[j]=choose(objective[j],~variants[j],k)
   pred=raw.copy();pred[ids]=a['pred'][np.arange(n),angles];rawmask[ids]=variants[np.arange(n),angles]
   ni,oi,final=P.E.postprocess(pred,rawmask,owner)
   if group=='O':assert np.array_equal(final,ev['final_ids']),'O postprocess replay'
   if group=='B':
    old=json.loads((P.CONTROL/f'results/{s:04d}_{f:04d}.json').read_text());prev=next(x for x in old['postprocess'] if x['group']=='B');assert final.tolist()==prev['final_ids'],'B postprocess replay'
   for cid in final[:10]:
    if int(cid) not in labels and not np.isfinite(mu[cid]) and not rawcoll[cid]:pending.add(int(cid))
   groups.append(dict(op=op,group=group,detector=detector,angles=angles.tolist(),fallback=fallback.tolist(),post_cd=int((~rawmask).sum()),nms_ids=ni.tolist(),object10_ids=oi.tolist(),final_ids=final.tolist()))
 if pending:
  newids=np.array(sorted(pending));r=direct.score(raw[newids]);mu[newids]=r['mu']
  assert np.array_equal(r['collision'],rawcoll[newids]) and np.array_equal(r['empty'],rawempty[newids])
 else:newids=np.array([],int)
 def label(cid,angles):
  if cid in labels:
   j=labels[cid];k=angles[j];u=float(v['mu'][j,k]);c=bool(v['collision'][j,k]);e=bool(v['empty'][j,k]);source='existing_angle'
  else:u=float(mu[cid]);c=bool(rawcoll[cid]);e=bool(rawempty[cid]);source='direct_extra' if cid in pending else 'existing_original_or_collision_only'
  return dict(candidate_id=int(cid),mu=u,collision=c,empty=e,status=status(u,c,e),source=source)
 for g in groups:
  g['final_results']=[label(cid,g['angles']) for cid in g['final_ids']];g['top1']=g['final_results'][0] if g['final_results'] else dict(status='no_candidate');g['fixed_top1']=label(int(ev['final_ids'][0]),g['angles'])
 result=dict(scene=s,frame=f,groups=groups,new_FC_ids=newids.tolist(),new_FC_count=len(newids),wall_seconds=time.time()-start,frozen_config_sha256=P.sha(R/'frozen_config.json'))
 out.write_text(json.dumps(result));print(s,f,'newFC',len(newids),round(time.time()-start,2),flush=True)
if __name__=='__main__':
 assert CFG is not None;(R/'joint').mkdir(exist_ok=True)
 with ProcessPoolExecutor(int(sys.argv[2])) as pool:list(pool.map(run,P.tasks('C3')[int(sys.argv[1])::3]))
