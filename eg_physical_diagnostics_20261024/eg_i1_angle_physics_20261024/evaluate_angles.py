import sys,json,time,hashlib,os
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'eg_pose_quality_diagnosis_v1_20261022'
R0=ROOT.parent/'eg_pcr_r0_20261024'
sys.path.insert(0,str(OLD));import evaluate as E
SCENES=[100,106,115,124,132,139,150,157,160,169,176,186]
FRAMES=[0,64,128,192]

def run(task):
 s,f=task;dest=ROOT/'evaluation';dest.mkdir(exist_ok=True);out=dest/f'{s:04d}_{f:04d}.npz'
 if out.exists():return dict(scene=s,frame=f,cached=True)
 start=time.time();cpu=time.process_time()
 cp=OLD/f'cache/C3_{s:04d}_{f:04d}.npz';ep=OLD/f'evaluation/C3_{s:04d}_{f:04d}.npz'
 a=np.load(cp);e=np.load(ep);ids=e['final_ids'];ids=ids[(e['P0_mu'][ids]>0)&~e['P0_collision'][ids]&~e['P0_empty'][ids]]
 pred=a['pred'][ids].astype(np.float64);n=len(ids);angles=a['angle_index'][ids];rows=np.repeat(pred,12,axis=0);classes=np.tile(np.arange(12),n)
 delta=(classes-np.repeat(angles,12))*np.pi/12
 rows[:,4:13]=(np.repeat(pred[:,4:13].reshape(n,3,3),12,axis=0)@E.rx(delta)).reshape(-1,9)
 nominal=np.arange(n)*12+angles;rows[nominal]=pred
 assert np.array_equal(rows[:,13:16],np.repeat(pred[:,13:16],12,axis=0))
 assert np.array_equal(rows[:,:4],np.repeat(pred[:,:4],12,axis=0))
 assert np.allclose(rows[:,4:13].reshape(-1,3,3)[:,:,0],np.repeat(pred[:,4:13].reshape(n,3,3)[:,:,0],12,axis=0),atol=1e-12,rtol=0)
 direct=E.Direct(s);direct.frame(f);owners=direct.owners(rows)
 assert np.array_equal(owners,np.repeat(e['P0_owner'][ids],12))
 N=len(rows);mu=np.full(N,np.nan);collision=np.zeros(N,bool);empty=np.zeros(N,bool);risk=np.full(N,np.nan);reason=np.full(N,'',dtype='<U100');cr=np.full(N,'',dtype='<U100');points=np.full((N,2,3),np.nan);normals=np.full((N,2,3),np.nan)
 for o in np.unique(owners):
  allix=np.flatnonzero(owners==o)
  for begin in range(0,len(allix),64):
   ix=allix[begin:begin+64];gl=[np.zeros((0,17)) for _ in direct.models];gl[o]=rows[ix]
   cs,es,gs=E.EU.collision_detection(gl,direct.mt,direct.dex,direct.poses,direct.allpts,outlier=.05,empty_thresh=10,return_dexgrasps=True)
   for j,k in enumerate(ix):
    collision[k]=bool(cs[o][j]);empty[k]=bool(es[o][j]);g=gs[o][j]
    if collision[k]:mu[k]=-1
    elif g is None:reason[k]='dexgrasp_missing'
    else:
     try:mu[k]=float(E.EU.get_grasp_score(g,direct.dex[o],E.FR[::-1],direct.fc))
     except Exception as ex:reason[k]='evaluation:'+type(ex).__name__
    if g is None:cr[k]='dexgrasp_missing';continue
    try:
     ok,ct=g.close_fingers(direct.dex[o],check_approach=False)
     if not ok or len(ct)!=2:cr[k]='close_fingers_failed';continue
     pts=np.stack([v.point for v in ct]);ns=np.stack([v.normal for v in ct]);axis=pts[1]-pts[0];length=np.linalg.norm(axis)
     if length<1e-12 or not np.isfinite(ns).all():cr[k]='invalid_contact';continue
     axis/=length;cos=np.array([np.dot(-axis,ns[0]),np.dot(axis,ns[1])]);risk[k]=np.mean(.5*(1-cos));points[k]=pts;normals[k]=ns;cr[k]='ok'
    except Exception as ex:cr[k]='contact:'+type(ex).__name__
 # Each nominal pose independently re-evaluated, not copied labels.
 assert np.array_equal(mu[nominal],e['P0_mu'][ids]),(s,f,'mu replay',mu[nominal],e['P0_mu'][ids])
 assert np.array_equal(collision[nominal],e['P0_collision'][ids]) and np.array_equal(empty[nominal],e['P0_empty'][ids])
 previous={r['candidate_id']:r for r in json.loads((R0/f'contacts/{s:04d}_{f:04d}.json').read_text())['rows']}
 previous_risk=np.array([previous[int(k)]['first_contact_risk'] for k in ids]);error=float(np.max(abs(previous_risk-risk[nominal])))
 # Contact normals differ at ~1e-9 across hosts; categorical physical labels
 # above remain exact. This tolerance is numerical, not a quality threshold.
 assert error<1e-7,(s,f,'contact replay',error)
 np.savez_compressed(out,candidate_id=ids,original_angle=angles,angle_logits=a['angle_logits'][:,ids],score=pred[:,0],original_mu=e['P0_mu'][ids],object_id=e['object_id'][ids],mu=mu.reshape(n,12),collision=collision.reshape(n,12),empty=empty.reshape(n,12),I1=risk.reshape(n,12),unknown_reason=reason.reshape(n,12),contact_reason=cr.reshape(n,12),contact_points=points.reshape(n,12,2,3),contact_normals=normals.reshape(n,12,2,3),pred=rows.reshape(n,12,17))
 receipt=dict(scene=s,frame=f,n=n,variants=N,nominal_replay=True,I1_replay_error=error,cache_sha256=E.sha(cp),evaluation_sha256=E.sha(ep),result_sha256=E.sha(out),wall_seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,unknown=int(np.isnan(mu).sum()),contact_missing=int(np.isnan(risk).sum()),original_checkpoint_sha256=str(a['checkpoint_sha256']))
 out.with_suffix('.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True);return receipt

if __name__=='__main__':
 if sys.argv[1]=='pilot':run((100,0))
 else:
  tasks=[(s,f) for s in SCENES for f in FRAMES];tasks=tasks[int(sys.argv[1])::3]
  with ProcessPoolExecutor(int(sys.argv[2])) as pool:list(pool.map(run,tasks))
