import sys,json,time,hashlib,os,traceback
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'eg_pose_quality_diagnosis_v1_20261022'
SOURCE=ROOT.parent/'eg_i1_angle_physics_20261024'
N0=ROOT.parent/'eg_n0_nref_main_20261024'
sys.path.insert(0,str(OLD));import evaluate as E
SCENES=[100,106,115,124,132,139,150,157,160,169,176,186]
FRAMES=[32,96,160,224]
STOP=ROOT/'STOP_ON_MISMATCH'

def evaluate(s,f,rows,expected_owners=None):
 direct=E.Direct(s);direct.frame(f);owners=direct.owners(rows)
 if expected_owners is not None:assert np.array_equal(owners,expected_owners),(s,f,'owner mismatch')
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
 return dict(mu=mu,collision=collision,empty=empty,I1=risk,unknown_reason=reason,contact_reason=cr)

def save_json(path,obj):
 path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def replay():
 start=time.time();cpu=time.process_time()
 path=SOURCE/'evaluation/0100_0000.npz';a=np.load(path);rows=a['pred'].reshape(-1,17)
 r=evaluate(100,0,rows)
 checks={k:bool(np.array_equal(r[k],a[k].reshape(-1),equal_nan=True)) for k in ['mu','collision','empty']}
 old=a['I1'].reshape(-1);new=r['I1'];mask=np.isfinite(old)&np.isfinite(new)
 same_missing=bool(np.array_equal(np.isnan(old),np.isnan(new)))
 err=float(np.max(np.abs(old[mask]-new[mask]))) if mask.any() else None
 checks.update(I1_missing_mask_equal=same_missing,I1_max_absolute_error=err)
 passed=all(checks[k] for k in ['mu','collision','empty','I1_missing_mask_equal']) and err is not None and err<1e-7
 receipt=dict(passed=bool(passed),scene=100,frame=0,candidates=len(a['candidate_id']),variants=len(rows),checks=checks,input_sha256=E.sha(path),evaluator_sha256=E.sha(Path(__file__)),official_evaluate_sha256=E.sha(OLD/'evaluate.py'),source_evaluator_sha256=E.sha(SOURCE/'evaluate_angles.py'),wall_seconds=time.time()-start,cpu_seconds=time.process_time()-cpu)
 save_json(ROOT/'c3_replay.json',receipt)
 assert passed,receipt
 print(json.dumps(receipt),flush=True)

def run(task):
 s,f=task
 if STOP.exists():raise RuntimeError('Stopped after another frame failed')
 start=time.time();cpu=time.process_time()
 try:
  cp=N0/f'cpu/P2_{s:04d}_{f:04d}.npz';ep=N0/f'per_pose/P2_{s:04d}_{f:04d}.npz'
  assert cp.is_file() and ep.is_file(),(s,f,'missing input; do not rebuild')
  a=np.load(cp);e=np.load(ep);ids=a['candidate_id'];angles=a['original_angle'];pred=a['pred'];n=len(ids)
  assert pred.shape==(n,12,17)
  assert np.array_equal(ids,e['candidate_id']) and np.array_equal(angles,e['original_angle']),(s,f,'candidate identity mismatch')
  assert np.array_equal(a['object_id'],e['object_id']),(s,f,'object identity mismatch')
  rows=pred.reshape(-1,17)
  r=evaluate(s,f,rows,np.repeat(a['owner'][ids],12))
  r={k:v.reshape(n,12) for k,v in r.items()}
  checks={k:bool(np.array_equal(r[k],e[k])) for k in ['collision','empty']}
  assert all(checks.values()),(s,f,checks)
  assert all(np.array_equal(r[k],a[k]) for k in ['collision','empty']),(s,f,'cpu input labels mismatch')
  out=ROOT/'evaluation'/f'{s:04d}_{f:04d}.npz';out.parent.mkdir(exist_ok=True)
  assert not out.exists(),(s,f,'refuse output overwrite')
  np.savez_compressed(out,candidate_id=ids,original_angle=angles,object_id=a['object_id'],pred=pred,**r)
  receipt=dict(scene=s,frame=f,input_sha256=E.sha(cp),per_pose_sha256=E.sha(ep),output_sha256=E.sha(out),candidates=n,variants=len(rows),unknown=int(np.isnan(r['mu']).sum()),contact_missing=int(np.isnan(r['I1']).sum()),wall_seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,collision_match=checks['collision'],empty_match=checks['empty'],owners_match=True,candidate_id_match=True,original_angle_match=True,pred_preserved=True,host=os.uname().nodename)
  save_json(out.with_suffix('.json'),receipt)
  print(json.dumps(receipt),flush=True)
  return receipt
 except BaseException:
  STOP.write_text(f'{s:04d}_{f:04d}\n'+traceback.format_exc());raise

def batch(shard,workers):
 assert json.loads((ROOT/'c3_replay.json').read_text())['passed']
 assert not STOP.exists()
 tasks=iter([(s,f) for s in SCENES for f in FRAMES][shard::3])
 with ProcessPoolExecutor(workers) as pool:
  pending={pool.submit(run,t) for t in [next(tasks,None) for _ in range(workers)] if t is not None}
  while pending:
   done,pending=wait(pending,return_when=FIRST_COMPLETED)
   for future in done:future.result()
   assert not STOP.exists()
   for _ in done:
    task=next(tasks,None)
    if task is not None:pending.add(pool.submit(run,task))

if __name__=='__main__':
 if sys.argv[1]=='replay':replay()
 else:batch(int(sys.argv[1]),int(sys.argv[2]))
