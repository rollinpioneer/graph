"""V3 CPU labels and full-depth EG collision ratio, without angle selection."""
import sys,json,time,os,traceback,importlib.util,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
from scipy.io import loadmat
from PIL import Image
ROOT=Path(__file__).resolve().parent;B=ROOT.parent.parent;STOP=ROOT/'STOP_ON_FAILURE'
sys.path.insert(0,str(B/'eg_p2_i1_mu_20261024'));import evaluate_p2 as P
E=P.E
assert E.sha(Path(P.__file__))=='15d5795c22a45f1149c4b6f38c3dab961888e2e883064fceca2474c44decb684'
cdpath=B/'eg_full_repro_v1_20261015/runtime/EconomicGrasp/utils/collision_detector.py'
spec=importlib.util.spec_from_file_location('v3_cd',cdpath);CD=importlib.util.module_from_spec(spec);spec.loader.exec_module(CD)
def write_json(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False))
def tasks(plan):
 return [(split,s,f) for split in ['cal','C3','P2','probe'] for s in (plan['probe_train_scenes']+plan['probe_validation_scenes'] if split=='probe' else plan['cal_scenes'] if split=='cal' else plan['test_scenes']) for f in plan[split+'_frames']]
def full_depth_R1(s,f,rows):
 p=E.D/f'scenes/scene_{s:04d}/kinect';m=loadmat(p/f'meta/{f:04d}.mat');K=m['intrinsic_matrix'];raw=np.asarray(Image.open(p/f'depth/{f:04d}.png'))
 depth=raw.astype(np.float32)/float(m['factor_depth'].item());yy,xx=np.indices(raw.shape)
 cloud=np.stack([(xx-K[0,2])*depth/K[0,0],(yy-K[1,2])*depth/K[1,1],depth],-1)
 detector=CD.ModelFreeCollisionDetector(cloud[depth>0],voxel_size=.005);ratios=[]
 for k in range(0,len(rows),24):
  c,p=detector.detect(E.GraspGroup(rows[k:k+24]),approach_dist=.05,collision_thresh=.01,return_ious=True)
  ratios.append(np.stack(p,1)[:,0])
 return np.concatenate(ratios) if ratios else np.empty(0)
def run(task):
 split,s,f=task;key=f'{split}_{s:04d}_{f:04d}';start=time.time();cpu=time.process_time()
 try:
  assert not STOP.exists()
  cp=ROOT/f'cache/{key}.npz';fp=ROOT/f'raw_features/{key}.npz';inf=json.loads((ROOT/f'receipts/infer_{key}.json').read_text())
  assert E.sha(cp)==inf['cache_sha256'] and E.sha(fp)==inf['raw_features_sha256']
  a=np.load(cp);raw=a['pred'].astype(float);direct=E.Direct(s);direct.frame(f);owner=direct.owners(raw)
  _,_,ids=E.postprocess(raw,a['cd_mask'],owner);nom=a['angle_index'][ids];n=len(ids)
  pred=np.repeat(raw[ids,None,:],12,axis=1)
  for j,k in enumerate(nom):
   for ang in range(12):
    if ang!=k:pred[j,ang,4:13]=(raw[ids[j],4:13].reshape(3,3)@E.rx(np.array([(ang-k)*np.pi/12]))[0]).ravel()
  assert np.ascontiguousarray(pred[np.arange(n),nom]).tobytes()==np.ascontiguousarray(raw[ids]).tobytes()
  oid=np.array(direct.ids)[owner[ids]];del direct
  result=P.evaluate(s,f,pred.reshape(-1,17),np.repeat(owner[ids],12));result={k:v.reshape(n,12) for k,v in result.items()}
  nominal=P.evaluate(s,f,raw[ids],owner[ids])
  checks={k:bool(np.array_equal(result[k][np.arange(n),nom],nominal[k])) for k in ['collision','empty']}
  assert all(checks.values()),(key,'nominal replay',checks)
  r1_start=time.perf_counter();r1=full_depth_R1(s,f,pred.reshape(-1,17)).reshape(n,12);r1_seconds=time.perf_counter()-r1_start
  assert np.isfinite(r1).all() and (r1>=0).all()
  rf=np.load(fp);features={k:rf[k][ids] for k in ['vp','ang','logit']}
  assert np.array_equal(features['logit'].argmax(1),nom)
  for k,d in [('vp',256),('ang',64),('logit',12)]:assert features[k].shape==(n,d) and features[k].dtype==np.float32
  ep=ROOT/f'eval/{key}.npz';feat=ROOT/f'features/{key}.npz'
  ep.parent.mkdir(exist_ok=True);feat.parent.mkdir(exist_ok=True);assert not ep.exists() and not feat.exists()
  np.savez_compressed(ep,candidate_id=ids,original_angle=nom,object_id=np.repeat(oid[:,None],12,axis=1),pred=pred,R1=r1,**result)
  np.savez_compressed(feat,candidate_id=ids,original_angle=nom,**features)
  with np.load(ep) as ev,np.load(feat) as fe:
   assert np.array_equal(ev['candidate_id'],fe['candidate_id']) and np.array_equal(ev['original_angle'],fe['original_angle'])
  rec=dict(split=split,scene=s,frame=f,candidates=n,variants=n*12,cache_sha256=E.sha(cp),raw_features_sha256=E.sha(fp),eval_sha256=E.sha(ep),features_sha256=E.sha(feat),nominal_collision_equal=checks['collision'],nominal_empty_equal=checks['empty'],candidate_id_equal=True,original_angle_equal=True,logit_argmax_equal=True,nominal_pose_preserved=True,unknown=int(np.isnan(result['mu']).sum()),contact_missing=int(np.isnan(result['I1']).sum()),wall_seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,R1_seconds=r1_seconds,evaluator_sha256=E.sha(Path(P.__file__)),official_evaluate_sha256=E.sha(Path(E.__file__)),collision_detector_sha256=E.sha(cdpath),R1_input='full depth>0 cloud without segmentation',host=os.uname().nodename)
  write_json(ROOT/f'receipts/eval_{key}.json',rec);print('FRAME '+key,flush=True);return rec
 except BaseException:STOP.write_text(key+'\n'+traceback.format_exc());raise
def main():
 assert json.loads((ROOT/'repeatability.json').read_text())['passed']
 plan=json.loads((ROOT/'protocol.json').read_text())
 if sys.argv[1]=='recover':
  shard='recover';workers=int(sys.argv[2]);todo=[t for i,t in enumerate(tasks(plan)) if i%3 in (1,2)]
 else:shard,workers=map(int,sys.argv[1:3]);todo=tasks(plan)[shard::3]
 todo=[t for t in todo if not (ROOT/f'receipts/eval_{t[0]}_{t[1]:04d}_{t[2]:04d}.json').exists()]
 with ProcessPoolExecutor(workers) as pool:
  pending=set()
  while todo or pending:
   assert not STOP.exists()
   for task in list(todo):
    if len(pending)>=workers:break
    split,s,f=task
    if not (ROOT/f'receipts/infer_{split}_{s:04d}_{f:04d}.json').exists():continue
    pending.add(pool.submit(run,task));todo.remove(task)
   if pending:
    done,pending=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
    for future in done:future.result()
   else:time.sleep(2)
 print('COMPLETE',shard,flush=True)
if __name__=='__main__':main()
