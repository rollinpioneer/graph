"""Official direct-pose labels; no scoring rule, fitting or angle selection."""
import sys,json,time,os,hashlib,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
ROOT=Path(__file__).resolve().parent;N0=ROOT.parent/'eg_n0_nref_main_20261024'
sys.path.insert(0,str(ROOT.parent/'eg_p2_i1_mu_20261024'))
import evaluate_p2 as P
E=P.E
assert E.sha(Path(P.__file__))=='15d5795c22a45f1149c4b6f38c3dab961888e2e883064fceca2474c44decb684'
STOP=ROOT/'STOP_V2_ON_MISMATCH'
def save_json(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False))
def run(task):
 split,s,f=task;key=f'{split}_{s:04d}_{f:04d}';start=time.time();cpu=time.process_time()
 try:
  assert not STOP.exists()
  ep=None
  if split=='cal':
   cp=N0/f'cpu/train_{s:04d}_{f:04d}.npz';ep=N0/f'per_pose/train_{s:04d}_{f:04d}.npz'
   a=np.load(cp);e=np.load(ep);ids=a['candidate_id'];angles=a['original_angle'];pred=a['pred'];oid=a['object_id'];owners=np.repeat(a['owner'][ids],12)
   assert np.array_equal(ids,e['candidate_id']) and np.array_equal(angles,e['original_angle'])
   assert np.array_equal(oid,e['object_id'])
  else:
   cp=ROOT/f'cache/{key}.npz';a=np.load(cp);raw=a['pred'].astype(float)
   direct=E.Direct(s);direct.frame(f);owner=direct.owners(raw);_,_,ids=E.postprocess(raw,a['cd_mask'],owner)
   angles=a['angle_index'][ids];pred=np.repeat(raw[ids,None,:],12,axis=1)
   for j,k in enumerate(angles):
    for ang in range(12):
     if ang!=k:pred[j,ang,4:13]=(raw[ids[j],4:13].reshape(3,3)@E.rx(np.array([(ang-k)*np.pi/12]))[0]).ravel()
   assert np.array_equal(pred[np.arange(len(ids)),angles],raw[ids])
   owners=np.repeat(owner[ids],12);oid=np.array(direct.ids)[owner[ids]]
   del direct
  n=len(ids);assert pred.shape==(n,12,17)
  # Import and call the previously replayed function, without modifying its source.
  result=P.evaluate(s,f,pred.reshape(-1,17),owners)
  result={k:v.reshape(n,12) for k,v in result.items()}
  rec=dict(split=split,scene=s,frame=f,candidates=n,variants=n*12,input_sha256=E.sha(cp),unknown=int(np.isnan(result['mu']).sum()),contact_missing=int(np.isnan(result['I1']).sum()),owner_mapping_equal=True,evaluator_sha256=E.sha(Path(P.__file__)),official_evaluate_sha256=E.sha(Path(E.__file__)),host=os.uname().nodename)
  if split=='cal':
   checks={k:bool(np.array_equal(result[k],e[k])) for k in ['collision','empty']}
   assert all(checks.values()),(key,checks)
   assert all(np.array_equal(result[k],a[k]) for k in ['collision','empty'])
   rec.update(per_pose_sha256=E.sha(ep),candidate_id_equal=True,original_angle_equal=True,collision_equal=checks['collision'],empty_equal=checks['empty'])
  else:
   rec['collision_or_empty_variants']=int((result['collision']|result['empty']).sum())
   fp=ROOT/f'raw_features/{key}.npz';rawf=np.load(fp)
   infer=json.loads((ROOT/f'receipts/infer_{key}.json').read_text())
   assert infer['cache_sha256']==E.sha(cp) and infer['raw_features_sha256']==E.sha(fp)
   feat=ROOT/f'features/{key}.npz';assert not feat.exists()
   np.savez_compressed(feat,candidate_id=ids,original_angle=angles,**{k:rawf[k][ids] for k in ['vp','ang','logit']})
   rec.update(features_sha256=E.sha(feat),nominal_pose_preserved=True)
  dest=ROOT/f'{split}_eval/{s:04d}_{f:04d}.npz';dest.parent.mkdir(exist_ok=True);assert not dest.exists()
  np.savez_compressed(dest,candidate_id=ids,original_angle=angles,object_id=oid,pred=pred,**result)
  rec.update(output_sha256=E.sha(dest),wall_seconds=time.time()-start,cpu_seconds=time.process_time()-cpu)
  save_json(dest.with_suffix('.json'),rec);print(json.dumps(rec),flush=True)
  return rec
 except BaseException:
  STOP.write_text(key+'\n'+traceback.format_exc());raise
def main():
 assert json.loads((ROOT/'history_v2_passed.json').read_text())['passed']
 plan=json.loads((ROOT/'protocol.json').read_text());shard,workers=map(int,sys.argv[1:3])
 tasks=[('cal',s,f) for s in plan['cal_scenes'] for f in plan['cal_frames']]+[('probe',s,f) for s in plan['probe_train_scenes']+plan['probe_validation_scenes'] for f in plan['probe_frames']]
 todo=tasks[shard::3]
 with ProcessPoolExecutor(workers) as pool:
  pending=set()
  while todo or pending:
   assert not STOP.exists()
   for task in list(todo):
    if len(pending)>=workers:break
    split,s,f=task
    if split=='probe' and not (ROOT/f'receipts/infer_probe_{s:04d}_{f:04d}.json').exists():continue
    pending.add(pool.submit(run,task));todo.remove(task)
   if pending:
    done,pending=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
    for future in done:future.result()
   else:time.sleep(2)
 print('COMPLETE',shard,flush=True)
if __name__=='__main__':main()
