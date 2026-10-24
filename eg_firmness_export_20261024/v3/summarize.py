"""Requested export counts only. No alternative-angle choice or benefit analysis."""
from pathlib import Path
import json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tasks(plan):
 return [(split,s,f) for split in ['cal','C3','P2','probe'] for s in (plan['probe_train_scenes']+plan['probe_validation_scenes'] if split=='probe' else plan['cal_scenes'] if split=='cal' else plan['test_scenes']) for f in plan[split+'_frames']]
def main():
 assert not (ROOT/'STOP_ON_FAILURE').exists()
 plan=json.loads((ROOT/'protocol.json').read_text());rep=json.loads((ROOT/'repeatability.json').read_text());assert rep['passed']
 ts=tasks(plan);assert len(ts)==784
 expected={f'{split}_{s:04d}_{f:04d}.npz' for split,s,f in ts}
 assert {p.name for p in (ROOT/'eval').glob('*.npz')}==expected
 assert {p.name for p in (ROOT/'features').glob('*.npz')}==expected
 totals={};manifest=[];gpu=set();forward_seconds=rep['forward_seconds'][1];infer_wall=rep['wall_seconds'][1];cpu_seconds=0.;eval_wall=0.
 first_started=[];last_ended=[]
 for split,s,f in ts:
  key=f'{split}_{s:04d}_{f:04d}';ep=ROOT/f'eval/{key}.npz';fp=ROOT/f'features/{key}.npz';rp=ROOT/f'receipts/eval_{key}.json';ip=ROOT/f'receipts/infer_{key}.json'
  rec=json.loads(rp.read_text());inf=json.loads(ip.read_text())
  assert sha(ep)==rec['eval_sha256'] and sha(fp)==rec['features_sha256']
  assert inf['cache_sha256']==rec['cache_sha256'] and inf['raw_features_sha256']==rec['raw_features_sha256']
  assert all(rec[k] for k in ['nominal_collision_equal','nominal_empty_equal','candidate_id_equal','original_angle_equal','logit_argmax_equal','nominal_pose_preserved'])
  a=np.load(ep);feat=np.load(fp);ids=a['candidate_id'];nom=a['original_angle'];n=len(ids);ii=np.arange(n)
  assert np.array_equal(ids,feat['candidate_id']) and np.array_equal(nom,feat['original_angle']) and np.array_equal(feat['logit'].argmax(1),nom)
  assert a['pred'].shape==(n,12,17) and a['object_id'].shape==(n,12)
  assert np.array_equal(a['object_id'],np.repeat(a['object_id'][:,0,None],12,axis=1))
  for k in ['mu','collision','empty','I1','R1','unknown_reason','contact_reason']:assert a[k].shape==(n,12)
  for k,d in [('vp',256),('ang',64),('logit',12)]:assert feat[k].shape==(n,d) and feat[k].dtype==np.float32 and np.isfinite(feat[k]).all()
  assert np.isfinite(a['R1']).all() and (a['R1']>=0).all()
  assert np.all(a['mu'][a['collision']]==-1)
  clean=~a['collision'][ii,nom]&~a['empty'][ii,nom];mu=a['mu'][ii,nom]
  row=totals.setdefault(split,dict(frames=0,candidates=0,variants=0,original_angle_clean=0,original_angle_feasible=0,original_angle_weak=0,mu_unknown=0,I1_missing=0,collision_or_empty_variants=0))
  row['frames']+=1;row['candidates']+=n;row['variants']+=n*12;row['original_angle_clean']+=int(clean.sum());row['original_angle_feasible']+=int((clean&(mu>0)&(mu<=.4)).sum());row['original_angle_weak']+=int((clean&(mu>.4)).sum())
  row['mu_unknown']+=int(np.isnan(a['mu']).sum());row['I1_missing']+=int(np.isnan(a['I1']).sum());row['collision_or_empty_variants']+=int((a['collision']|a['empty']).sum())
  assert rec['unknown']==int(np.isnan(a['mu']).sum()) and rec['contact_missing']==int(np.isnan(a['I1']).sum())
  forward_seconds+=inf['forward_seconds'];infer_wall+=inf['wall_seconds'];gpu.add(inf['gpu']);cpu_seconds+=rec['cpu_seconds'];eval_wall+=rec['wall_seconds']
  first_started.append(ip.stat().st_mtime-inf['wall_seconds']);last_ended.append(rp.stat().st_mtime)
  for p in [ep,fp,rp,ip]:manifest.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
 for split,row in totals.items():
  assert row['frames']==(640 if split=='probe' else 48)
  row['collision_or_empty_variant_fraction']=row.pop('collision_or_empty_variants')/row['variants']
 summary=dict(splits=totals,two_run_consistency=dict(frame=rep['frame'],**rep['bitwise_equal'],passed=rep['passed']),gpu_models=sorted(gpu),timing_seconds=dict(total_gpu_forward=forward_seconds,total_inference_frame_wall= infer_wall,total_evaluation_cpu=cpu_seconds,total_evaluation_frame_wall=eval_wall,batch_elapsed_from_first_frame_start=max(last_ended)-min(first_started)))
 (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));(ROOT/'output_manifest.json').write_text(json.dumps(manifest,indent=2))
 (ROOT/'integrity_audit.json').write_text(json.dumps(dict(frames=784,all_feature_label_ids_match=True,all_feature_label_original_angles_match=True,all_logit_argmax_match=True,all_nominal_collision_replays_match=True,all_nominal_empty_replays_match=True,all_receipt_hashes_verified=True,all_feature_shapes_and_dtypes_valid=True,all_R1_values_finite_nonnegative=True),indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
