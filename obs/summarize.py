"""Only the predeclared geometry export counts and verification results."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent;V3=ROOT.parent.parent/'eg_firmness_export_20261024/v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not (ROOT/'STOP_ON_FAILURE').exists()
 plan=json.loads((ROOT/'protocol.json').read_text());rep=json.loads((ROOT/'repeatability.json').read_text());assert rep['passed']
 tasks=[(sp,s,f) for sp in ['cal','C3','P2'] for s in plan['cal_scenes' if sp=='cal' else 'test_scenes'] for f in plan[sp+'_frames']]
 assert len(tasks)==144
 counts={};removed=[];checks=[];manifest=[];cpu=rep['repeat_cpu_seconds'];framewall=rep['repeat_wall_seconds'];starts=[];ends=[]
 for sp,s,f in tasks:
  key=f'{sp}_{s:04d}_{f:04d}';path=ROOT/(key+'.npz');rp=path.with_suffix('.json');r=json.loads(rp.read_text());assert sha(path)==r['output_sha256']
  a=np.load(path);ref=np.load(V3/f'eval/{key}.npz');n=len(a['candidate_id']);N=n*12
  assert np.array_equal(a['candidate_id'],ref['candidate_id']) and np.array_equal(a['original_angle'],ref['original_angle'])
  assert np.array_equal(np.isfinite(a['gt_pts_cam']).all(axis=(2,3)),np.isfinite(ref['I1']))
  row=counts.setdefault(sp,dict(frames=0,candidates=0,variants=0,found_counts={str(v):[0,0,0] for v in [2,3,5]},removed=[]));row['frames']+=1;row['candidates']+=n;row['variants']+=N
  for mm in [2,3,5]:
   found=a[f'found_r{mm}'];risk=a[f'I1obs_r{mm}'];assert found.shape==(n,12) and found.dtype==np.int8 and np.isin(found,[0,1,2]).all()
   assert np.array_equal(np.isfinite(risk),found==2)
   row['found_counts'][str(mm)]=[x+int((found==i).sum()) for i,x in enumerate(row['found_counts'][str(mm)])]
  assert a['I1half_r3'].shape==(n,12) and np.array_equal(np.isfinite(a['I1half_r3']),a['found_r3']>0)
  assert a['I1obs_seg_r3'].shape==(n,12) and np.array_equal(np.isfinite(a['I1obs_seg_r3']),a['found_seg_r3']==2)
  assert a['obs_pts_r3'].shape==(n,12,2,3) and a['obs_normals_r3'].shape==(n,12,2,3)
  assert np.array_equal(np.isfinite(a['obs_pts_r3']).all(-1).sum(-1),a['found_r3'])
  assert np.array_equal(np.isfinite(a['obs_pts_r3']),np.isfinite(a['obs_normals_r3']))
  c=r['checks'];assert c['convention_passed'] and c['I1_passed'] and c['candidate_id_equal'] and c['original_angle_equal'];checks.append(c)
  removed.append(r['outlier_removed_fraction']);row['removed'].append(r['outlier_removed_fraction']);cpu+=r['cpu_seconds'];framewall+=r['wall_seconds'];starts.append(rp.stat().st_mtime-r['wall_seconds']);ends.append(rp.stat().st_mtime)
  manifest.extend([dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in [path,rp]])
 for sp,row in counts.items():
  assert row['frames']==48
  row['outlier_removed_fraction']=dict(mean=float(np.mean(row['removed'])),max=float(np.max(row['removed'])));del row['removed']
  row['found_proportions']={f'r{mm}':{str(i):v/row['variants'] for i,v in enumerate(values)} for mm,values in row.pop('found_counts').items()}
 denominator=sum(c['gt_two_contact_variants'] for c in checks)
 overall=sum(round(c['joint_convention_fraction']*c['gt_two_contact_variants']) for c in checks)/denominator
 summary=dict(splits=counts,outlier_removed_fraction=dict(mean=float(np.mean(removed)),max=float(np.max(removed))),verification=dict(convention=dict(all_frames_passed=True,required_fraction=.99,actual_joint_fraction=overall,minimum_frame_joint_fraction=min(c['joint_convention_fraction'] for c in checks),GT_two_contact_variants=denominator),I1_replay=dict(all_missing_masks_equal=True,max_absolute_error=max(c['I1_max_absolute_error'] for c in checks),required_error_less_than=1e-7,scope='recompute stored finite I1; preserve originally missing entries'),alignment=dict(all_candidate_ids_equal=True,all_original_angles_equal=True),repeatability=rep),segmentation_mapping='segmentation pixel value = zero-based GraspNet object_id + 1; background = 0; scene IDs+1 checked against meta cls_indexes',CPU=dict(worker_processes=12,threads_per_process=1,total_cpu_seconds=cpu,total_frame_wall_seconds=framewall,elapsed_from_first_frame_seconds=max(ends)-min(starts)))
 (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));(ROOT/'output_manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
