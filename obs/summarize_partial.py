"""Aggregate only recorded export counts and required checks after a mandatory stop."""
from pathlib import Path
import hashlib,json
import numpy as np

R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rep=json.loads((R/'repeatability.json').read_text())
records=[];manifest=[];splits={};removed=[]
for sp in ['cal','C3','P2']:
    rows=[];counts={str(mm):[0,0,0] for mm in [2,3,5]};nc=0
    for p in sorted(R.glob(sp+'_*.json')):
        r=json.loads(p.read_text());q=p.with_suffix('.npz')
        assert sha(q)==r['output_sha256']
        with np.load(q) as a:
            n=len(a['candidate_id']);assert n==r['candidates'];nc+=n
            for mm in [2,3,5]:
                f=a[f'found_r{mm}'];assert f.shape==(n,12)
                counts[str(mm)]=[v+int((f==i).sum()) for i,v in enumerate(counts[str(mm)])]
        assert all(r['checks'][k] for k in ['convention_passed','I1_passed','candidate_id_equal','original_angle_equal'])
        rows.append(r);records.append(r);removed.append(r['outlier_removed_fraction'])
        manifest.extend(dict(path=t.name,bytes=t.stat().st_size,sha256=sha(t)) for t in [p,q])
    nv=nc*12;rr=[r['outlier_removed_fraction'] for r in rows]
    splits[sp]=dict(completed_frames=len(rows),planned_frames=48,candidates=nc,variants=nv,
        outlier_removed_fraction=dict(mean=float(np.mean(rr)),max=max(rr)) if rr else None,
        found_proportions={f'r{mm}':{str(i):v/nv for i,v in enumerate(vals)} for mm,vals in counts.items()} if nv else None)
failures=[json.loads(p.read_text()) for p in sorted(R.glob('failure_*.json'))]
all_records=records+failures;checks=[r['checks'] for r in all_records]
den=sum(c['gt_two_contact_variants'] for c in checks)
summary=dict(status='STOPPED_I1_REPLAY_MISMATCH',complete=False,
    scope='Partial completed exports only; failed frame is evidence, not an exported frame. In-flight tasks completed before workers exited.',
    splits=splits,outlier_removed_fraction=dict(mean=float(np.mean(removed)),max=max(removed)),
    verification=dict(
        convention=dict(scope='completed and failed recorded attempts, excluding duplicate pilot',required_fraction=.99,
            actual_joint_fraction=sum(c['joint_convention_fraction']*c['gt_two_contact_variants'] for c in checks)/den,
            minimum_frame_joint_fraction=min(c['joint_convention_fraction'] for c in checks),GT_two_contact_variants=den,
            all_recorded_attempts_passed=all(c['convention_passed'] for c in checks)),
        I1_replay=dict(all_recorded_attempts_passed=all(c['I1_passed'] for c in checks),all_missing_masks_equal=all(c['I1_missing_mask_equal'] for c in checks),
            max_absolute_error=max(c['I1_max_absolute_error'] for c in checks),required_error_less_than=1e-7,
            scope='Only stored finite I1 recomputed; originally missing entries preserved.'),
        alignment=dict(scope='completed exports',all_candidate_ids_equal=True,all_original_angles_equal=True),repeatability=rep),
    failed_frames=[dict(frame=r['frame'] if isinstance(r['frame'],str) else f"{r['split']}_{r['scene']:04d}_{r['frame']:04d}",checks=r['checks']) for r in failures],
    segmentation_mapping='pixel label = zero-based GraspNet object_id + 1; 0 background; scene IDs+1 checked against meta cls_indexes in completed exports',
    CPU=dict(worker_processes=12,BLAS_OMP_MKL_threads_per_process=1,
        recorded_cpu_seconds=sum(r['cpu_seconds'] for r in all_records)+rep['repeat_cpu_seconds'],
        recorded_frame_wall_seconds_sum=sum(r['wall_seconds'] for r in all_records)+rep['repeat_wall_seconds'],
        timing_scope='Recorded completed frames, failed frame and repeated C3 pilot; excludes setup and orchestration. Wall sum is not elapsed time.'))
(R/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
(R/'output_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summary,indent=2))
