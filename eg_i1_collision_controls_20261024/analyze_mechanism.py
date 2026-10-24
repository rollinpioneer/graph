import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
files=[json.loads(p.read_text()) for p in sorted((ROOT/'mechanism').glob('*.json'))]
assert len(files)==48
a=pd.DataFrame([r for p in files for r in p['rows']]);d=a[a.scope=='D_residual_collision'];f=a[a.scope=='feasible_false_filter']
assert len(d)==121 and len(f)==164
missing=d.observed_points_in_GT_collision_solid==0
counted=d.observed_GT_points_also_counted_by_EG>0
different=~missing&~counted
rows=dict(D_residual=dict(n=len(d),scenes=int(d.scene.nunique()),observed_true_static_collision_points_counted_but_below_threshold=int(counted.sum()),
    observed_static_collision_points_not_in_EG_volume=int(different.sum()),no_observed_points_in_GT_static_collision_solid=int(missing.sum()),
    assigned_object_collision=int((d.gt_assigned_object>0).sum()),other_object_collision=int((d.gt_other_objects>0).sum()),table_collision=int((d.gt_table>0).sum()),
    only_other_objects_or_table=int(((d.gt_assigned_object==0)&((d.gt_other_objects+d.gt_table)>0)).sum()),
    left_finger=int((d.gt_left>0).sum()),right_finger=int((d.gt_right>0).sum()),bottom=int((d.gt_bottom>0).sum()),
    RC_RS_both_zero=int(((d.RC==0)&(d.RS==0)).sum()),nearest_obs_min_m_median=float(d.gt_collision_nearest_observation_min_m.median()),
    nearest_obs_min_over5mm=int((d.gt_collision_nearest_observation_min_m>.005).sum())),
  feasible_false_filter=dict(n=len(f),scenes=int(f.scene.nunique()),full_GT_has_EG_volume_occupancy=int((f.full_GT_points_in_EG_union>0).sum()),
    full_GT_has_EG_static_occupancy=int((f.full_GT_points_in_EG_static>0).sum()),full_GT_sweep_only=int(((f.full_GT_points_in_EG_static==0)&(f.full_GT_points_in_EG_sweep>0)).sum()),
    no_full_GT_EG_occupancy=int((f.full_GT_points_in_EG_union==0).sum())))
a['mechanism']='not_applicable'
a.loc[d.index[counted],'mechanism']='observed_true_collision_counted_but_subthreshold'
a.loc[d.index[different],'mechanism']='observed_true_collision_outside_EG_volume'
a.loc[d.index[missing],'mechanism']='no_observed_points_in_official_static_solid'
a.to_csv(ROOT/'mechanism_rows.csv',index=False)
(ROOT/'mechanism_summary.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
for kind,z in a[a.scope=='D_residual_collision'].groupby('mechanism'):
 print(kind,z[['scene','frame','candidate_id','angle_class','gt_assigned_object','gt_other_objects','gt_table','observed_points_in_GT_collision_solid','observed_GT_points_also_counted_by_EG','EG_global_iou','RC','RS']].head(2).to_string(index=False))
