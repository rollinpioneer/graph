"""Descriptive decomposition of the already checked paired table."""
import collections
import csv
import json
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1])
rows = list(csv.DictReader((root/'results/all_pairs.csv').open()))
rows = [v for v in rows if v['use_cd'] == 'True']
harm = [v for v in rows if v['transition'] == 'success_to_failure']
rescue = [v for v in rows if v['transition'] == 'failure_to_success']


def describe(vs, field):
    a = np.array([float(v[field]) for v in vs])
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                q25=float(np.quantile(a,.25)), q75=float(np.quantile(a,.75)))


def counts(vs, key):
    return dict(collections.Counter(v[key] for v in vs))


matrix = collections.Counter((v['old_category'],v['new_category']) for v in rows)
frames = collections.defaultdict(list)
for v in harm:
    frames[v['scene'],v['frame']].append(v)
all_three = [vs for vs in frames.values() if len(vs)==3]
scene_net = []
for s in sorted({int(v['scene']) for v in rows}):
    h=[v for v in harm if int(v['scene'])==s]
    r=[v for v in rescue if int(v['scene'])==s]
    scene_net.append(dict(scene=s,damage=len(h),rescue=len(r),net=len(r)-len(h),
                          new_failure_categories=counts(h,'new_category')))
details = dict(pooled_events_are_repeated_heads_not_independent_samples=True,
    harm=counts(harm,'new_category'),rescue=counts(rescue,'old_category'),
    harm_new_mu=counts(harm,'new_mu'),harm_old_mu=counts(harm,'old_mu'),
    damaged_pair_statistics={k:describe(harm,k) for k in ['old_Q','new_Q','old_EG_score','new_EG_score',
        'old_utility','new_utility','half_zEG_gap','half_zQ_gap','old_raw_F05_rank']},
    all_three_damaged_frames=len(all_three),
    all_three_same_new_candidate=sum(len({v['new_id'] for v in vs})==1 for vs in all_three),
    all_three_same_failure_category=sum(len({v['new_category'] for v in vs})==1 for vs in all_three),
    old_still_F05_selected=sum(v['old_in_F05_selected']=='True' for v in harm),
    frame_ids_by_scene={s:sorted({int(v['frame']) for v in harm if v['scene']==s}) for s in sorted({v['scene'] for v in harm})},
    scene_net=sorted(scene_net,key=lambda v:v['net']),
    collision_to_still_failed_noncollision=sum(n for (old,new),n in matrix.items()
        if old=='nonempty_collision' and new in ['weak_positive_quality','no_force_closure_in_evaluated_range']),
    weak_quality_to_nonempty_collision=matrix['weak_positive_quality','nonempty_collision'])
(root/'results/details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in details.items() if k!='frame_ids_by_scene'},indent=2))
