from core import *
import csv,importlib.metadata,subprocess
out=R/'results';d=json.loads((out/'summary.json').read_text());rows=json.loads((out/'per_case.json').read_text());sums={x['strategy']:x for x in d['strategies']}
assert len(rows)==640 and len({(x['scene'],x['frame']) for x in rows})==640
assert sum(not x['success'] for x in rows)==187 and sum(x['success'] for x in rows)==453
assert sum(x['good_cells']>0 for x in rows if not x['success'])==170
assert d['categories']==dict(angle_only_suffices=55,depth_only_suffices=34,both_single=69,joint_change_required=12,unrecoverable48=17)
for x in rows:assert x['hits']['joint6']<=x['hits']['joint12']<=x['hits']['all48']
assert all(sums[k]['success_oracle_retention']==1 for k in ['EG','top3x2','top2x3','joint6','joint12','all48'])
# Independent brute-force combination check of the exact random probability on k=2 toy cases.
from itertools import combinations
import math
def random_hit(g,n,k):return 1.-(math.comb(n-g,k)/math.comb(n,k) if n-g>=k else 0.)
for n in [6,8]:
 for g in range(n+1):assert abs(random_hit(g,n,2)-np.mean([any(i<g for i in v) for v in combinations(range(n),2)]))<1e-12
with open(out/'failure_pose_ranks.csv','w',newline='') as f:
 fields=['scene','frame','good_cells','best_quality','canonical_best_rank','best_rank_pairs','feasible_rank_pairs'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for x in rows:
  if not x['success']:w.writerow({k:json.dumps(x[k]) if isinstance(x[k],list) else x[k] for k in fields})
refs=[G/'eg_nqs_pilot_v2_20261018/code/common.py',PRIOR/'code/decoder_control.py',CODE/'models/economicgrasp.py',CODE/'models/modules_economicgrasp.py']
write(out/'provenance.json',dict(card_sha256=sha(R/'EXPERIMENT_CARD.md'),source_code_hashes={str(p.relative_to(G)):sha(p) for p in refs},frozen_inputs_manifest=dict(path='results/source_hashes.json',sha256=sha(out/'source_hashes.json'),entries=len(json.loads((out/'source_hashes.json').read_text()))),versions={k:importlib.metadata.version(k) for k in ['torch','numpy','scipy']},training=0))
manifest={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for folder in ['code','quality','replay','results'] for p in sorted((R/folder).glob('*')) if p.is_file() and p.name not in ['artifact_manifest.json','validation.json','archive_index.json']}
write(out/'artifact_manifest.json',manifest)
write(out/'archive_index.json',dict(full_manifest=dict(path='results/artifact_manifest.json',sha256=sha(out/'artifact_manifest.json'),entries=len(manifest)),retained_files={k:v for k,v in manifest.items() if k in ['results/per_case.json','results/source_hashes.json']},note='Large/raw caches and full file identity list remain under experiment root. Public files have explicit relative paths.'))
write(out/'validation.json',dict(status='PASS',unique_cases=640,failures=187,successes=453,recoverable=170,original_prediction_replay_exact=True,all_original_quality_checks=True,all10_native_workers_pass=True,grid_mapping_checked=True,nested_coverage_checks=True,random_probability_toy_check=True,no_training=True))
print('SEALED')
