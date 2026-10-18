"""Small reproducible checks for a research plan; does not run an experiment."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
FR = [.2, .4, .6, .8, 1., 1.2]

def utility(mu):
    return sum(0 < mu <= f + 1e-7 for f in FR) / 6

def ce(z, target):
    m = max(z)
    return m + math.log(sum(math.exp(x-m) for x in z)) - z[target]

z = [0., 1., 2., 3., 4., 5.]
shifted = [v+7 for v in z]
raw_score = lambda xs: sum(x*w for x,w in zip(xs,[0,.2,.4,.6,.8,1]))
assert abs(ce(z, 3)-ce(shifted, 3)) < 1e-12
assert abs(raw_score(shifted)-raw_score(z)-21) < 1e-12
assert utility(-1) == 0 and utility(1.2) == 1/6
assert all(utility(FR[i]) > utility(FR[i+1]) for i in range(5))

# Exact fixed-pool AP equals a decreasing weighted sum of per-grasp utilities.
def ap(values):
    return sum(sum(values[:k])/k for k in range(1,51))/50
values = [0., 1/6, .5, 1.]
ranked = sorted(values, reverse=True)
assert ap(ranked) == max(ap(list(p)) for p in itertools.permutations(values))
weights = [sum(1/k for k in range(i,51))/50 for i in range(1,51)]
assert abs(ap(values)-sum(weights[i]*v for i,v in enumerate(values))) < 1e-12

prior_path = ROOT.parent/'eg_sub25_v1_20261015'/'frozen_split.json'
prior = json.loads(prior_path.read_text(encoding='utf-8'))
rng = random.Random(20261018)
reserve = sorted(prior['reserve_scene_ids'])
rng.shuffle(reserve)
split = {'probe_fit': sorted(reserve[:24]), 'probe_calibration': sorted(reserve[24:32]),
         'probe_development': sorted(reserve[32:]), 'official_confirmation': {}}
used = {119,122,188,189}
for name,start,end in [('seen',100,130),('similar',130,160),('novel',160,190)]:
    ids = [s for s in range(start,end) if s not in used]
    rng.shuffle(ids)
    split['official_confirmation'][name] = sorted(ids[:4])
assert len(set(split['probe_fit']+split['probe_calibration']+split['probe_development'])) == 40
assert not (set(prior['train_scene_ids']+prior['dev_scene_ids']) & set(reserve))
assert not (used & set(sum(split['official_confirmation'].values(),[])))
split.update(seed=20261018, selection_uses_model_outputs=False,
    fit_and_calibration_frames=list(range(0,256,32)),
    development_frames=list(range(0,256,8)), confirmation_frames=list(range(256)),
    prior_split_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
    exposure_status='Candidate holdout based on this thread; verify other experiment exposure before execution. Do not redraw based on scores.')
(ROOT/'planned_split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')

timing=json.loads((ROOT.parent/'eg_quality_validation_v1_20261018'/'cost_actual_quality.json').read_text())
rates=[r['ms_per_candidate']/1000 for r in timing['rows']]
low,high=min(rates),max(rates)
count = {'original_stage1_two_models_plus_subset_seen_ablation':800*1024*3,
         'original_stage2_two_models':800*256*6*2,
         'revised_fit_plus_calibration_per_model_max':32*8*256*6,
         'revised_dev_evaluation_uncached_nominal50':8*32*20*50,
         'revised_confirmation_uncached_nominal50_per_model':12*256*20*50}
budget={k:{'candidate_calls':n,'serial_process_hours':[n*low/3600,n*high/3600]} for k,n in count.items()}
features_per_model=32*8*1024*324*4
out={
 'checks':'PASS', 'new_training':False, 'new_remote_geometry_calls':0,
 'official_native_grid':FR,
 'mapping':[{'mu':m,'utility':utility(m),'naive_EG_torch_long_class':int((1.1-m)*5)} for m in FR],
 'invalid_utility':utility(-1),
 'ce_shift_invariant_error':abs(ce(z,3)-ce(shifted,3)),
 'raw_logit_score_shift':raw_score(shifted)-raw_score(z),
 'fixed_pool_utility_ranking_check':True,
 'candidate_timing_seconds':[low,high], 'budgets':budget,
 'float32_feature_bytes_fit_cal_all1024_per_model':features_per_model,
 'mlp_parameters':324*128+128+128+1,
 'limits':'Arithmetic and static checks only. AP estimates assume 50 evaluated candidates; native ties can increase this, so these are not upper bounds. Timings exclude setup, contention, loading and selection; no new feature export, probe training or AP run.'}
(ROOT/'plan_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
manifest=[]
for p in sorted((ROOT/'evidence').glob('*')):
    if p.is_file(): manifest.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'checks':'PASS','split':{k:v for k,v in split.items() if k in ['probe_fit','probe_calibration','probe_development','official_confirmation']},'budget':budget,'mlp_parameters':out['mlp_parameters']},indent=2))
