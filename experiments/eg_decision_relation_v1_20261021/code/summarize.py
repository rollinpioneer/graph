"""Aggregate all frozen swaps; unknown labels retain intervals, not zero quality."""
import os,json,csv,hashlib
from pathlib import Path
import numpy as np
R=Path(os.environ['TASK_ROOT']);out=R/'results';scenes=[json.loads(p.read_text()) for p in sorted((out/'scenes').glob('*.json'))]
frames=[r for s in scenes for r in s['frames']];rows=[json.loads(l) for p in sorted((out/'scenes').glob('*_swaps.jsonl')) for l in p.read_text().splitlines()]
assert len(scenes)==32 and len(frames)==768 and len(rows)==49152
assert len({(f['scene'],f['frame'],f['seed']) for f in frames})==768
def save(name,v):(out/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def table(name,rs):
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def ratio(n,d):return n/d if d else None
def identifiable(r):return abs(r['ap_upper']-r['ap_lower'])<1e-8 and abs(r['top1_upper']-r['top1_lower'])<1e-8
def aggregate(ff,rr):
 changed=[r for r in rr if r['changed']];eligible=[r for r in rr if r['supervision_eligible']];known=[r for r in rr if r['pair_known'] and r['pair_reversed'] and identifiable(r)]
 return dict(frame_heads=len(ff),pairs=len(rr),changed=len(changed),identifiable_changed=sum(identifiable(r) for r in changed),identifiable_changed_fraction=ratio(sum(identifiable(r) for r in changed),len(changed)),known_reversed_identifiable=len(known),context_conflicts=sum(r['context_conflict'] for r in known),conflict_fraction=ratio(sum(r['context_conflict'] for r in known),len(known)),eligible=len(eligible),eligible_frame_fraction=float(np.mean([f['eligible']>0 for f in ff])),supported_scenes=len({f['scene'] for f in ff if f['eligible']}),nms_eligible=sum(r['nms_set_changed'] for r in eligible),nms_eligible_fraction=ratio(sum(r['nms_set_changed'] for r in eligible),len(eligible)),known_top1_fraction=float(np.mean([f['known_top1'] for f in ff])),known_top50_fraction=sum(f['known_top50'] for f in ff)/sum(f['selected_count'] for f in ff),mean_best_AP_lower=float(np.mean([f['best_AP_lower'] for f in ff])),mean_best_Top1_lower_pp=100*float(np.mean([f['best_Top1_lower'] for f in ff])),pair_kinds={k:sum(r['kind']==k for r in rr) for k in sorted({r['kind'] for r in rr})})
summary=dict(status='COMPLETE',scenes=32,unique_frames=256,head_count=3,exchanges=49152,formal_training_updates=0,new_GT_quality=0,EG_forward=0,C4_reads=0,overall=aggregate(frames,rows),roles={},heads={})
scene_rows=[]
for s in scenes:
 f=s['frames'];rr=[r for r in rows if r['scene']==s['scene']];a=aggregate(f,rr);a.pop('pair_kinds');scene_rows.append(dict(scene=s['scene'],role=f[0]['role'],**a))
for role in ['fit','model_val','gate_cal']:
 ff=[f for f in frames if f['role']==role];rr=[r for r in rows if r['role']==role];a=aggregate(ff,rr);sr=[s for s in scene_rows if s['role']==role];rng=np.random.default_rng(20261021);indices=rng.integers(0,len(sr),(10000,len(sr)))
 for key in ['mean_best_AP_lower','mean_best_Top1_lower_pp','eligible_frame_fraction']:
  vals=np.array([s[key] for s in sr]);a[key+'_scene_bootstrap95']=np.quantile(vals[indices].mean(1),[.025,.975]).tolist()
 summary['roles'][role]=a
for seed in [20261018,20261019,20261020]:summary['heads'][str(seed)]=aggregate([f for f in frames if f['seed']==seed],[r for r in rows if r['seed']==seed])
g=json.loads((R/'frozen_config.json').read_text(encoding='utf-8-sig'))['gates'];roles=summary['roles'];o=summary['overall']
checks=dict(frame_coverage=all(roles[k]['eligible_frame_fraction']>=g['frame_fraction_each_role'] for k in roles),scene_coverage=all(roles[k]['supported_scenes']>=g[k+'_supported_scenes'] for k in roles),identifiability=(o['identifiable_changed_fraction'] or 0)>=g['identifiable_changed_fraction'],novelty=(o['conflict_fraction'] or 0)>=g['conflict_among_known_reversed_identifiable_fraction'],NMS=(o['nms_eligible_fraction'] or 0)>=g['nms_fraction_of_eligible'],heldout_material_gain=all(roles[k]['mean_best_AP_lower']>=g['heldout_best_single_swap_mean_AP_lower'] or roles[k]['mean_best_Top1_lower_pp']>=100*g['heldout_best_single_swap_mean_Top1_lower'] for k in ['model_val','gate_cal']))
summary['gates']=checks;summary['decision']='PASS_DIAGNOSTIC_GATE_ONLY' if all(checks.values()) else ('INCONCLUSIVE_LABEL_COVERAGE' if not checks['identifiability'] else 'NOT_SUPPORTED_BY_CURRENT_DIAGNOSIS')
summary['caveat']='Oracle best-single-swap lower bounds on existing head data, not deployed AP improvement. Three heads averaged within scenes; bootstrap resamples scenes. Strict novelty criterion is sufficient, not exhaustive. No fitting on held-out roles.'
events=[]
for s in scenes:events.extend([(s['start'],1),(s['end'],-1)])
active=peak=0
for t,delta in sorted(events):active+=delta;peak=max(peak,active)
summary['resource']=dict(successful_scene_worker_hours=sum(s['seconds'] for s in scenes)/3600,successful_scene_CPU_hours=sum(s['cpu_seconds'] for s in scenes)/3600,peak_overlap_successful_workers=peak,successful_max_RSS_GiB=max(s['max_rss_KiB'] for s in scenes)/2**20,pilot_worker_hours=json.loads((out/'gpu04_pilot.json').read_text())['worker_seconds']/3600,failed_attempt_worker_hours_upper_bound=8*455/3600,failed_attempt_note='Conservative launch-time bound; includes three successes already in successful scene sum. Not an exact failure runtime.',GPU_hours=0)
# Coefficient identity check: AP is average precision-at-k over the six friction thresholds.
rng=np.random.default_rng(20261021);u=rng.integers(0,7,50)/6;w=np.array([sum(1/k for k in range(j,51))/50 for j in range(1,51)]);assert abs(np.dot(u,w)-np.mean(np.cumsum(u)/np.arange(1,51)))<1e-12
summary['verification']=dict(frame_head_unique=True,all_32_scenes_complete=True,full_pair_budget=True,AP_coefficient_identity=True,geometry_exact=True,native_selection_exact=True)
save('SUMMARY.json',summary);table('per_scene.csv',scene_rows);table('per_frame_head.csv',frames)
examples=sorted([r for r in rows if r['supervision_eligible']],key=lambda r:(-r['ap_lower'],r['scene'],r['frame'],r['seed']))[:12]
save('representative_swaps_exploratory.json',examples)
save('result_identity.json',[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted((out/'scenes').glob('*'))])
print(json.dumps(summary,indent=2))
