"""Interpret official flags without changing evaluation or primary outcomes."""
from common import *
import pandas as pd
import inspect
import graspnetAPI.utils.eval_utils as EU

source=inspect.getsource(EU.collision_detection)
assert 'collision_mask = (collision_mask | empty_mask)' in source
semantics={'official_collision_flag':'geometric_intersection OR empty','empty_cases_may_also_intersect':True,'nonempty_collision':'official_collision AND NOT empty; unambiguously nonempty geometric collision, not all possible geometric intersections','noncollision_quality_failure':'official quality<=0, no collision flag and no empty flag; could be failed force closure or invalid grasp, not further distinguished','legacy_known_true_collisions_columns':'Earlier diagnosis counts use the official combined rejection flag; do not interpret as pure geometric intersection','evaluation_changed':False,'source_path':inspect.getsourcefile(EU.collision_detection),'source_sha256':hashlib.sha256(source.encode()).hexdigest()}
write('results/collision_semantics.json',semantics)
scenes=sum([CFG['exploration'][s] for s in SPLITS],[])
ctx=pd.concat([pd.read_csv(R/f'diagnosis/{s:04d}/candidate_context.csv') for s in scenes],ignore_index=True)
ctx['category']=np.select([ctx['empty']==1,ctx['collision']==1,ctx['quality']<=0,ctx['quality']>.4000001],['empty','nonempty_collision','noncollision_quality_failure','weak_mu_above_0.4'],default='qualified_mu_at_most_0.4')
counts=ctx.groupby(['scene','seed','use_cd','model','category']).size().rename('candidate_observations').reset_index();counts.to_csv(R/'results/diagnosis_failure_categories.csv',index=False)
rows=[]
for (model,width),g in ctx[(ctx.scene==136)&ctx.use_cd].groupby(['model','width_bin']):
 vals=g.category.value_counts();rows.append(dict(model=model,width_bin=int(width),N=len(g),**{k:int(vals.get(k,0)) for k in ['empty','nonempty_collision','noncollision_quality_failure','weak_mu_above_0.4','qualified_mu_at_most_0.4']}))
csv_write('results/scene0136_width_failure_categories.csv',rows)
comparisons=[]
for phase,groups in [('exploration',CFG['exploration']),('confirmation',json.loads((R/'C2_method_lock.json').read_text())['selected'] if (R/'C2_method_lock.json').exists() else None)]:
 path=R/f'results/{phase}_per_frame.csv'
 if not path.exists():continue
 df=pd.read_csv(path);assert all(df[c].dtype==bool for c in ['top1_collision','top1_empty']);assert not (df.top1_empty&~df.top1_collision).any()
 df['top1_nonempty_collision']=df.top1_collision&~df.top1_empty
 agg=df.groupby(['split','scene','configuration'])[['top1_collision','top1_empty','top1_nonempty_collision']].mean().reset_index();agg.to_csv(R/f'results/{phase}_collision_disaggregated_per_scene.csv',index=False)
 for family in (['S','F05'] if phase=='exploration' else ['F05']):
  for cd in [False,True]:
   suffix='_CD' if cd else '';est={}
   for metric in ['top1_collision','top1_empty','top1_nonempty_collision']:
    split_values=[]
    for split in SPLITS:
     by_scene=[]
     for scene in groups[split]:
      ref=agg[(agg.scene==scene)&(agg.configuration=='R0'+suffix)][metric].iloc[0];vals=[agg[(agg.scene==scene)&(agg.configuration==f'{family}_{seed}{suffix}')][metric].iloc[0]-ref for seed in SEEDS];by_scene.append(np.mean(vals))
     split_values.append(np.mean(by_scene))
    est[metric]=float(np.mean(split_values)*100)
   assert np.isclose(est['top1_collision'],est['top1_empty']+est['top1_nonempty_collision'])
   comparisons.append(dict(phase=phase,family=family,reference='R0',use_cd=cd,delta_percentage_points=est,scope='Descriptive disaggregation, not an additional confirmation endpoint'))
write('results/collision_disaggregated_comparisons.json',comparisons)
print('COLLISION_FLAG_DISAGGREGATED',json.dumps(comparisons),flush=True)
