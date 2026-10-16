from pathlib import Path
import json,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'reports/decomposition.json').read_text())
assert len(rows)==96
v=json.loads((ROOT/'receipts/validation.json').read_text());assert v['passed']
assert json.loads((ROOT/'receipts/exit.json').read_text())['exit_code']==0

def combine(values):
 if isinstance(values[0],dict):return {k:combine([x[k] for x in values]) for k in values[0]}
 return float(np.mean(values))
keys=['reference_target_drift','accepted_increment_drift','actual_target_drift','main_and_interactions','shapley','clipped_fraction_before','clipped_fraction_after']
groups=collections.defaultdict(list)
for x in rows:groups[(x['seed'],x['arm'],x['old_offset'],x['new_offset'],x['region'])].append(x)
summary=[]
for (s,a,o,n,g),rr in groups.items():
 assert len(rr)==2
 x=dict(seed=s,arm=a,old_offset=o,new_offset=n,region=g,**{k:combine([r[k] for r in rr]) for k in keys})
 x['largest_main_effect_by_mean_abs_each_repeat']=[max(('A','V','C'),key=lambda k:r['main_and_interactions'][k]['mean_abs']) for r in rr]
 x['largest_shapley_by_mean_abs_each_repeat']=[max(('A','V','C'),key=lambda k:r['shapley'][k]['mean_abs']) for r in rr]
 x['sum_interaction_mean_abs']=sum(x['main_and_interactions'][k]['mean_abs'] for k in ('AV','AC','VC','AVC'))
 summary.append(x)
(ROOT/'reports/compact_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

def table(headers,values):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in values])
def f(v):return f'{v:.4f}'
lines=['# TD Target Drift Decomposition: fixed-checkpoint evidence','',
'Analysis of seeds 0/3, arms R2/R3/R4/R5, existing fork/+50k/+100k checkpoints. No training, no optimizer updates, no environment construction/reset/step. All numbers average the two pre-fixed noise schedules; each schedule is separately retained in decomposition.json.','',
'## Sampling and interface','',
'2048 original processed training transitions from current-state start cells and 2048 outside, sampled without replacement with the same indices for all models. Outside is a comparison region, not a verified normal region or a posture-matched control. Target action is generated at next state; VaBC at current state and scored at next state, preserving target_use_next=false. Discount .995, mean ensemble and mean actor/flow aggregation preserved. All component swaps use the same state and random input. No trajectory-cluster significance claim.','',
'## Reference drift versus accepted-increment drift: fork to +100k','',
'MAE below means mean absolute CHANGE on the fixed inputs, not a prediction error against true value. Positive signed change means target increases (less negative in this reward convention). These changes span checkpoints and are not per-update jumps.','',
 table(['seed','arm','region','reference signed drift','reference mean abs drift','accepted mean abs drift','total mean abs drift'],[[x['seed'],x['arm'],x['region'],f(x['reference_target_drift']['mean']),f(x['reference_target_drift']['mean_abs']),f(x['accepted_increment_drift']['mean_abs']),f(x['actual_target_drift']['mean_abs'])] for x in summary if x['old_offset']==0 and x['new_offset']==100000]),'',
'## Old-anchored factorial signed decomposition: fork to +100k','',
'A=replace reference Actor only; V=replace VaBC only; C=replace Target Critic only. Main effects hold other factors at the earlier checkpoint. AC/VC are interactions; AV and AVC should vanish under the additive target formula. Values sum exactly to total reference drift per state; averaging preserves this identity. Main effects depend on the chosen anchor. Shapley values distribute interactions symmetrically as a secondary accounting convention, not a causal explanation.','',
 table(['seed','arm','region','A','V','C','AC','VC','interaction mean abs sum'],[[x['seed'],x['arm'],x['region'],*[f(x['main_and_interactions'][k]['mean']) for k in ('A','V','C','AC','VC')],f(x['sum_interaction_mean_abs'])] for x in summary if x['old_offset']==0 and x['new_offset']==100000]),'',
'## Slow EMA R3 across intervals','',
 table(['seed','region','interval','reference signed drift','reference mean abs drift','A signed','V signed','C signed','AC signed','VC signed'],[[x['seed'],x['region'],str(x['old_offset'])+'->'+str(x['new_offset']),f(x['reference_target_drift']['mean']),f(x['reference_target_drift']['mean_abs']),*[f(x['main_and_interactions'][k]['mean']) for k in ('A','V','C','AC','VC')]] for x in summary if x['arm']=='R3']), '',
'## Necessary validation','',json.dumps(v,indent=2),'',
'## Interpretation limits','',
'Existing seed-3 BFC anomalies occurred around +30k; no +30k checkpoint is available. At +50k the system is already after that event. Endpoints cannot reconstruct an unrecorded transient peak, establish temporal precedence, or prove that the largest changing component was wrong. Seed-0 and seed-3 forks differ in training age (600k vs 300k) as well as parameters. A changing Critic may reflect policy/bootstrapping feedback or legitimate learning; no true long-horizon action values are measured here. Small direct policy effects under a frozen Critic do not rule out policy-driven effects mediated by Critic training.','',
'For the two BFC arms, accepted increment is always bounded by epsilon at an endpoint, so its between-endpoint change is at most 2 epsilon. Reference drift is unconstrained by this formula. A large reference/accepted ratio alone is partly structural; localization and the component-swapping results provide the additional evidence.','',
'Interaction magnitudes, per-state quantiles, signed/absolute/RMS statistics and both noise repeats are retained in decomposition.json and compact_summary.json. Raw Q cross-products and exact fixed inputs remain on the server under arrays/. No model/data/checkpoint was overwritten.','']
(ROOT/'reports/MEASUREMENTS.md').write_text('\n'.join(lines))
for x in summary:
 if x['region']=='start':print(x['seed'],x['arm'],x['old_offset'],x['new_offset'],'ref',round(x['reference_target_drift']['mean'],3),'ref_abs',round(x['reference_target_drift']['mean_abs'],3),'inc_abs',round(x['accepted_increment_drift']['mean_abs'],4),'A,V,C',*[round(x['main_and_interactions'][k]['mean'],3) for k in ('A','V','C')],'inter_abs',round(x['sum_interaction_mean_abs'],3))
print('MAIN_C_DOMINANCE',sum(x['largest_main_effect_by_mean_abs_each_repeat']==['C','C'] for x in summary),len(summary))
