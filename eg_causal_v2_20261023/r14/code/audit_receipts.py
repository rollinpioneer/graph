"""Read-only R1.4 receipt audit. No torch/model/data/prediction access.

Writes its compact result to stdout; caller may redirect it into an audit file.
"""
import argparse, json, math
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path.home()/'work/grasp/eg_causal_mechanism_v2_20261023/r14');args=ap.parse_args()
root=args.root;errors=[]
def check(ok,label,detail=None):
 if not ok:errors.append(dict(check=label,detail=detail))
def read(p):
 try:return json.loads(p.read_text())
 except Exception as exc:errors.append(dict(check='read_json',file=str(p.relative_to(root)),detail=str(exc)));return {}
def number(x):return isinstance(x,(int,float)) and math.isfinite(x)

protocol=read(root/'protocol_R14_frozen.json');scenes=[x['scene'] for x in protocol.get('update_batches',[])];arms=protocol.get('arms',[])
check(len(scenes)==8 and len(set(scenes))==8,'eight_registered_sources',scenes)
check(len(arms)==7 and len(set(arms))==7,'seven_registered_arms',arms)
expected={(s,a) for s in scenes for a in arms};found={};modelhashes=set();totalsteps=0;norm_errors=[];max_initial_logit=0.;histories={};source_summary=[]
for path in sorted((root/'receipts').glob('src_*/*.json')):
 rec=read(path);key=(rec.get('source_scene'),rec.get('arm'))
 check(key in expected,'unexpected_branch',str(path.relative_to(root)))
 check(key not in found,'duplicate_branch',key);found[key]=rec
check(set(found)==expected,'complete_branch_set',dict(expected=len(expected),found=len(found),missing=[list(x) for x in sorted(expected-set(found))]))

mapping=read(root/'optimizer_mapping.json');params=mapping.get('parameters',[])
check(len(params)==152 and len({x.get('index') for x in params})==152 and len({x.get('name') for x in params})==152,'152_unique_Adam_parameter_entries')
check(mapping.get('checkpoint_sha256')==protocol.get('checkpoint'),'optimizer_checkpoint_identity')
check(bool(mapping.get('groups')) and all(g.get('lr')==protocol.get('learning_rate') for g in mapping.get('groups',[])),'original_Adam_learning_rate')

for s in scenes:
 pref=read(root/'receipts'/f'train_{s:04d}_preflight.json');den=pref.get('valid');r0=found.get((s,'R0_original'),{});r0rows=r0.get('steps',[])
 check(pref.get('shared_step1_graph') is True,'shared_native_initial_graph',s)
 check(pref.get('points_views_exact') is True and pref.get('AD_classes_exact') is True,'source_query_and_discrete_controls',s)
 check(all(v==0 for v in pref.get('fixed_path_logits_max_abs',{'missing':1}).values()),'same_backbone_exact_logits',s)
 check(all(number(v) and v>0 for v in pref.get('step1_backbone_gradient_norms',{}).values()) and set(pref.get('step1_backbone_gradient_norms',{}))=={'total','zero','random'},'true_backbone_gradients_all_components',s)
 check(number(den) and den>0,'positive_original_valid_denominator',s)
 # Earlier successful common graphs receive the same AD audit from cached
 # logits, so no initial computation or parameter update is repeated.
 adpath=root/'receipts'/f'train_{s:04d}_existing_common_graph_AD_audit.json'
 if adpath.exists():
  ad=read(adpath);changes=ad.get('changes',{})
  check(ad.get('new_forwards')==0 and ad.get('parameter_updates')==0,'cached_AD_audit_no_new_execution',s)
  check(len(changes)==2 and all(v==0 for x in changes.values() for v in x.values()),'historical_AD_exact',s)
 else:check(pref.get('historical_AD_classes_exact') is True,'historical_AD_gate_documented',s)
 fold=next((x for x in protocol.get('folds',[]) if s not in x),[])
 count=0
 for arm in arms:
  rec=found.get((s,arm),{});rows=rec.get('steps',[]);steps=[x.get('step') for x in rows]
  check(rec.get('completed_steps')==5 and steps==[1,2,3,4,5],'exact_five_step_trajectory',[s,arm,steps])
  count+=len(rows);totalsteps+=len(rows)
  h=rec.get('initial_model_and_buffer_hash');modelhashes.add(h)
  check(isinstance(h,str) and len(h)==64,'initial_model_buffer_hash_present',[s,arm])
  check(rec.get('optimizer_state_restored') is True and rec.get('checkpoint')==protocol.get('checkpoint'),'original_checkpoint_and_Adam_restored',[s,arm])
  check(rec.get('shared_native_graph_step1') is True and rec.get('fresh_full_network_steps')==[2,3,4,5] and rec.get('backbone_recomputed') is True,'no_frozen_training_features',[s,arm])
  check(rec.get('denominator')==den and rec.get('zero_count')==pref.get('zero') and rec.get('random_q0_overlap')==pref.get('random_overlap'),'source_fixed_population',[s,arm])
  check(rec.get('holdout_scenes')==fold and rec.get('training_frame')==0 and rec.get('holdout_frame')==128,'registered_opposite_fold',[s,arm])
  check(rec.get('training_TF32') is False and rec.get('heldout_TF32') is True and rec.get('train_BN_dropout') is True,'registered_precision_and_modes',[s,arm])
  if rows:
   first=rows[0];err=first.get('initial_forward_max_abs',{})
   check(len(err)==4 and all(v==0 for v in err.values()),'seven_arm_initial_logits_exact',[s,arm]);max_initial_logit=max([max_initial_logit]+[v for v in err.values() if number(v)])
   gc=first.get('gradient_assignment_check') or {}
   check(gc.get('assignment_relative_L2')==0 and gc.get('ADQ_final_output_gradients_exact') is True and gc.get('shared_native_full_network_graph') is True,'step1_gradient_scope',[s,arm])
   check(gc.get('local_final_width_retained')==arm.startswith('RW_local'),'local_width_scope',[s,arm])
   if r0rows:check(first.get('loss')==r0rows[0].get('loss') and first.get('zero_width_loss')==r0rows[0].get('zero_width_loss'),'same_initial_forward_losses',[s,arm])
  for j,row in enumerate(rows):
   check(row.get('valid_denominator')==den,'unchanged_N_valid',[s,arm,j+1])
   check(isinstance(row.get('forward_rng_hash'),str) and len(row['forward_rng_hash'])==64,'forward_RNG_hash_present',[s,arm,j+1])
   if j<len(r0rows):check(row.get('forward_rng_hash')==r0rows[j].get('forward_rng_hash'),'same_source_step_RNG',[s,arm,j+1])
   if 'actual_step_norm_matched' in arm and j<len(r0rows):
    target=r0rows[j].get('applied_delta_l2');actual=row.get('applied_delta_l2');raw=row.get('actual_adam_delta_l2');scale=row.get('scale')
    good=all(number(v) and v>0 for v in [target,actual,raw,scale]);check(good,'finite_positive_actual_norms',[s,arm,j+1])
    if good:
     rel=abs(actual-target)/target;norm_errors.append(rel)
     check(rel<.001,'actual_parameter_norm_match',[s,arm,j+1,rel]);check(row.get('R0_target_norm')==target,'same_step_R0_target',[s,arm,j+1])
     check(abs(scale-target/raw)<=1e-12*max(1,abs(scale)),'recorded_actual_delta_scale',[s,arm,j+1])
 complete=read(root/'receipts'/f'src_{s:04d}_complete.json')
 check(complete.get('source_scene')==s and complete.get('arms')==7 and complete.get('adam_steps')==35 and complete.get('parameters_restored') is True,'source_complete_restored',s)
 hfiles=sorted((root/'receipts').glob(f'train_{s:04d}_initial_history_check*.json'));hh=[read(p) for p in hfiles]
 check(bool(hh) and all(x.get('tolerance')==3e-5 for x in hh),'historical_threshold_retained',s)
 histories[str(s)]=[dict(file=p.name,passed=x.get('passed'),max_raw_logit_error=max(x['logits_max_abs'].values()) if x.get('logits_max_abs') else None,attempt_index=x.get('attempt_index',1)) for p,x in zip(hfiles,hh)]
 source_summary.append(dict(scene=s,branches=sum((s,a) in found for a in arms),steps=count,valid_denominator=den,initial_history_pass=pref.get('stage1_legacy_tolerance_pass',max(pref.get('stage1_logits_max_abs',{'missing':1}).values())<=3e-5),common_start_rebuilt_after_prior_history_failure=pref.get('common_start_rebuilt_after_prior_history_failure',False)))

check(totalsteps==280,'280_total_Adam_steps',totalsteps);check(len(norm_errors)==120,'24_norm_matched_branches_x5',len(norm_errors))
check(len(modelhashes)==1 and None not in modelhashes,'same_initial_model_and_buffers_all_56_branches',sorted(str(x) for x in modelhashes))
pending=[str(p.relative_to(root)) for p in (root/'journals').glob('*') if p.is_file()]
check(not pending,'no_unfinished_journals_or_recovery_snapshots',pending)
h64=histories.get('64',[]);pref64=read(root/'receipts'/'train_0064_preflight.json')
check(len(h64)==2 and h64[0]['file']=='train_0064_initial_history_check.json' and h64[0]['passed'] is False,'scene64_first_history_failure_preserved_once',h64)
check(pref64.get('common_start_rebuilt_after_prior_history_failure') is True and pref64.get('no_numeric_selection_or_retry') is True,'scene64_explicit_single_reconstruction')
if len(h64)==2:
 second=read(root/'receipts'/h64[1]['file']);check(second.get('attempt_index')==2 and second.get('prior_failed_history_checks')==1 and second.get('common_start_rebuilt_after_prior_history_failure') is True and second.get('no_numeric_selection_or_retry') is True,'scene64_reconstruction_provenance')
result=dict(status='PASS' if not errors else 'FAIL',scope='read-only JSON receipt audit; no model/data/prediction execution',sources=source_summary,branches=len(found),adam_steps=totalsteps,norm_matched_branches=len(norm_errors)//5,norm_matched_steps=len(norm_errors),maximum_actual_norm_relative_error=max(norm_errors) if norm_errors else None,maximum_step1_between_arm_logit_error=max_initial_logit,initial_model_and_buffer_hash=next(iter(modelhashes)) if len(modelhashes)==1 else None,remaining_journal_files=pending,historical_numeric_checks=histories,failures=errors)
print(json.dumps(result,indent=2,allow_nan=False));raise SystemExit(bool(errors))
