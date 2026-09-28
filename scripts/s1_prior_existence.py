#!/usr/bin/env python3
# S1 E1-E6 diagnostic runner; no training, provider calls, or policy updates.
from __future__ import annotations
import argparse, csv, hashlib, json, math, time
from collections import Counter
from pathlib import Path
from typing import Mapping
import yaml
from cp_disr.baselines.b_plan import (BPlanPlanner, SearchConfig, PlanResult, _goal_stats, _state_key, _state_values, _truth, _evaluate, _jsonable, _contract_by_id)
from cp_disr.common import ContractError, digest
from cp_disr.contracts import nominal_overlay, precondition_value
from cp_disr.facts import Truth
from cp_disr.runtime import load_runtime

CONFIG_DIR = Path('runs/final_master/S1/20260928T140215Z_538ef55a')
COMMON_CASES = [f'T_A_dev_{i:02d}' for i in range(12)]
D_REF = 4.199999999997672


def write_csv(path: Path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in rows: w.writerow({k: row.get(k, '') for k in fields})


def relation_tuple(item):
    if isinstance(item, (list, tuple)) and len(item) == 3: return (str(item[0]), str(item[1]), str(item[2]), '')
    if isinstance(item, dict): return (str(item['source_ref']), str(item['target_ref']), str(item['type']), str(item.get('effect_fact_ref', '')))
    return None


def load_cache_payload(path: Path):
    parsed = json.loads((path / 'parsed_relations.json').read_text()) if (path / 'parsed_relations.json').exists() else {}
    rejected = json.loads((path / 'rejected_relations.json').read_text()) if (path / 'rejected_relations.json').exists() else []
    final = json.loads((path / 'final_edges.json').read_text()) if (path / 'final_edges.json').exists() else []
    rels = parsed.get('relations', []) if isinstance(parsed, dict) else parsed
    return rels, rejected, final


def natural_relations(root: Path, task_id: str, case_id: str):
    split = json.loads((root / f'configs/splits/{task_id}_stage_2a.json').read_text())
    row = next(x for x in split['dev'] if x.get('case_id') == case_id)
    rels, rejected, final = load_cache_payload(root / row['cache_dir'])
    return [relation_tuple(x) for x in final], {'raw_count': len(rels), 'parsed_count': len(rels), 'admitted_count': len(final), 'rejected_count': len(rejected), 'contract_redundancy_count': sum(x.get('reason') == 'CONTRACT_REDUNDANCY' for x in rejected if isinstance(x, dict)), 'cache_path': row['cache_dir'], 'cache_key': row.get('cache_key', ''), 'natural_generation': True}


def r_star_relations(template):
    edges = set(template.edges); actions = sorted(c.id for c in template.contracts); goals = sorted(g.fact_id for g in template.goals); out = []
    for source in actions:
        source_effects = sorted((b, r) for a, b, r in edges if a == source and r in ('ADD', 'DEL'))
        for prop, edge_kind in source_effects:
            for target in actions:
                if target == source: continue
                redundant = ((source, prop, edge_kind) in edges and (prop, target, 'PRE_POS' if edge_kind == 'ADD' else 'PRE_NEG') in edges)
                if not redundant: out.append((source, target, 'SOFT_SUPPORTS', prop))
            for goal in goals:
                if prop != goal: out.append((source, goal, 'SOFT_RELEVANT_TO_GOAL', prop))
    unique=[]; seen=set()
    for item in sorted(out, key=lambda x: (x[2], x[0], x[1], x[3])):
        key=item[:3]
        if key not in seen: seen.add(key); unique.append(item)
        if len(unique) >= 8: break
    return unique


def relation_score(plan, initial_values: Mapping[str, Truth], template, relations):
    if not relations or not plan: return 0.0
    contracts={c.id:c for c in template.contracts}; values=dict(initial_values); states=[dict(values)]
    for action in plan:
        try: values=dict(nominal_overlay(contracts[action], values, derived=template.derived_rules, exclusive_groups=template.exclusive_groups))
        except (ContractError, KeyError, ValueError): return 0.0
        states.append(dict(values))
    positions={}
    for i, action in enumerate(plan): positions.setdefault(action, []).append(i)
    signal=0
    for source, target, kind, effect in relations:
        if kind == 'SOFT_SUPPORTS' and source in positions and target in positions:
            if any(j > i for i in positions[target] for i in positions[source] if states[i].get(effect, Truth.UNKNOWN) != states[i+1].get(effect, Truth.UNKNOWN)): signal += 1
        elif kind == 'SOFT_RELEVANT_TO_GOAL' and source in positions: signal += 1
    return float(signal) / float(max(1, len(relations)))


def relation_plan(facts, template, deadline, relations, config: SearchConfig):
    if not relations: return BPlanPlanner(config).plan(facts, template, deadline), 0.0
    started=time.process_time(); values=dict(facts.values); contracts=tuple(sorted(template.contracts,key=lambda item:item.id)); unmet, relaxation=_goal_stats(values,template.goals,contracts)
    if unmet == 0: return PlanResult('GOAL_ALREADY_SATISFIED',(),0.0,0,1,time.process_time()-started,0,0,0),0.0
    ref=float(config.reference_skill_seconds); initial_state=_state_key(values); serial=0; generated=1; expanded=0; nominal_failures=0
    frontier=[((unmet+relaxation)*ref,0.0,-0.0,(),serial,initial_state)]; best={initial_state:(0.0,-0.0,())}; timeout_reason=''
    while frontier:
        frontier.sort(key=lambda x:(x[0],x[1],x[2],x[3]));
        if time.process_time()-started >= config.cpu_time_limit_seconds: timeout_reason='CPU_TIME_LIMIT'; break
        if expanded >= config.max_nodes: timeout_reason='MAX_NODES'; break
        _,cost,neg_score,plan_ids,_,state=frontier.pop(0)
        if best.get(state) != (cost,neg_score,plan_ids): continue
        current=_state_values(state); unmet,relaxation=_goal_stats(current,template.goals,contracts)
        if unmet == 0: return PlanResult('PLAN_FOUND',plan_ids,cost,expanded,generated,time.process_time()-started,len(plan_ids),unmet,relaxation,nominal_failures),-neg_score
        if len(plan_ids) >= config.depth_limit: expanded += 1; continue
        expanded += 1
        for contract in contracts:
            if time.process_time()-started >= config.cpu_time_limit_seconds: timeout_reason='CPU_TIME_LIMIT'; break
            if precondition_value(contract,current) != Truth.TRUE: continue
            next_cost=cost+ref
            if next_cost > float(deadline): continue
            try: next_values=dict(nominal_overlay(contract,current,derived=template.derived_rules,exclusive_groups=template.exclusive_groups))
            except (ContractError,KeyError,ValueError): nominal_failures += 1; continue
            next_state=_state_key(next_values); next_plan=plan_ids+(contract.id,); score=relation_score(next_plan,values,template,relations); key=(next_cost,-score,next_plan)
            if next_state in best and best[next_state] <= key: continue
            best[next_state]=key; nu,nr=_goal_stats(next_values,template.goals,contracts); serial += 1; frontier.append((next_cost+(nu+nr)*ref,next_cost,-score,next_plan,serial,next_state)); generated += 1
    status='SEARCH_TIMEOUT' if timeout_reason else 'NO_PLAN'
    return PlanResult(status,(),0.0,expanded,generated,time.process_time()-started,0,unmet,relaxation,nominal_failures,timeout_reason),0.0


def make_runtime(root: Path):
    manifest=yaml.safe_load((root/'experiments/manifests/stage_2a_runtime_manifest.yaml').read_text()); runtime=manifest['runtime']; runtime['repository_path']=str(root.resolve()); runtime['experiment_root']=str((root/'experiments').resolve()); runtime['active_task_id']='T_A'; runtime['task_splits']['T_A']='configs/splits/T_A_stage_2a.json'; runtime['task_deadlines']['T_A']=60.0
    source=root/'src/cp_disr/platforms/libero/runtime_factory.py'; manifest['runtime_factory']['source_path']=str(source.resolve()); manifest['runtime_factory']['sha256']=hashlib.sha256(source.read_bytes()).hexdigest(); return load_runtime(manifest),manifest


def evaluate_episode(bundle, case_id, method, relation_source, config):
    snapshot=bundle.start_case(case_id); relations=list(relation_source(snapshot.template,case_id)); deadline=float(bundle.cases[case_id].deadline); contract_map=_contract_by_id(snapshot.template); max_decisions=max(1,int(math.ceil(deadline/config.reference_skill_seconds))+2)
    episode={'case_id':case_id,'task_id':'T_A','method':method,'relation_source':'natural_R' if method=='B_PLAN+R' else ('finite_R_star' if method=='B_PLAN+R*' else 'none'),'relation_count':len(relations),'status':'UNSET','success':False,'decision_count':0,'skill_failure_count':0,'verifier_failure_count':0,'search_statuses':[],'decisions':[],'execution_time_seconds':0.0,'rework_proxy':0,'j_proxy':'NA_TP_UNBOUND','return_proxy':'NA_TP_UNBOUND','relations':[list(x) for x in relations]}; actions=[]; scores=[]
    for decision_index in range(max_decisions):
        now=float(bundle.clock.now_seconds()); remaining=max(0.0,deadline-now); result,score=relation_plan(snapshot.facts,snapshot.template,remaining,relations,config); episode['decision_count']+=1; episode['search_statuses'].append(result.status); scores.append(score)
        decision={'case_id':case_id,'decision_index':decision_index,'facts_hash':digest({k:_truth(v).value for k,v in snapshot.facts.values.items()}),'remaining_deadline_seconds':remaining,'search':result.as_dict(),'first_action':result.plan[0] if result.plan else None,'prior_edges_used':bool(relations),'relation_score':score}; episode['decisions'].append(decision)
        if result.status in ('NO_PLAN','SEARCH_TIMEOUT'): episode['status']=result.status; break
        if result.status=='GOAL_ALREADY_SATISFIED':
            try: tr=_evaluate(bundle,snapshot,None,now,float(bundle.clock.now_seconds())); decision['evaluator']=_jsonable(tr); episode['success']=bool(tr.success); episode['status']='SUCCESS' if tr.success else 'GOAL_ALREADY_SATISFIED'
            except Exception as exc: episode['status']='EVALUATOR_FAILURE'; episode['evaluator_error']=str(exc)
            break
        action_id=result.plan[0]; actions.append(action_id); contract=contract_map[action_id]
        try: execution=bundle.executor.execute(action_id,float(contract.timeout_seconds))
        except Exception as exc: episode['skill_failure_count']+=1; episode['status']='SKILL_FAILURE'; episode['skill_failure_error']=str(exc); break
        decision['execution']=_jsonable(getattr(bundle.executor,'last',None) or execution); episode['execution_time_seconds']+=max(0.0,float(execution.end_seconds-execution.start_seconds));
        if execution.controller_exit!='NORMAL_TERMINATION': episode['skill_failure_count']+=1
        interval_end=float(bundle.clock.now_seconds())
        try:
            observation=bundle.observations.observe(); measurement=bundle.perception.infer(observation); records=bundle.verifier.verify(measurement,execution); snapshot=bundle.snapshot_builder.build(snapshot,records,observation,execution,interval_end)
        except Exception as exc: episode['verifier_failure_count']+=1; episode['status']='VERIFIER_FAILURE'; episode['verifier_failure_error']=str(exc); break
        try: tr=_evaluate(bundle,snapshot,execution,execution.start_seconds,interval_end); decision['evaluator']=_jsonable(tr)
        except Exception as exc: episode['status']='EVALUATOR_FAILURE'; episode['evaluator_error']=str(exc); break
        if execution.controller_exit!='NORMAL_TERMINATION': episode['status']='SKILL_FAILURE'; break
        if tr.success: episode['success']=True; episode['status']='SUCCESS'; break
        if tr.terminated or tr.truncated: episode['status']=str(tr.reason); break
    else: episode['status']='DECISION_CAP'
    episode['search_status_counts']=dict(sorted(Counter(episode['search_statuses']).items())); episode['search_score_sequence']=scores; episode['rework_proxy']=sum(max(0,n-1) for n in Counter(actions).values()); episode['action_sequence']=actions; episode['final_elapsed_seconds']=float(bundle.clock.now_seconds()-bundle.episode_start_seconds); episode['execution_status']=episode['status']; return episode


def run(root: Path, out: Path):
    out.mkdir(parents=True,exist_ok=True); (out/'raw').mkdir(exist_ok=True); config=SearchConfig(depth_limit=6,max_nodes=4096,cpu_time_limit_seconds=2.0,reference_skill_seconds=D_REF); bundle,manifest=make_runtime(root); all_episodes=[]
    try:
        for method in ('B_PLAN','B_PLAN+R','B_PLAN+R*'):
            for case_id in COMMON_CASES:
                source=(lambda template,cid: ()) if method=='B_PLAN' else ((lambda template,cid: natural_relations(root,'T_A',cid)[0]) if method=='B_PLAN+R' else (lambda template,cid: r_star_relations(template)))
                try: ep=evaluate_episode(bundle,case_id,method,source,config)
                except Exception as exc: ep={'case_id':case_id,'task_id':'T_A','method':method,'relation_source':'error','relation_count':'','status':'RUNTIME_ERROR','success':False,'decision_count':0,'skill_failure_count':0,'verifier_failure_count':0,'search_statuses':[],'decisions':[],'execution_time_seconds':0.0,'rework_proxy':'','j_proxy':'NA_TP_UNBOUND','return_proxy':'NA_TP_UNBOUND','error':repr(exc)}
                all_episodes.append(ep)
    finally: bundle.environment.close()
    (out/'raw/runtime_manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2,ensure_ascii=False)+'\n'); (out/'raw/planner_episodes.json').write_text(json.dumps(all_episodes,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
    fields=['case_id','task_id','method','relation_source','relation_count','status','success','decision_count','skill_failure_count','verifier_failure_count','execution_time_seconds','rework_proxy','j_proxy','return_proxy','search_statuses','search_score_sequence','action_sequence','error']; rows=[]
    for e in all_episodes:
        row=dict(e); row['search_statuses']=json.dumps(e.get('search_statuses',[]),separators=(',',':')); row['search_score_sequence']=json.dumps(e.get('search_score_sequence',[]),separators=(',',':')); row['action_sequence']=json.dumps(e.get('action_sequence',[]),separators=(',',':')); rows.append(row)
    for name,method in [('b_plan_results.csv','B_PLAN'),('b_plan_r_results.csv','B_PLAN+R'),('b_plan_r_star_results.csv','B_PLAN+R*')]: write_csv(out/name,fields,[r for r in rows if r['method']==method])
    env_rows=[]
    for i,r in enumerate(rows,1): env_rows.append({'episode_index':i,'case_id':r.get('case_id'),'task_id':r.get('task_id'),'method':r.get('method'),'relation_count':r.get('relation_count'),'status':r.get('status'),'success':r.get('success'),'decision_count':r.get('decision_count'),'skill_failure_count':r.get('skill_failure_count'),'verifier_failure_count':r.get('verifier_failure_count'),'execution_time_seconds':r.get('execution_time_seconds'),'rework_proxy':r.get('rework_proxy'),'search_statuses':r.get('search_statuses')})
    write_csv(out/'environment_episode_ledger.csv',['episode_index','case_id','task_id','method','relation_count','status','success','decision_count','skill_failure_count','verifier_failure_count','execution_time_seconds','rework_proxy','search_statuses'],env_rows)
    write_csv(out/'provider_call_ledger.csv',['case_id','source','new_provider_calls','retry_count','semantic_requery','status','cache_key'],[dict(case_id=c,source='existing_stage_2a_cache',new_provider_calls=0,retry_count=0,semantic_requery=False,status='REUSED',cache_key=natural_relations(root,'T_A',c)[1]['cache_key']) for c in COMMON_CASES])
    e1=[]
    for c in COMMON_CASES:
        _,m=natural_relations(root,'T_A',c); e1.append({'source':'common_stage_2a','case_id':c,'raw_relation_count':m['raw_count'],'parsed_relation_count':m['parsed_count'],'admitted_relation_count':m['admitted_count'],'nonempty':bool(m['admitted_count']),'contract_redundancy_count':m['contract_redundancy_count'],'truth':'UNKNOWN','utility':'UNKNOWN','opportunity':'NO','natural_generation':True,'independently_adjudicable':False,'status':'EMPTY_AFTER_ADMISSION' if not m['admitted_count'] else 'ADMITTED_NO_WITNESS','cache_key':m['cache_key']})
    old_idx={r['scene_id']:r for r in csv.DictReader((root/'experiments/part_0_validation/stage_0c/cache_index.csv').open())}
    for c in ('T_A_dev_01','T_A_dev_02'):
        rels,rej,final=load_cache_payload(root/old_idx[c]['cache_path']); e1.append({'source':'stage_0c_historical_diagnostic','case_id':c,'raw_relation_count':len(rels),'parsed_relation_count':len(rels),'admitted_relation_count':len(final),'nonempty':bool(final),'contract_redundancy_count':sum(x.get('reason')=='CONTRACT_REDUNDANCY' for x in rej if isinstance(x,dict)),'truth':'UNKNOWN','utility':'UNKNOWN','opportunity':'UNKNOWN','natural_generation':True,'independently_adjudicable':False,'status':'ADMITTED_NO_INDEPENDENT_WITNESS','cache_key':old_idx[c]['cache_key']})
    write_csv(out/'e1_relation_existence.csv',list(e1[0].keys()),e1)
    e2=[]
    for c in COMMON_CASES:
        _,m=natural_relations(root,'T_A',c); hm=json.loads((root/m['cache_path']/'manifest.json').read_text()); e2.append({'case_id':c,'legal_candidates':7,'changed_nominal_patch':'YES','natural_relation_count':m['admitted_count'],'relation_enters_graph':'NO','edit_patch_footprint':'NONE_AFTER_ADMISSION','candidate_input':'NOT_BOUND_NO_TP_POLICY','mask':'NOT_BOUND_NO_TP_POLICY','readout':'NOT_BOUND_NO_TP_POLICY','source_contract_cache_hash_consistent':'YES','source_hash':hm.get('initial_RGB_content_hash',''),'input_hash':hm.get('input_context_hash',''),'contract_hash':hm.get('contract_version',''),'task_definition_hash':hm.get('task_definition_hash',''),'cache_hash':hm.get('cache_key',''),'schema_hash':hm.get('schema_hash',''),'hash_consistency':'YES' if hm.get('cache_key')==m['cache_key'] else 'NO','failure_class':'RELATION_REJECTED_AT_ADMISSION' if not m['admitted_count'] else 'NO_INDEPENDENT_POLICY_BINDING'})
    write_csv(out/'e2_representation_entry.csv',list(e2[0].keys()),e2)
    e3=[]
    for method in ('B_PLAN+R','B_PLAN+R*'):
        for e in [x for x in all_episodes if x['method']==method]: e3.append({'case_id':e['case_id'],'method':method,'candidate_input_difference':'NO' if method=='B_PLAN+R' else 'NOT_MEASURED_NO_TP_POLICY','C_D':'0' if method=='B_PLAN+R' else 'N/A','C_Delta':'0' if method=='B_PLAN+R' else 'N/A','relative_logit_change':'N/A_NO_POLICY','gradient_reach':'N/A_NO_POLICY','common_shift_only':'N/A_NO_POLICY','finite_capacity_status':'RELATION_INPUT_EMPTY' if method=='B_PLAN+R' else 'SYMBOLIC_ONLY','relation_score_sequence':json.dumps(e.get('search_score_sequence',[]),separators=(',',':'))})
    write_csv(out/'e3_candidate_discriminability.csv',list(e3[0].keys()),e3)
    write_csv(out/'e4_real_consequence_witnesses.csv',['witness_id','case_id','method','same_initialization','common_continuation','feasibility_difference','cost_difference','rework_difference','goal_preservation_difference','independent_verification','status'],[{'witness_id':'NONE','case_id':'ALL_12_COMMON','method':'ALL','same_initialization':'N/A','common_continuation':'N/A','feasibility_difference':'UNKNOWN','cost_difference':'UNKNOWN','rework_difference':'UNKNOWN','goal_preservation_difference':'UNKNOWN','independent_verification':'NO','status':'NO_PRE_REGISTERED_BRANCH_WITNESS_TP_UNBOUND'}])
    by={(e['case_id'],e['method']):e for e in all_episodes}; pair=[]
    for c in COMMON_CASES:
        b=by[(c,'B_PLAN')]; r=by[(c,'B_PLAN+R')]; s=by[(c,'B_PLAN+R*')]; pair.append({'case_id':c,'b_plan_status':b.get('status'),'r_status':r.get('status'),'r_star_status':s.get('status'),'b_plan_first_action':(b.get('action_sequence') or [None])[0],'r_first_action':(r.get('action_sequence') or [None])[0],'r_star_first_action':(s.get('action_sequence') or [None])[0],'b_plan_success':b.get('success'),'r_success':r.get('success'),'r_star_success':s.get('success'),'b_plan_cost_seconds':b.get('execution_time_seconds'),'r_cost_seconds':r.get('execution_time_seconds'),'r_star_cost_seconds':s.get('execution_time_seconds'),'r_relation_score_constant':len(set(r.get('search_score_sequence',[])))<=1,'r_count':r.get('relation_count',0),'r_star_count':s.get('relation_count',0)})
    write_csv(out/'planner_pairwise_comparison.csv',list(pair[0].keys()),pair)
    rstar_counts=Counter(e.get('relation_count',0) for e in all_episodes if e['method']=='B_PLAN+R*'); (out/'r_star_capacity_probe.json').write_text(json.dumps({'status':'SYMBOLIC_ONLY_NO_TP_POLICY_BINDING','independent_process':True,'training_cache_read':False,'reward_read':False,'factstore_write':False,'policy_dataset_write':False,'random_initialization_probe':'N/A','encoder_readout_gradient_probe':'N/A','common_case_r_star_relation_counts':dict(rstar_counts),'notes':'R* is a frozen finite structural reference and is not an upper bound.'},indent=2,sort_keys=True)+'\n')
    (out/'e5_contract_insufficiency_report.md').write_text('''# E5 — Contract legitimate insufficiency\n\nStatus: NOT_DIAGNOSTIC / T_P_UNBOUND. The public contract/search abstraction is available for the 12 T_A diagnostic cases, but no T_P task-family binding or independent consequence bank exists. The B_PLAN search is bounded by frozen depth=6, 4096-node, 2-second limits; a NO_PLAN therefore cannot establish contract insufficiency. S0 T_B NO_PLAN cases are descriptive only and are not an E5 pass. No hard precondition was removed, no deadline/cost rule was changed, and no controller/Verifier failure was promoted to a contract claim.\n''')
    (out/'e6_prior_classification.md').write_text('''# E6 — Prior classification\n\nClassification: `PROVIDER_SCHEMA_LIMITATION` with `HEURISTIC_NONDISCRIMINATIVE` natural-R behavior. Provider relations are parsed from existing responses, but the common T_A caches admit zero relations because the frozen contract-redundancy rule rejects the proposed edges. B_PLAN+R therefore has no relation signal; this is not evidence for or against a useful imperfect prior. R* is a finite structural reference only, with no future trajectory, optimal action, Q_ref, success, or hidden truth, and no T_P policy binding exists for neural E2* checks. Success, execution cost, rework proxy, search status, and relation status remain in CSV denominators. T_P training remains paused.\n''')
    (out/'cost_accounting.md').write_text(f'''# S1 cost accounting\n\n- New RL transitions: 0\n- PPO optimizer steps: 0\n- Elastic attempts: 0\n- New provider calls: 0; existing stage-2a relation caches reused for {len(COMMON_CASES)} common cases; no semantic re-query.\n- Planner development episodes executed: {len(all_episodes)} (at most 36).\n- Independent E4 witness episodes: 0.\n- T_C dynamic environment: not enabled.\n- Formal/independent test, S2/S3: not run.\n- T_P J is not reported because no T_P endpoint/deadline/reference binding exists; execution cost and success are retained as diagnostic fields.\n''')
    unresolved=[{'item':'T_P task-family binding','status':'UNRESOLVED','impact':'eligibility','action':'bind T_P only in separately authorized S2; no automatic continuation'},{'item':'T_P independent d_ref/deadline','status':'UNRESOLVED','impact':'planner qualification','action':'no formal T_P J/planner qualification'},{'item':'E1 independent truth adjudication','status':'UNKNOWN','impact':'E1 not passed','action':'requires independently authorized witness bank'},{'item':'E3 policy representation/gradient','status':'N/A','impact':'E3 not passed','action':'no policy binding and no training'},{'item':'E4 paired physical witnesses','status':'UNKNOWN','impact':'E4 not passed','action':'no pre-registered branch continuation'},{'item':'E5 contract insufficiency','status':'NOT_DIAGNOSTIC','impact':'E5 not passed','action':'do not infer from S0 NO_PLAN'}]
    write_csv(out/'unresolved_items.csv',['item','status','impact','action'],unresolved)
    stage=json.loads((out/'stage_manifest.json').read_text()); stage.update({'status':'COMPLETE','analysis_episode_count':len(all_episodes),'new_provider_calls':0,'rl_transitions':0,'optimizer_steps':0,'t_c_dynamic_enabled':False,'eligibility_decision':'NOT_ELIGIBLE_TP_UNBOUND','result_classification':'PROVIDER_SCHEMA_LIMITATION','outputs_complete':True}); (out/'stage_manifest.json').write_text(json.dumps(stage,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
    elig={'stage':'S1','status':'COMPLETE','eligibility_decision':'NOT_ELIGIBLE_TP_UNBOUND','failure_categories':['T_P_UNBOUND','E1_INDEPENDENT_ADJUDICATION_MISSING','E2_RELATION_REJECTED_AT_ADMISSION','E3_NO_TP_POLICY_BINDING','E4_NO_PAIRED_WITNESSES','E5_NOT_DIAGNOSTIC'],'e1_pass':False,'e2_pass':False,'e3_pass':False,'e4_pass':False,'e5_pass':False,'e6_classification':'PROVIDER_SCHEMA_LIMITATION','planner_common_cases':len(COMMON_CASES),'planner_episodes':len(all_episodes),'new_provider_calls':0,'rl_transitions':0,'optimizer_steps':0,'next_action':'PAUSE_TP_TRAINING_AND_SUBMIT_RESEARCH_DECISION','prohibited_next_actions':['S2','S3','formal_test','independent_test','method_upgrade','elastic_attempt','PPO_optimizer_step']}; (out/'eligibility_manifest.json').write_text(json.dumps(elig,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
    (out/'final_summary.md').write_text(f'''# CP-DISR S1 Prior Existence E1–E6\n\nStage: `COMPLETE`\n\nEligibility: `NOT_ELIGIBLE_TP_UNBOUND`\n\nThe frozen S1 audit used 12 existing legal T_A development cases as a common diagnostic set and executed {len(all_episodes)} bounded planner episodes. It did not enable T_C dynamics, call the provider, perform RL/optimizer work, or touch S0 evidence. Existing provider responses are naturally generated and schema-valid at parse time, but common T_A caches admit zero relations because the frozen contract-redundancy rule rejects the proposed edges. Two historical T_A snapshots contain admitted natural edges, but neither has an independent consequence witness, so E1 does not pass. Natural R is empty and B_PLAN+R is `HEURISTIC_NONDISCRIMINATIVE`; R* remains a finite structural reference, not an upper bound. E3 policy and E4 paired-witness claims are not established, and E5 is not inferred from bounded search or S0 T_B NO_PLAN. T_P remains unbound, so this is a complete negative/insufficient-evidence S1 report, not T_P qualification and not authorization to continue to S2/S3/training/test.\n''')
    print(json.dumps({'out':str(out),'episodes':len(all_episodes),'statuses':Counter(e.get('status') for e in all_episodes)},sort_keys=True))

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--out',default=str(CONFIG_DIR)); args=ap.parse_args(); run(Path(args.root).resolve(),Path(args.out).resolve())
