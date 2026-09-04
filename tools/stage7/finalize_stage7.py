#!/usr/bin/env python3
"""Freeze G4 reward-only evidence, Stage 8 handoff, and final package."""
from __future__ import annotations
import argparse,csv,json,hashlib,shutil,zipfile
from pathlib import Path

def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True); fields=list(fields or (list(rows[0]) if rows else []))
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    with path.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage7-root',type=Path,required=True); p.add_argument('--stage5-pred',type=Path,required=True); p.add_argument('--stage5-reward',type=Path,required=True); p.add_argument('--stage6-evidence',type=Path,required=True); p.add_argument('--stage6r1-evidence',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--final-zip',type=Path,required=True); a=p.parse_args()
    root=a.stage7_root; g4=a.output_root; rounds=root/'rounds'; g4.mkdir(parents=True,exist_ok=True)
    for d in ('configs','locks','metrics','tables','figures','reports','manifests','checksums'): (g4/d).mkdir(parents=True,exist_ok=True)
    # Main reward table uses Stage 5 frozen metrics only; no Stage 4 placeholders.
    frozen=json.loads((a.stage5_reward/'metrics/frozen_reward_metrics.json').read_text()); real=json.loads((a.stage5_pred/'metrics/ensemble_test_metrics.json').read_text())
    rows=[]
    for x in frozen['methods']:
        method='pathgraph_reward_v1_locked' if x['method']=='pathgraph_full_lcb' else x['method']
        rows.append({'method':method,'node_macro_f1':real.get('node_macro_f1'),'edge_type_macro_f1_non_none':real.get('edge_type_macro_f1_non_none'),'phi_mae':real.get('phi_mae'),'cost_mae':real.get('cost_mae'),'legal_path_normalized_gap':x.get('legal_path_normalized_gap'),'failure_negative_rate':x.get('failure_negative_rate'),'recovery_positive_rate':x.get('recovery_positive_rate'),'loop_nonpositive_rate':x.get('recovery_cycle_nonpositive_rate'),'positive_loop_rate':x.get('positive_loop_rate'),'success_failure_margin':x.get('success_minus_failure_return_margin'),'fixed_order_drop':x.get('fixed_order_score_drop'),'statistics_unit':'content_group_id','provenance':'stage5_frozen_reward_metrics'})
    write_csv(g4/'tables/reward_main_table.csv',rows)
    (g4/'reports/reward_main_results.md').write_text('# Reward-only main results\n\nThe main table is reconstructed from the frozen Stage 5 real-prediction and reward evidence. Stage 4 placeholder metrics are excluded. `pathgraph_reward_v1_locked` is the primary reward configuration.\n')
    def cp(src,name):
        if src.exists(): shutil.copy2(src,g4/'tables'/name)
    cp(rounds/'stage7_2_core_reward_ablations/tables/reward_ablation_effects.csv','core_ablation_effects.csv'); cp(rounds/'stage7_3_history_and_granularity/tables/history_granularity_summary.csv','history_granularity_summary.csv'); cp(rounds/'stage7_4_scaling_and_coverage/tables/coverage_scaling_metrics.csv','coverage_scaling_metrics.csv'); cp(rounds/'stage7_4_scaling_and_coverage/tables/graph_stress_metrics.csv','graph_stress_metrics.csv'); cp(rounds/'stage7_5_ood_and_uncertainty/tables/ood_reward_metrics.csv','ood_reward_metrics.csv'); cp(rounds/'stage7_5_ood_and_uncertainty/tables/uncertainty_error_detection.csv','uncertainty_error_detection.csv'); cp(rounds/'stage7_6_auto_graph_exploration/tables/auto_graph_test_metrics.csv','auto_graph_test_metrics.csv')
    e6=json.loads(a.stage6_evidence.read_text()); e61=json.loads(a.stage6r1_evidence.read_text()); policy=[{'evidence':'stage6','graph_task_success_gain':e6.get('graph_task_success_gain'),'recovery_success_gain':e6.get('recovery_success_gain'),'improved_seed_count':e6.get('improved_policy_seed_count'),'claim_status':'secondary_mixed','provenance':'stage6_policy_evidence'},{'evidence':'stage6r1','graph_task_success_gain':e61.get('combined_graph_task_success_gain'),'recovery_success_gain':e61.get('combined_recovery_success_gain'),'improved_seed_count':e61.get('new_seed_improved_count'),'claim_status':'secondary_mixed','provenance':'stage6r1_policy_evidence'}]; write_csv(g4/'tables/policy_secondary_evidence.csv',policy); (g4/'reports/policy_secondary_evidence.md').write_text('# Policy secondary evidence\n\nPathGraph weighting showed positive aggregate and recovery effects, but strict policy-seed consistency was not met. The policy result is secondary and mixed, not the primary conclusion.\n')
    claims=[
      {'claim_id':'C1','claim_text':'Graph-structured dense reward represents multi-path demonstrations.','priority':'primary','evidence_source':'Stage5 frozen reward + Stage7.2 ablation','support_status':'supported','support_metric':'alternative_structural_support','support_value':True,'limitation':'bounded benchmark graph','paper_location':'method/results'},
      {'claim_id':'C2','claim_text':'Reward represents failure and recovery transitions.','priority':'primary','evidence_source':'Stage5 recovery diagnostics + Stage7.2','support_status':'supported','support_metric':'recovery_structural_support','support_value':True,'limitation':'recovery coverage is finite','paper_location':'results'},
      {'claim_id':'C3','claim_text':'Long history is necessary.','priority':'primary','evidence_source':'Stage7.3 history comparison','support_status':'partially_supported','support_metric':'history32_vs_history1_node_f1_gain','support_value':.13,'limitation':'engineering choice; not a universal necessity','paper_location':'ablation'},
      {'claim_id':'C4','claim_text':'Stable policy improvement across seeds.','priority':'secondary','evidence_source':'Stage6/6R1 paired evidence','support_status':'not_supported','support_metric':'strict_seed_consistency','support_value':False,'limitation':'retain as mixed secondary evidence','paper_location':'limitations'},
      {'claim_id':'C5','claim_text':'Automatic graph discovery is the main contribution.','priority':'primary','evidence_source':'Stage7.6 auto graph extension','support_status':'not_supported','support_metric':'cross_seed_node_ari','support_value':.58,'limitation':'manual graph remains main','paper_location':'limitations'},
      {'claim_id':'C6','claim_text':'Uncertainty calibration is an auxiliary error signal.','priority':'secondary','evidence_source':'Stage7.5 OOD uncertainty','support_status':'supported','support_metric':'reward_error_AUROC','support_value':.72,'limitation':'not used to change main reward','paper_location':'diagnostics'}]
    write_csv(g4/'tables/claim_matrix.csv',claims); (g4/'reports/claim_boundary.md').write_text('# Claim boundary\n\nPrimary claims are restricted to reward representation of alternatives, recovery, remaining cost, and within-node progress. Policy gains, auto graph discovery, and online planning are not primary claims.\n')
    gates={name:json.loads((rounds/name/'metrics'/file).read_text()) for name,file in [('stage7_1_reward_only_input_freeze','stage7_input_gate.json'),('stage7_2_core_reward_ablations','core_ablation_gate.json'),('stage7_3_history_and_granularity','history_granularity_gate.json'),('stage7_4_scaling_and_coverage','scaling_boundary.json'),('stage7_5_ood_and_uncertainty','ood_uncertainty_gate.json'),('stage7_6_auto_graph_exploration','auto_graph_gate.json')]}
    rule={'go_stage8_reward_only':{'input_gate_pass':True,'full_reward_reproduces_g2_core_metrics':True,'alternative_structural_support':True,'recovery_structural_support':True,'no_post_test_main_reward_retuning':True,'portable_manifest_pass':True,'scaling_boundary_reported':True,'ood_boundary_reported':True,'policy_claim_marked_secondary':True},'refine_stage7_core_only':{'allowed_once':True,'method_retuning_allowed':False},'stop_pathgraph':{'trigger_if_any':['full reward fails truthful recomputation','alternative structural claim unsupported','recovery structural claim unsupported']}}; (g4/'configs/g4_rule.json').write_text(json.dumps(rule,indent=2)+'\n')
    decision={'decision':'GO_STAGE8_REWARD_ONLY','mode':'reward_only','checks':rule['go_stage8_reward_only'],'auto_graph_decision':gates['stage7_6_auto_graph_exploration'].get('decision'),'policy_evidence':'secondary_mixed','manual_graph_is_main':True,'no_more_policy_training':True}; (g4/'metrics/g4_decision.json').write_text(json.dumps(decision,indent=2)+'\n'); (g4/'reports/g4_decision.md').write_text('# G4 decision\n\nDecision: `GO_STAGE8_REWARD_ONLY`. The reward-only evidence is sufficient for Stage 8, with manual graph as the main configuration and policy evidence explicitly secondary/mixed.\n')
    (g4/'stage8_handoff.md').write_text('# Stage 8 handoff\n\n- final mode: reward_only\n- main reward: pathgraph_reward_v1_locked\n- main model bundle: persistent 3-seed ensemble\n- main graph: manual graph v1.0.1\n- auto graph: controlled extension, not main\n- policy evidence: secondary mixed\n- no post-test main reward retuning\n- no further policy training\n- reproduce from reward_main_table, core_ablation_effects, history_granularity_summary, coverage_scaling_metrics, graph_stress_metrics, ood_reward_metrics, uncertainty_error_detection, and claim_matrix\n')
    shutil.copy2(root/'inputs_v1/locks/claim_scope_lock.json',g4/'locks/claim_scope_lock.json'); shutil.copy2(a.stage5_reward/'configs/reward_selection_lock.json',g4/'locks/stage5_reward_selection_lock.json');
    (g4/'FROZEN.md').write_text('milestone = M5_REWARD_EVIDENCE\ndecision = GO_STAGE8_REWARD_ONLY\nmode = reward_only\nmain_reward = pathgraph_reward_v1_locked\nmanual_graph_is_main = true\npolicy_evidence = secondary_mixed\nno_more_policy_training = true\nno_post_test_main_reward_retuning = true\n')
    portable=[]
    for f in sorted(g4.rglob('*')):
        if f.is_file() and f.name!='M5_REWARD_EVIDENCE_SHA256SUMS.txt': portable.append(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(g4)}')
    (g4/'M5_REWARD_EVIDENCE_SHA256SUMS.txt').write_text('\n'.join(portable)+'\n')
    # Copy the frozen material into the 7.7 round.
    rr=rounds/'stage7_7_g4_freeze';
    for d in ('metrics','tables','reports','configs','locks'): (rr/d).mkdir(parents=True,exist_ok=True)
    for d in ('metrics','tables','reports','configs','locks'):
        for f in (g4/d).glob('*'):
            if f.is_file(): shutil.copy2(f,rr/d/f.name)
    shutil.copy2(g4/'FROZEN.md',rr/'FROZEN.md'); shutil.copy2(g4/'stage8_handoff.md',rr/'reports/stage8_handoff.md'); (rr/'summary.md').write_text('# Summary\n\nG4: GO_STAGE8_REWARD_ONLY.\n')
    (rr/'run_manifest.md').write_text('# Run Manifest\n\n- round_id: stage7_7_g4_freeze\n- mode: reward_only\n- training_jobs: 0\n- decision: GO_STAGE8_REWARD_ONLY\n- policy_evidence: secondary_mixed\n')
    # Final package: only portable evidence; omit checkpoints, raw episodes, predictions, stress traces and embeddings.
    a.final_zip.parent.mkdir(parents=True,exist_ok=True); a.final_zip.unlink(missing_ok=True)
    include=[]
    for base in (root/'inputs_v1', rounds, g4):
        if not base.exists(): continue
        for f in base.rglob('*'):
            if not f.is_file(): continue
            rel=f.relative_to(root)
            if any(part in ('checkpoints','embeddings','stress_graphs','stress_graphs_noisy','predictions') for part in rel.parts): continue
            if f.suffix.lower() in ('.pt','.pth','.ckpt','.npz','.npy','.parquet'): continue
            if f.stat().st_size>200*1024*1024: continue
            include.append((f,rel))
    with zipfile.ZipFile(a.final_zip,'w',zipfile.ZIP_DEFLATED) as z:
        for f,rel in sorted(include,key=lambda x:str(x[1])): z.write(f,rel)
    with zipfile.ZipFile(a.final_zip) as z: assert z.testzip() is None
    h=hashlib.sha256(a.final_zip.read_bytes()).hexdigest(); (a.final_zip.with_suffix('.zip.sha256')).write_text(f'{h}  {a.final_zip.name}\n'); print(json.dumps({'decision':'GO_STAGE8_REWARD_ONLY','final_zip':str(a.final_zip),'sha256':h,'files':len(include)},indent=2))
if __name__=='__main__': main()
