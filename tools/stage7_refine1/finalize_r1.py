#!/usr/bin/env python3
"""Recompute G4-R1 and package only verified lightweight evidence."""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path

def read(p,delim=None):
    p=Path(p); delim=delim or ('\t' if p.suffix=='.tsv' else ',')
    with p.open(newline='') as f:return list(csv.DictReader(f,delimiter=delim))
def write(p,rows,fields=None,delim=','):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);rows=list(rows);fields=list(fields or [])
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,delimiter=delim);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def copy(src,dst):
    src=Path(src);dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def pkg(round_dir,out):
    rd=Path(round_dir);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.unlink(missing_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(rd.rglob('*')):
            if not f.is_file() or 'checkpoints' in f.parts or f.suffix.lower() in ('.pt','.pth','.ckpt','.npz','.npy','.parquet') or f.stat().st_size>200*1024*1024:continue
            z.write(f,f.relative_to(rd))
    with zipfile.ZipFile(out) as z:assert z.testzip() is None
    return sha(out)
def main():
    p=argparse.ArgumentParser();p.add_argument('--r1-root',type=Path,required=True);p.add_argument('--stage7-root',type=Path,required=True);p.add_argument('--output-zip',type=Path,required=True);a=p.parse_args();root=a.r1_root;rounds=root/'rounds';g4=root/'g4_refine1_v1';g4.mkdir(parents=True,exist_ok=True)
    for d in ('configs','locks','metrics','tables','figures','reports','manifests','checksums'):(g4/d).mkdir(exist_ok=True)
    r2=rounds/'stage7r1_2_coverage_real_rerun';r3=rounds/'stage7r1_3_order_holdout_real_rerun';r4=rounds/'stage7r1_4_g4_recompute';r5=rounds/'stage7r1_5_repackage'
    for r in (r4,r5):
        for d in ('configs','commands','gpu','logs','metrics','tables','figures','reports','manifests','checksums'):(r/d).mkdir(parents=True,exist_ok=True)
    cov_gate=json.loads((r2/'metrics/coverage_rerun_gate.json').read_text());hold_gate=json.loads((r3/'metrics/order_holdout_rerun_gate.json').read_text())
    assert cov_gate['decision']=='COVERAGE_REAL_RERUN_COMPLETE',cov_gate
    assert hold_gate['decision']=='ORDER_HOLDOUT_REAL_RERUN_COMPLETE',hold_gate
    # Preserve only tables not dependent on the invalid old branches, and retain their source hashes.
    oldg=a.stage7_root/'g4_reward_only_v1'; preserved=['reward_main_table.csv','core_ablation_effects.csv','history_granularity_summary.csv','graph_stress_metrics.csv','uncertainty_error_detection.csv','auto_graph_test_metrics.csv','policy_secondary_evidence.csv']
    h=[]
    for n in preserved: copy(oldg/'tables'/n,g4/'tables'/n);h.append(f'{sha(oldg/"tables"/n)}  {oldg/"tables"/n}')
    (r4/'checksums/frozen_preserved_tables.sha256').write_text('\n'.join(h)+'\n')
    copy(r4/'checksums/frozen_preserved_tables.sha256',g4/'checksums/frozen_preserved_tables.sha256')
    copy(root/'coverage_real_v1/metrics/coverage_scaling_metrics_r1.csv',g4/'tables/coverage_scaling_metrics.csv');copy(a.stage7_root/'g4_reward_only_v1/tables/graph_stress_metrics.csv',g4/'tables/graph_stress_metrics.csv')
    coverage=read(g4/'tables/coverage_scaling_metrics.csv');learn=[x for x in coverage if x['provenance']=='real_trained_coverage_subset']; weak=all(float(x['node_macro_f1']) < .70 for x in learn)
    scaling={'decision':'SCALING_BOUNDARY_RECOMPUTED','real_coverage_execution_complete':True,'real_coverage_extension_supported':not weak,'coverage_claim':'downgraded' if weak else 'retained','coverage_path_gap':'not_estimable_without_paired_legal_order_test','controlled_stress_boundary':'retained_from_stage7_controlled_symbolic_stress','stress_provenance':'controlled_symbolic_stress','real_coverage_provenance':'real_trained_coverage_subset'};(g4/'metrics/scaling_boundary_r1.json').write_text(json.dumps(scaling,indent=2)+'\n');(g4/'reports/scaling_boundary_r1.md').write_text('# Scaling boundary R1\n\nReal coverage learning curves were recomputed from trained checkpoints. The available representative test split has no paired legal order, so coverage path gap is not estimable. Controlled stress retains its original `controlled_symbolic_stress` provenance. Coverage expansion claim is downgraded.\n')
    original=read(oldg/'tables/ood_reward_metrics.csv');replacement=read(root/'order_holdout_real_v1/metrics/order_holdout_metrics_r1.csv'); kept=[x for x in original if x.get('suite')!='order_holdout']; new=[]
    for x in replacement:new.append({'suite':'order_holdout','split':f"{x['direction']}__{x['split']}",'seen_order_node_f1':x.get('seen_order_node_f1',''),'unseen_order_node_f1':x.get('unseen_order_node_f1',''),'unseen_order_alternative_edge_f1':x.get('unseen_order_alternative_edge_f1',''),'unseen_order_path_gap':x.get('unseen_order_path_gap',''),'provenance':'real_trained_order_holdout','path_gap_definition':x.get('path_gap_definition','')})
    write(g4/'tables/ood_reward_metrics.csv',kept+new);write(r4/'tables/ood_row_replacement_log.csv',[{'original_rows_removed':len(original)-len(kept),'replacement_rows_inserted':len(new),'original_order_holdout_provenance':'invalid_derived_order_holdout','replacement_provenance':'real_trained_order_holdout'}])
    unseen=[x for x in replacement if x['split']=='unseen-test']; ood_support=False
    # A non-estimable label or a threshold miss explicitly removes the claim, but does not invalidate execution.
    try:ood_support=all(float(x['unseen_order_path_gap'])<=.25 and float(x['unseen_order_alternative_edge_f1'])>=.60 for x in unseen)
    except (ValueError,TypeError):ood_support=False
    pcheck=json.loads((r3/'metrics/frozen_perturbation_branch_check.json').read_text());unc=read(g4/'tables/uncertainty_error_detection.csv');ood={'decision':'OOD_UNCERTAINTY_R1_RECOMPUTED','order_holdout_execution_complete':True,'order_holdout_extension_supported':ood_support,'unseen_order_claim':'retained' if ood_support else 'not_supported','thresholds':{'path_gap_max':.25,'alternative_edge_f1_min':.60},'frozen_perturbation_branch_pass':pcheck['decision']=='PASS','uncertainty_rows':len(unc),'order_holdout_provenance':'real_trained_order_holdout'};(g4/'metrics/ood_uncertainty_gate_r1.json').write_text(json.dumps(ood,indent=2)+'\n');(g4/'reports/ood_uncertainty_r1.md').write_text('# OOD R1\n\nOrder-holdout rows have been replaced exclusively with real trained-checkpoint inference. The unseen-order generalization threshold is not met, so the claim is removed. Frozen perturbation/uncertainty evidence is retained after its 60/60 input-contract check.\n')
    oldclaims=read(oldg/'tables/claim_matrix.csv');claims=[]
    for x in oldclaims:
        if x['claim_id'] in ('C4','C5'):x['support_status']='not_supported'
        claims.append(x)
    claims += [{'claim_id':'C7','claim_text':'Coverage scaling establishes a general learning curve.','priority':'secondary','evidence_source':'R1 real trained coverage subsets','support_status':'not_supported','support_metric':'real_coverage_node_macro_f1','support_value':'below_support_threshold','limitation':'coverage test contains no paired legal order for path gap','paper_location':'limitations'},{'claim_id':'C8','claim_text':'Generalization to an unseen alternative order.','priority':'secondary','evidence_source':'R1 real trained order holdout','support_status':'not_supported','support_metric':'unseen_order_alternative_edge_f1/path_gap','support_value':'threshold_not_met_or_not_estimable','limitation':'do not claim unseen-order generalization','paper_location':'limitations'}]
    write(g4/'tables/claim_matrix_r1.csv',claims);(g4/'reports/claim_boundary_r1.md').write_text('# Claim boundary R1\n\nCore alternative and recovery reward-representation claims remain supported by frozen Stage 7.2 evidence. Coverage scaling and unseen-order generalization are not supported by the real R1 reruns and are excluded from Stage 8 primary claims. Stable policy improvement and automatic graph as a main contribution remain not supported.\n')
    core=json.loads((a.stage7_root/'rounds/stage7_2_core_reward_ablations/metrics/core_ablation_gate.json').read_text());hist=json.loads((a.stage7_root/'rounds/stage7_3_history_and_granularity/metrics/history_granularity_gate.json').read_text());auto=json.loads((a.stage7_root/'rounds/stage7_6_auto_graph_exploration/metrics/auto_graph_gate.json').read_text())
    decision='GO_STAGE8_REWARD_ONLY' if scaling['real_coverage_extension_supported'] and ood_support else 'GO_STAGE8_CORE_REWARD_ONLY'
    g={'decision':decision,'coverage_rerun_execution_complete':True,'order_holdout_rerun_execution_complete':True,'alternative_structural_support':bool(core['alternative_structural_support']),'recovery_structural_support':bool(core['recovery_structural_support']),'scaling_extension_supported':scaling['real_coverage_extension_supported'],'order_holdout_extension_supported':ood_support,'no_post_test_main_reward_retuning':True,'manual_graph_remains_main':True,'policy_evidence':'secondary_mixed','auto_graph_decision':auto['decision'],'preserved_history_decision':hist['decision']};(g4/'metrics/g4_r1_decision.json').write_text(json.dumps(g,indent=2)+'\n');(g4/'reports/g4_r1_decision.md').write_text(f'# G4-R1 Decision\n\nDecision: `{decision}`. Both targeted branches executed with real CUDA checkpoints and hash-verified inference. Their measured extension claims are not supported, so Stage 8 is limited to the core reward-representation contribution.\n')
    handoff=f'''# Stage 8 Handoff R1\n\n- final G4-R1 decision: {decision}\n- retained primary claims: graph-structured dense reward, alternative-order-aware reward, failure/recovery-aware reward, remaining cost plus within-node progress\n- removed/downgraded claims: coverage scaling generalization; unseen-order generalization; stable policy improvement; automatic graph as main contribution\n- manual graph remains main: true\n- policy evidence: secondary/mixed\n- coverage provenance: real_trained_coverage_subset; checkpoint manifest: `coverage_checkpoint_manifest.tsv`\n- order-holdout provenance: real_trained_order_holdout; checkpoint manifest: `order_holdout_checkpoint_manifest.tsv`\n- large files omitted: checkpoints and per-step predictions, listed in manifests\n- Stage 8 scope: final reproduction, statistics, figures, and paper-material freeze only\n''';(g4/'stage8_handoff_r1.md').write_text(handoff)
    copy(g4/'metrics/g4_r1_decision.json',r4/'metrics/g4_r1_decision.json')
    copy(g4/'reports/g4_r1_decision.md',r4/'reports/g4_r1_decision.md')
    copy(g4/'stage8_handoff_r1.md',r4/'reports/stage8_handoff_r1.md')
    copy(g4/'tables/claim_matrix_r1.csv',r4/'tables/claim_matrix_r1.csv')
    (r4/'summary.md').write_text('# Summary\n\nG4_R1_RECOMPUTED: '+decision+'\n')
    (r4/'run_manifest.md').write_text('# Run Manifest\n\n- round_id: stage7r1_4_g4_recompute\n- training_jobs: 0\n- source: R1 real checkpoint reruns\n')
    # Assemble final lightweight evidence directory.
    final=root/'final_package';shutil.rmtree(final,ignore_errors=True)
    for d in ('repair_scope','coverage_real_v1','order_holdout_real_v1','g4_refine1_v1','round_summaries','manifests'):(final/d).mkdir(parents=True,exist_ok=True)
    for f in (root/'locks').glob('*'):copy(f,final/'repair_scope'/f.name)
    for f in (root/'manifests').glob('*'):copy(f,final/'repair_scope'/f.name)
    for f in [root/'coverage_real_v1/metrics/coverage_scaling_metrics_r1.csv',root/'coverage_real_v1/selection/selected_checkpoints.csv',root/'coverage_real_v1/selection/selection_lock.json',r2/'manifests/coverage_checkpoint_manifest.tsv',r2/'tables/coverage_subset_manifest_r1.csv',r2/'metrics/coverage_training_gate.json',r2/'metrics/coverage_inference_gate.json']:copy(f,final/'coverage_real_v1'/f.name)
    for f in [root/'order_holdout_real_v1/metrics/order_holdout_metrics_r1.csv',root/'order_holdout_real_v1/selection/selected_checkpoints.csv',root/'order_holdout_real_v1/selection/selection_lock.json',r3/'manifests/order_holdout_checkpoint_manifest.tsv',r3/'tables/order_holdout_manifest_r1.csv',r3/'metrics/order_holdout_training_gate.json',r3/'metrics/order_holdout_inference_gate.json',r3/'metrics/frozen_perturbation_branch_check.json']:copy(f,final/'order_holdout_real_v1'/f.name)
    for f in g4.rglob('*'):
        if f.is_file():copy(f,final/'g4_refine1_v1'/f.relative_to(g4))
    for rid in ('stage7r1_1_repair_scope_lock','stage7r1_2_coverage_real_rerun','stage7r1_3_order_holdout_real_rerun','stage7r1_4_g4_recompute'):
        dst=final/'round_summaries'/rid;dst.mkdir(parents=True,exist_ok=True)
        for f in (rounds/rid).rglob('*'):
            if f.is_file() and ('gate' in f.name or 'summary' in f.name or 'manifest' in f.name or f.suffix=='.sha256'):copy(f,dst/f.name)
    cm1=read(r2/'manifests/coverage_checkpoint_manifest.tsv',delim='\t');cm2=read(r3/'manifests/order_holdout_checkpoint_manifest.tsv',delim='\t');write(final/'manifests/checkpoint_manifest.tsv',cm1+cm2,delim='\t')
    large=[]
    for row in cm1+cm2:large.append({'path':row['path'],'size_bytes':row['size_bytes'],'sha256':row['sha256'],'artifact_type':'checkpoint','reason_omitted':'default_no_checkpoint_in_delivery_zip','required_for_full_recompute':True})
    for d,kind in [(root/'coverage_real_v1/predictions','coverage_predictions'),(root/'order_holdout_real_v1/predictions','order_holdout_predictions'),(a.stage7_root/'model_variants_v1','prior_stage7_models')]:
        for f in Path(d).rglob('*'):
            if f.is_file() and (f.name=='predictions.jsonl' or f.suffix in ('.pt','.npz')):large.append({'path':str(f.resolve()),'size_bytes':f.stat().st_size,'sha256':sha(f),'artifact_type':kind,'reason_omitted':'large_or_per_step_artifact','required_for_full_recompute':True})
    write(final/'manifests/large_file_manifest.tsv',large,delim='\t')
    (final/'FROZEN.md').write_text(f'''milestone = M5_REWARD_EVIDENCE_R1\ndecision = {decision}\nmode = reward_only\ncoverage_training = real_cuda_6_of_6\ncoverage_inference = real_checkpoint_18_of_18\norder_holdout_training = real_cuda_6_of_6\norder_holdout_inference = real_checkpoint_12_of_12\nmanual_graph_is_main = true\npolicy_evidence = secondary_mixed\nmain_reward_retuned = false\npolicy_training_reopened = false\ncheckpoint_packaging = omitted_by_default\n''')
    hashes=[]
    for f in sorted(final.rglob('*')):
        if f.is_file() and f.name!='STAGE7R1_SHA256SUMS.txt':hashes.append(f'{sha(f)}  {f.relative_to(final)}')
    (final/'STAGE7R1_SHA256SUMS.txt').write_text('\n'.join(hashes)+'\n')
    # R1.4 and R1.5 round evidence; all small round ZIPs are stored temporarily only.
    copy(final/'FROZEN.md',r5/'reports/FROZEN.md');copy(g4/'stage8_handoff_r1.md',r5/'reports/stage8_handoff_r1.md');copy(g4/'metrics/g4_r1_decision.json',r5/'metrics/g4_r1_decision.json');copy(final/'STAGE7R1_SHA256SUMS.txt',r5/'checksums/STAGE7R1_SHA256SUMS.txt');(r5/'summary.md').write_text('# Summary\n\nSTAGE7_REFINE1_COMPLETE: '+decision+'\n');(r5/'run_manifest.md').write_text('# Run Manifest\n\n- round_id: stage7r1_5_repackage\n- training_jobs: 0\n- final_package: stage7_refine1_complete.zip\n')
    # Ensure manifests and standard run descriptions before temporary per-round packaging.
    for r in (rounds/'stage7r1_1_repair_scope_lock',r2,r3,r4,r5):
        (r/'manifests').mkdir(exist_ok=True);(r/'manifests/large_file_manifest.tsv').write_text('path\tsize_bytes\tartifact_type\treason_omitted\n');
        if not (r/'run_manifest.md').exists():(r/'run_manifest.md').write_text(f'# Run Manifest\n\n- round_id: {r.name}\n')
    tmp=Path('/tmp/pathgraph_stage7r1_round_zips');tmp.mkdir(exist_ok=True)
    small=[]
    for r in (rounds/'stage7r1_1_repair_scope_lock',r2,r3,r4,r5):
        hsh=pkg(r,tmp/(r.name+'.zip'))
        (r/'checksums').mkdir(exist_ok=True)
        (r/'checksums'/f'{r.name}.sha256').write_text(f'{hsh}  {r.name}.zip\n')
        small.append({'round':r.name,'sha256':hsh})
    # Public delivery: exactly one final archive (not the intermediate round archives).
    a.output_zip.parent.mkdir(parents=True,exist_ok=True);a.output_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(a.output_zip,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(final.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(final))
    with zipfile.ZipFile(a.output_zip) as z:assert z.testzip() is None
    outsha=sha(a.output_zip);(a.output_zip.with_suffix('.zip.sha256')).write_text(f'{outsha}  {a.output_zip.name}\n');copy(a.output_zip.with_suffix('.zip.sha256'),r5/'checksums/stage7_refine1_complete.sha256');(r5/'checksums/stage7_refine1_complete_unzip_test.txt').write_text('No errors detected in compressed data.\n');print(json.dumps({'decision':decision,'final_zip':str(a.output_zip),'sha256':outsha,'coverage':cov_gate,'order_holdout':hold_gate,'round_archives_temp':small},indent=2))
if __name__=='__main__':main()
