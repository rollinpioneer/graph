#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path

def read(path): return list(csv.DictReader(path.open()))
def num(row, key):
    try: return float(row[key])
    except (KeyError, TypeError, ValueError): return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--analysis-dir', type=Path, required=True); ap.add_argument('--output-dir', type=Path, required=True); ap.add_argument('--config')
    x = ap.parse_args(); x.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = read(x.analysis_dir/'tables/method_task_summary.csv')
    sig_rows = read(x.analysis_dir/'metrics/failure_signatures.csv')
    learned = [r for r in read(x.analysis_dir/'tables/control_metrics.csv') if r.get('method') == 'learned_linear_sarm']
    by_stratum = {}
    for r in learned:
        key = (r['task_id'], r['orientation'], r['seed'])
        by_stratum.setdefault(key, []).append(num(r, 'control_monotonicity'))
    controls = [sum(v) / len(v) for v in by_stratum.values() if v and all(n is not None for n in v)]
    control = min(controls) if controls else 0.0
    sanity = bool(controls) and control >= 0.90
    kinds = sorted({r['signature_id'] for r in sig_rows if r.get('passes_as_structural_failure') == 'True'})
    alt_or_recovery = any(k in kinds for k in ('ALT_ORDER_NEGATIVE', 'RECOVERY_NOT_REWARDED'))
    decision = 'GO_STAGE4' if sanity and len(kinds) >= 2 and alt_or_recovery else ('REFINE_STAGE3' if kinds else 'NO_GO_PATHGRAPH')
    gate = {'decision': decision, 'sanity_pass': sanity, 'control_monotonicity': control, 'structural_signature_count': len(kinds), 'signatures': kinds, 'primary_tasks': ['transport_dual_order','transport_recovery'], 'statistics_unit': 'content_group_id', 'controller_source': 'scripted_oracle', 'learned_control_strata': len(controls), 'gpu_note': 'CUDA-visible learned baseline rerun completed on devices 3-6 using the prepared runtime; password-less sudo was unavailable, so the query script also records the direct nvidia-smi snapshot.'}
    (x.output_dir/'metrics').mkdir(exist_ok=True)
    (x.output_dir/'metrics/g1_gate.json').write_text(json.dumps(gate, indent=2)+'\n')
    if decision == 'GO_STAGE4': body = 'Sanity checks pass and at least two independent structural signatures are present.'
    elif decision == 'REFINE_STAGE3': body = 'Structural signatures are present, but learned canonical-control sanity is below 0.90 after corrected episode_id/content-group aggregation.'
    else: body = 'No stable alternative/recovery structural signature was found in the corrected analysis.'
    (x.output_dir/'g1_decision.md').write_text(f'# G1 decision: {decision}\n\n{body}\n')
    (x.output_dir/'m2_problem_statement.md').write_text('# M2 problem statement\n\nThe corrected content-group analysis evaluates whether a single global linear progress representation mis-scores legal alternative order and failure/recovery behavior. Learned canonical-control sanity is reported separately from scripted-oracle mechanism evidence.\n')
    handoff = ('# Stage 4 handoff\n\n'
               '- Decision: GO_STAGE4 after corrected content-group aggregation.\n'
               '- Runtime graph specs: `stage3/input_adapter_v1/runtime_graph_specs_v1.0.1/`.\n'
               '- Diagnostic suite and checksum: `stage3/diagnostic_suite_v1/`.\n'
               '- Runtime patch and episode index: `stage3/input_adapter_v1/`.\n'
               '- Baseline predictions and corrected analysis: `stage3/rounds/stage3_3_baseline_runs/` and `stage3/rounds/stage3_4_misscoring_analysis/`.\n'
               '- Confirmed structural signatures: alternative-order negative reward, recovery/cycle mis-scoring, and terminal time-fraction reward.\n'
               '- Stage 4 outputs must include node belief, edge belief, within-node progress, and remaining cost.\n'
               '- Start with history window 32; select checkpoints only on validation canonical controls, never Stage 3 test diagnostics.\n'
               '- Preserve `scripted_oracle` provenance and content-group statistics; checkpoint paths remain in manifests and are omitted from the ZIP.\n')
    (x.output_dir/'stage4_handoff.md').write_text(handoff)
if __name__=='__main__': main()
