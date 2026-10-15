#!/usr/bin/env python3
import argparse, glob, json, csv
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--seed-metrics',nargs='+',required=True)
    p.add_argument('--bundle-verification',required=True); p.add_argument('--output-json',required=True); p.add_argument('--output-md',required=True); a=p.parse_args()
    cfg=json.load(open(a.config)) if str(a.config).endswith('.json') else None
    if cfg is None:
        import yaml; cfg=yaml.safe_load(open(a.config))
    gate=cfg['model_gate']; bundle=json.load(open(a.bundle_verification)); metrics=[json.load(open(x)) for pat in a.seed_metrics for x in glob.glob(pat)]
    metrics=sorted(metrics,key=lambda d:d.get('prediction_files',[''])[0]); checks=[]
    thresholds={'node_macro_f1':('min',gate['node_macro_f1_min']),'edge_type_macro_f1_non_none':('min',gate['edge_type_macro_f1_non_none_min']),'phi_mae':('max',gate['phi_mae_max']),'phi_spearman':('min',gate['phi_spearman_min']),'cost_mae':('max',gate['cost_mae_max']),'cost_spearman':('min',gate['cost_spearman_min']),'cost_pair_accuracy':('min',gate['cost_pair_accuracy_min']),'failure_cost_increase_rate':('min',gate['failure_cost_increase_rate_min']),'recovery_cost_decrease_rate':('min',gate['recovery_cost_decrease_rate_min'])}
    for d in metrics:
        row={'prediction':d.get('prediction_files',[''])[0],'checks':{}}
        for k,(op,t) in thresholds.items():
            v=d.get(k); row['checks'][k]=bool(v is not None and (v>=t if op=='min' else v<=t))
        row['passed']=all(row['checks'].values()); checks.append(row)
    passed=sum(x['passed'] for x in checks)
    bundle_ok=all(bundle.get(k) for k in ('all_checkpoints_exist','all_hashes_match','all_loadable','all_outputs_finite','input_response_nonconstant'))
    # A truthful recompute is explicitly required; no fixed Stage 4 metrics are accepted.
    ready=bool(bundle_ok and passed>=int(gate.get('required_seed_passes',2)) and all(d.get('metric_version','').startswith('stage5-real') for d in metrics))
    decision='REAL_MODEL_READY' if ready else 'REFINE_STAGE4_MINIMAL'
    failed=[]
    for x in checks:
        failed.extend(k for k,v in x['checks'].items() if not v)
    out={'decision':decision,'bundle_ok':bundle_ok,'seed_count':len(checks),'seed_pass_count':passed,'required_seed_passes':gate.get('required_seed_passes',2),'seed_checks':checks,'failed_metrics':sorted(set(failed)),'selection_allowed':ready,'statistics_unit':'content_group_id'}
    Path(a.output_json).parent.mkdir(parents=True,exist_ok=True); Path(a.output_json).write_text(json.dumps(out,indent=2))
    lines=[f'# Stage 5 Entry Decision', '', f'- `STAGE5_ENTRY_DECISION = {decision}`', f'- checkpoint verification: `{bundle_ok}`', f'- seed passes: `{passed}/{len(checks)}` (required `{gate.get("required_seed_passes",2)}`)', '- metrics source: real checkpoint prediction files only', '- test predictions are reporting-only and were not used for this decision']
    if not ready:
        lines += ['', '## Minimal refinement', '', 'Retrain only the remaining-cost/failure-recovery calibration head using the existing Stage 4 configuration, preserving the encoder, graph specification, split, and seed set. Reward parameter search is prohibited until a subsequent real recompute emits `REAL_MODEL_READY`.', '', '- failed checks: `'+', '.join(sorted(set(failed)))+'`']
        for d in metrics:
            lines.append('- observed: failure direction `%.3f`, recovery direction `%.3f`' % (d.get('failure_cost_increase_rate',float('nan')), d.get('recovery_cost_decrease_rate',float('nan'))))
    Path(a.output_md).parent.mkdir(parents=True,exist_ok=True); Path(a.output_md).write_text('\n'.join(lines)+'\n')
    print(decision)
if __name__=='__main__': main()
