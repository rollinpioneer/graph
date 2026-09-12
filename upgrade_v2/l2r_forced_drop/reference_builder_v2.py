from __future__ import annotations
import csv,json
from pathlib import Path
def build_reference_v2(development_root:Path, output_root:Path):
    output_root.mkdir(parents=True,exist_ok=False); rows=[]
    for status in sorted(development_root.rglob('status.json')):
        row=json.loads(status.read_text()); rows.append({'family_id':row['family_id'],'case_id':row['case_id'],'trace_complete':row['trace_complete'],'numeric_health_pass':row['numeric_health_pass']})
    physical_loss_confirmed=sum(bool(r.get('physical_loss_confirmed',False)) for r in rows); pre_hold_verified=sum(bool(r.get('pre_hold_verified',False)) for r in rows); commanded_release=sum(bool(r.get('commanded_release',False)) for r in rows)
    gate={'schema':'l2rar2_r16_generator_gate_v2','generator_gate':True,'trace_complete':sum(r['trace_complete'] for r in rows)==32,'numeric_health':sum(r['numeric_health_pass'] for r in rows)==32,'physical_loss_confirmed':physical_loss_confirmed,'pre_hold_verified':pre_hold_verified,'COMMANDED_RELEASE':commanded_release,'rollouts':len(rows),'candidate_evaluation_allowed':len(rows)==32 and all(r['trace_complete'] and r['numeric_health_pass'] for r in rows)}
    (output_root/'physical_reference_events.csv').write_text('family_id,case_id,trace_complete,numeric_health_pass\n'+'\n'.join(','.join(str(r[k]) for k in ('family_id','case_id','trace_complete','numeric_health_pass')) for r in rows)+'\n'); (output_root/'generator_gate.json').write_text(json.dumps(gate,indent=2)+'\n'); return gate
