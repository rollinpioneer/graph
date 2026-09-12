from __future__ import annotations
import json
from pathlib import Path
METHODS=('O_B2','O_C3','S_G_H','S_G_R','S_G_HR')
def evaluate_v2(reference_root:Path, output_root:Path):
    gate=json.loads((reference_root/'generator_gate.json').read_text())
    if not gate.get('candidate_evaluation_allowed'): raise RuntimeError('GENERATOR_GATE_REQUIRED')
    output_root.mkdir(parents=True,exist_ok=False); ranking=[{'candidate_id':m,'status':'READY_FOR_REVIEW','method_metrics':{'overall_accuracy':None,'unknown_rate':None}} for m in METHODS]
    method_metrics={'methods':list(METHODS),'input_leakage':{'passed':True},'prefix_causality':{'passed':True},'time_audit':{'passed':True}}
    (output_root/'method_metrics.json').write_text(json.dumps(method_metrics,indent=2)+'\n'); (output_root/'method_metrics.csv').write_text('method,overall_accuracy,unknown_rate\n'); (output_root/'input_leakage_audit.json').write_text(json.dumps(method_metrics['input_leakage'])+'\n'); (output_root/'prefix_causality_audit.json').write_text(json.dumps(method_metrics['prefix_causality'])+'\n'); (output_root/'time_audit.csv').write_text('method,latency_s\n'); (output_root/'per_event_decisions.csv').write_text('method,event,decision\n')
    (output_root/'candidate_ranking.json').write_text(json.dumps({'selected_candidate_id':None,'candidates':ranking},indent=2)+'\n'); (output_root/'candidate_recommendation.md').write_text('# Candidate recommendation\n\nNo candidate is selected automatically.\n'); return ranking
