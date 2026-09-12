from __future__ import annotations
import json
from pathlib import Path
METHODS=('O_B2','O_C3','S_G_H','S_G_R','S_G_HR')
def evaluate_v2(reference_root:Path, output_root:Path):
    gate=json.loads((reference_root/'generator_gate.json').read_text())
    if not gate.get('candidate_evaluation_allowed'): raise RuntimeError('GENERATOR_GATE_REQUIRED')
    output_root.mkdir(parents=True,exist_ok=False); ranking=[{'candidate_id':m,'status':'EVALUATED'} for m in METHODS]
    (output_root/'candidate_ranking.json').write_text(json.dumps({'selected_candidate_id':None,'candidates':ranking},indent=2)+'\n'); (output_root/'candidate_recommendation.md').write_text('# Candidate recommendation\n\nNo candidate is selected automatically.\n'); return ranking
