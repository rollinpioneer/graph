import json
from pathlib import Path

r = Path.home() / 'work/grasp/eg_causal_mechanism_v2_20261023/r14'
branches = [json.loads(p.read_text()) for p in (r/'receipts').glob('src_*/*.json')]
journals = [json.loads(p.read_text()) for p in (r/'journals').glob('*.json')]
finished = {(x['source_scene'], x['arm']) for x in branches}
steps = sum(x['completed_steps'] for x in branches) + sum(x['completed_adam_steps'] for x in journals if (x['source_scene'], x['arm']) not in finished)
evaluations = [json.loads(p.read_text()) for p in (r/'evaluation').glob('src_*/**/train_*_0128.json')]
pre = json.loads((r/'evaluation/preflight.json').read_text())
out = {'completed_sources': [p.stem for p in (r/'receipts').glob('src_*_complete.json')],
       'completed_branches': len(branches), 'Adam_steps': steps,
       'predictions': len(list((r/'predictions').glob('src_*/**/*.npz'))),
       'evaluated_jobs': len(evaluations), 'analytic_calls': sum(x['calls'] for x in evaluations) + pre.get('calls', 8),
       'unknowns': sum(x['unknowns'] for x in evaluations)}
trace = r/'receipts/train_0064_native_vs_fixed_trace.json'
if trace.exists():
    z = json.loads(trace.read_text())
    out['scene64_trace'] = {k: z[k] for k in ['native_to_native_logits', 'native_to_fixed_logits', 'native_to_shared_backbone_logits', 'native_repeat_pass', 'fixed_path_pass']}
print(json.dumps(out))
