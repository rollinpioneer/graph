import json,time
from pathlib import Path
R=Path.home()/'work/grasp/eg_tf32_residual_p2_v1_20261021'
receipts=[json.loads(p.read_text()) for p in (R/'results').glob('eval_*.json') if not p.name.endswith('_pilot.json')]
controllers={}
for n in ['161','245','248']:
 p=R/f'results/controller_eval_{n}.json'
 if p.exists():controllers[n]=json.loads(p.read_text())['status']
print(json.dumps(dict(completed_shards=len(receipts),frames_in_completed_shards=sum(len(v['frames']) for v in receipts),written_frames=len(list((R/'evaluation/development').glob('*/*.json'))),total_shards=96,total_frames=3072,controllers=controllers,new_labels_completed=sum(v['new_evaluation_labels'] for v in receipts))))
for p in sorted((R/'logs').glob('eval_*.log')):
 s=p.read_text(errors='replace')
 if 'Traceback' in s:print(p.name,s[-1800:])
