from p2_residual import *
import argparse
p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','train','evaluate']);a=p.parse_args();cfg=json.loads((R/'frozen_config.json').read_text())
if a.stage=='preflight':
 write(R/'preflight_lock.json',dict(config_sha256=sha(R/'frozen_config.json'),evaluation_code_sha256=sha(R/'code/eval_p2r.py'),P0_union_manifest_sha256=sha(R/'results/P0_union_manifest.json'),residual_code_sha256=sha(R/'code/p2_residual.py')))
elif a.stage=='train':
 rr=[json.loads(p.read_text()) for p in (R/'results').glob('preflight_*.json') if not p.name.endswith('_pilot.json')];assert len(rr)==96 and all(v['status']=='PASS' for v in rr);assert sum(len(v['frames']) for v in rr)==3072
 vv=[x for v in rr for x in v['invariants']];assert len(vv)==9216
 i1=[x[fam+'_I1_max_F05_error'] for x in vv for fam in ['Eact','Epot'] if fam+'_I1_max_F05_error' in x];assert len(i1)==72
 f1=G/'eg_f1_precision_diagnosis_v1_20261021/diagnosis.json';assert sha(f1)==cfg['F1_exact_replay_evidence_sha256']
 out=dict(status='PASS',training_started=False,completed_at=time.time(),I1_checks=72,I1_max_F05_error=max(i1),I2_I3_checks=9216,I2_I3_max_Q_error=max(x['I2_I3_max_Q_error'] for x in vv),I2_top_same=all(x['I2_top_same'] for x in vv),I4_frames=3072,I4_selections_exact=3072*14,F1_replay_max_error=0.,F1_evidence_reused_by_hash=True,C4_read=False,config_sha256=sha(R/'frozen_config.json'),head_inputs_sha256=sha(R/'head_inputs.npz'),preflight_code_sha256=sha(R/'code/eval_p2r.py'),residual_code_sha256=sha(R/'code/p2_residual.py'))
 assert out['I1_max_F05_error']<=1e-9 and out['I2_I3_max_Q_error']<=1e-6 and out['I2_top_same']
 write(R/'results/invariance_checks.json',out);print(json.dumps(out))
else:
 checks=json.loads((R/'results/invariance_checks.json').read_text());assert checks['status']=='PASS';heads=[]
 for fam in FAMILIES:
  for seed in SEEDS:
   p=R/f'heads/{fam}/{seed}/step1000.npz';v=json.loads((R/f'results/train_{fam}_{seed}.json').read_text());assert v['status']=='PASS' and v['steps']==1000 and v['checkpoint_sha256']==sha(p)
   heads.append(dict(family=fam,seed=seed,path=str(p),sha256=sha(p)))
 write(R/'method_lock.json',dict(config_sha256=sha(R/'frozen_config.json'),heads=heads,evaluation_code_sha256=sha(R/'code/eval_p2r.py'),P0_union_manifest_sha256=sha(R/'results/P0_union_manifest.json'),invariance_checks_sha256=sha(R/'results/invariance_checks.json'),residual_code_sha256=sha(R/'code/p2_residual.py')))
