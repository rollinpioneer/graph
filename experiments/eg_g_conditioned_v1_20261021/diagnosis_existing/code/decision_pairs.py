import json,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[2];out=R/'diagnosis_existing/results'
agg={k:[] for k in ['success_to_failure','failure_to_success']}
for p in sorted((R/'evaluation/development').glob('*/*.json')):
 rows={v['configuration']:v for v in json.loads(p.read_text())}; pairs=[]
 for seed in [20261018,20261019,20261020]:
  g=rows[f'G_{seed}_CD'];f=rows[f'FiLM_{seed}_CD']
  if g['top1_valid']!=f['top1_valid']:pairs.append((seed,g,f,'success_to_failure' if g['top1_valid'] else 'failure_to_success'))
 if not pairs:continue
 z=np.load(p.with_suffix('.npz'))
 ep=R.parent/f"eg_sub25_b_f05_repl_v1_20261019/B/exports/subset/scene_{int(p.parent.name):04d}/{int(p.stem):04d}.npz"
 e=np.load(ep)['pred'][:,0]
 for seed,g,f,tag in pairs:
  i=g['top_candidate'];j=f['top_candidate'];qg=z[f'Q_G_{seed}'];qf=z[f'Q_FiLM_{seed}']
  agg[tag].append(dict(scene=g['scene'],seed=seed,frame=g['frame'],G_q_prefers_new=bool(qg[j]>qg[i]),FiLM_q_prefers_new=bool(qf[j]>qf[i]),EG_score_prefers_new=bool(e[j]>e[i]),utility_new_minus_old=f['top1_utility']-g['top1_utility'],FiLM_q_delta=float(qf[j]-qf[i]),G_q_delta=float(qg[j]-qg[i])))
outdata={k:dict(n=len(v),EG_prefers_new=sum(x['EG_score_prefers_new'] for x in v),G_q_prefers_new=sum(x['G_q_prefers_new'] for x in v),FiLM_q_prefers_new=sum(x['FiLM_q_prefers_new'] for x in v),new_q_only_reversal=sum(x['FiLM_q_prefers_new'] and not x['G_q_prefers_new'] for x in v),mean_utility_change=float(np.mean([x['utility_new_minus_old'] for x in v]))) for k,v in agg.items()}
(out/'film_decision_pairs.json').write_text(json.dumps(dict(summary=outdata,records=agg),indent=2));print(json.dumps(outdata))

