"""Make early fall censoring explicit; zero candidates with no eligible windows is limited evidence."""
import os,pathlib,json,numpy as np
R=pathlib.Path(os.environ['R']);records=json.loads((R/'results/P2_mechanism_candidates.json').read_text());out=[]
for f in sorted((R/'runs').glob('normal*/data/completed.json')):
 name=f.parent.parent.name
 with np.load(f.parent/'trajectory.npz') as z:
  fall=(z['states'][:,:,2]<.975).any(1);first=np.flatnonzero(fall);first=int(first[0]) if len(first) else None
 out.append({'run':name,'first_any_below_fall_threshold_state':first,'first_confirmed5state_fall':json.loads((f.parent/'events.json').read_text())['first_fall'],'eligible_pending_near_prefall_20step_windows':sum(q['run']==name for q in records),'statement':'early fall leaves no eligible20stepcandidatewindow; absence of hit cannot reject command mechanism' if first is not None and first<=20 else 'only observed eligible windows support mechanism assessment; population causality not estimated'})
(R/'results/normal_review_opportunity_and_censoring.json').write_text(json.dumps(out,indent=2));print('CENSORING_FACTS',out,flush=True)
