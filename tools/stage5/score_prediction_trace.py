#!/usr/bin/env python3
import argparse,gzip,json,yaml,math
from pathlib import Path
from .lib.reward_engine import PathGraphRewardEngine
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--reward-config',required=True);p.add_argument('--lambda-value',type=float,required=True);p.add_argument('--eta-value',type=float,required=True);p.add_argument('--beta-value',type=float,required=True);p.add_argument('--confidence',type=float,default=None);p.add_argument('--output',required=True);p.add_argument('--summary',required=True);a=p.parse_args(); rows=[json.loads(x) for x in gzip.open(a.predictions,'rt')]; cfg=yaml.safe_load(open(a.reward_config)); eng=PathGraphRewardEngine(cfg,a.lambda_value,a.eta_value,a.beta_value,a.confidence); states={}; out=[]
 for i,q in enumerate(rows):
  k=(q['task_id'],q['episode_id']); st=states.setdefault(k,eng.new_episode(*k,len(q.get('per_seed_remaining_cost',[0,0,0]))));
  if i and rows[i-1]['episode_id']==q['episode_id']:
   rr=eng.step(rows[i-1],q,st); d=rr.__dict__.copy(); d.update({'episode_id':q['episode_id'],'content_group_id':q['content_group_id'],'task_id':q['task_id'],'step':q['step']}); out.append(d)
 with gzip.open(a.output,'wt') as f:
  for d in out:
   if not all(math.isfinite(float(v)) for v in [d['reward_mu'],d['reward_lcb'],d['weight_positive']]): raise SystemExit('nonfinite reward')
   f.write(json.dumps(d,separators=(',',':'))+'\n')
 Path(a.summary).parent.mkdir(parents=True,exist_ok=True); Path(a.summary).write_text(json.dumps({'rows':len(out),'episodes':len(states),'nonzero_rate':sum(d['weight_positive']>0 for d in out)/max(1,len(out)),'statistics_unit':'content_group_id'},indent=2))
if __name__=='__main__': main()
