#!/usr/bin/env python3
"""Score frozen diagnostic episodes with outcome-blind linear baselines."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import yaml
def rowsj(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def nodeat(a,t):
 for x in a['node_intervals']:
  if x['start_step']<=t<=x['end_step']: return x['node_id'],x
 return 'unknown',{'start_step':t,'end_step':t}
def edgeat(a,t):
 return next((e for e in a['edge_intervals'] if e['end_step']==t),None)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--adapter-dir',type=Path,required=True);ap.add_argument('--suite-dir',type=Path,required=True);ap.add_argument('--linearization-spec',type=Path,required=True);ap.add_argument('--methods',default='linear_time_fraction,oracle_linear_chain,sequential_transition_oracle');ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--task');a=ap.parse_args()
 anns={x['episode_id']:x for x in rowsj(a.adapter_dir/'runtime_episode_annotations.jsonl')}; diags=list(csv.DictReader((a.suite_dir/'tables/diagnostic_episodes.csv').open())); specs=yaml.safe_load(a.linearization_spec.read_text())['linearizations']; a.output_dir.mkdir(parents=True,exist_ok=True)
 for d in diags:
  if a.task and d['task_id']!=a.task: continue
  an=anns[d['episode_id']]; T=max(x['end_step'] for x in an['node_intervals'])+1
  chains=[]
  if d['task_id']=='transport_dual_order': chains=[('A_first',specs['dual_order_A_first']['canonical_nodes']),('B_first',specs['dual_order_B_first']['canonical_nodes'])]
  else: chains=[('recovery_chain',specs['transport_recovery_chain']['canonical_nodes'])]
  for method in a.methods.split(','):
   targets=[('none',None)] if method=='linear_time_fraction' else chains
   for orient,chain in targets:
    out=a.output_dir/f'{method}__{d["episode_id"]}__{orient}.jsonl'; last_rank=0; previous=0.0; lines=[]; K=(len(chain)-1) if chain else max(1,T-1)
    for t in range(T):
     node,interval=nodeat(an,t); edge=edgeat(an,t)
     if method=='linear_time_fraction': p=t/max(1,T-1); rank=None; phi=t/max(1,T-1)
     else:
      if node in chain:
       rank=chain.index(node); last_rank=rank; phi=(t-interval['start_step'])/max(1,interval['end_step']-interval['start_step']); p=(rank+phi)/K
      else: rank=last_rank; phi=0.0; p=last_rank/K
     if method=='sequential_transition_oracle':
      if edge:
       typ=edge['edge_type']; src=nodeat(an,int(edge['start_step']))[0]; dst=nodeat(an,int(edge['end_step']))[0]
       if typ=='stagnation': delta=0.0
       elif typ=='failure': delta=-1.0/K
       elif src in chain and dst in chain: delta=(chain.index(dst)-chain.index(src))/K
       elif typ=='recovery' and dst in chain: delta=(chain.index(dst)-last_rank)/K
       else: delta=0.0
       phi=0.0 if typ in {'failure','recovery','alternative'} else phi
      else: delta=0.0
      p=float(max(0.0,min(1.0,previous+delta)))
     else:
      delta=0.0 if t==0 else p-previous
     lines.append({'episode_id':d['episode_id'],'task_id':d['task_id'],'scenario':d['scenario'],'split':d['split_original'],'content_group_id':d['content_group_id'],'method':method,'orientation':orient,'seed':None,'step':t,'node_id_runtime':node,'edge_id_runtime':edge['edge_id'] if edge else None,'semantic_edge_type':edge['edge_type'] if edge else None,'progress':p,'reward_delta':delta,'stage_rank':rank,'within_node_phi':phi,'uncertainty':None,'controller_source':d['controller_source']})
     previous=p
    out.write_text(''.join(json.dumps(x)+'\n' for x in lines))
if __name__=='__main__':main()
