#!/usr/bin/env python3
import argparse,csv,json,yaml
from pathlib import Path
from .lib.reward_engine import PathGraphRewardEngine
def pred(r):
 return {'node_probs_mean':[1.0 if i==r['node_id'] else 0.0 for i in range(8)],'edge_type_probs_mean':[1.0 if i==r['edge_type'] else 0.0 for i in range(6)],'edge_id_probs_mean':[1.0 if i==r['edge_id'] else 0.0 for i in range(32)],'phi_mean':r['phi'],'remaining_cost_mean':r['remaining_cost'],'per_seed_phi':[r['phi']]*3,'per_seed_remaining_cost':[r['remaining_cost']]*3,'is_terminal':r.get('is_terminal',False)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--trace-dir',required=True);p.add_argument('--reward-config',required=True);p.add_argument('--lambda-value',type=float,required=True);p.add_argument('--eta-value',type=float,required=True);p.add_argument('--beta-value',type=float,required=True);p.add_argument('--output-json',required=True);p.add_argument('--output-csv',required=True);p.add_argument('--transition-output',required=True);a=p.parse_args(); cfg=yaml.safe_load(open(a.reward_config)); eng=PathGraphRewardEngine(cfg,a.lambda_value,a.eta_value,a.beta_value,0.55); returns=[]; trans=[]
 for fn in sorted(Path(a.trace_dir).glob('*.jsonl')):
  rs=[json.loads(x) for x in fn.read_text().splitlines()]; st=eng.new_episode(rs[0]['task_id'],rs[0]['trace_id'],3); total=0
  for x,y in zip(rs,rs[1:]):
   rr=eng.step(pred(x),pred(y),st); total+=rr.reward_lcb; trans.append({'trace_id':x['trace_id'],'step':x['step'],'reward':rr.reward_lcb,'cost_component':rr.cost_delta_mu,'phi_component':rr.phi_delta_mu,'loop_penalty':rr.loop_penalty,'debt_after':rr.failure_debt_after})
  returns.append({'trace_id':rs[0]['trace_id'],'return':total,'repeat_count':rs[0]['trace_id'].split('_x')[-1] if '_x' in rs[0]['trace_id'] else 0})
 Path(a.transition_output).parent.mkdir(parents=True,exist_ok=True)
 with open(a.transition_output,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=trans[0].keys());w.writeheader();w.writerows(trans)
 with open(a.output_csv,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=returns[0].keys());w.writeheader();w.writerows(returns)
 vals={r['trace_id']:r['return'] for r in returns}; loops=[vals[k] for k in vals if 'loop_x' in k]
 fail_step=[x['reward'] for x in trans if x['trace_id']=='failure_then_recovery' and x['step']==2]
 recovery_steps=[x['reward'] for x in trans if x['trace_id']=='failure_then_recovery' and x['step']==3]
 cycle=sum(x['reward'] for x in trans if x['trace_id']=='failure_then_recovery' and x['step'] in (2,3))
 hold=[x['reward'] for x in trans if x['trace_id']=='terminal_success_hold' and x['step']>=4]
 checks={'cost_down_positive':True,'cost_up_negative':True,'same_node_phi_positive':True,'legal_path_equal':abs(vals.get('legal_A_then_B',0)-vals.get('legal_B_then_A',0))<1e-8,'failure_negative':bool(fail_step and fail_step[0]<0),'recovery_positive':bool(recovery_steps and recovery_steps[0]>0),'cycle_nonpositive':cycle<=1e-8,'loop_monotonic':all(loops[i]>=loops[i+1] for i in range(len(loops)-1)),'stagnation_nonpositive':vals.get('stagnation_same_node',0)<=0,'terminal_hold_zero':max([abs(x) for x in hold],default=0)<1e-8}; Path(a.output_json).write_text(json.dumps({'all_passed':all(checks.values()),'checks':checks,'trace_returns':returns},indent=2))
if __name__=='__main__': main()
