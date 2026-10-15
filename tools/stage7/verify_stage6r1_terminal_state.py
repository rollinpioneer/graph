#!/usr/bin/env python3
import argparse,json
def main():
 p=argparse.ArgumentParser(); p.add_argument('--decision',required=True); p.add_argument('--evidence',required=True); p.add_argument('--replication',required=True); p.add_argument('--output',required=True); a=p.parse_args()
 d=json.load(open(a.decision)); e=json.load(open(a.evidence)); r=json.load(open(a.replication)); tol=1e-6
 checks={
  'decision_narrow_reward_only':d.get('decision')=='NARROW_TO_REWARD_ONLY',
  'combined_seed_count_6':e.get('combined_seed_count')==6,
  'new_seed_improved_count_1':r.get('new_seed_improved_count')==1,
  'new_seed_ceiling_tie_count_1':r.get('new_seed_ceiling_tie_count')==1,
  'new_seed_degraded_count_0':r.get('new_seed_degraded_count')==0,
  'graph_gain':abs(float(e.get('combined_graph_task_success_gain',99))-.0666666666666667)<=tol,
  'recovery_gain':abs(float(e.get('combined_recovery_success_gain',99))-.1666666666666666)<=tol,
  'fixed_order_drop_zero':abs(float(e.get('combined_fixed_order_drop',99)))<=tol,
  'reward_retuned_after_test_false':e.get('reward_retuned_after_test') is False,
  'paired_evaluation':e.get('paired_evaluation') is True,
 }
 out={'decision':'STAGE6R1_TERMINAL_VERIFIED' if all(checks.values()) else 'STAGE6R1_TERMINAL_MISMATCH','checks':checks}
 open(a.output,'w').write(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); return 0 if all(checks.values()) else 2
if __name__=='__main__': raise SystemExit(main())
