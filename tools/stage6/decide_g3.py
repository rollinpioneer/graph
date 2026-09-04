#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--policy-data-gate',type=Path,required=True);p.add_argument('--weighting-gate',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();e=json.loads(a.evidence.read_text());pd=json.loads(a.policy_data_gate.read_text());wg=json.loads(a.weighting_gate.read_text())
 checks={'real_action_data_ready':pd.get('decision')=='POLICY_DATA_REAL_ACTION_READY','weighting_ready':wg.get('decision')=='WEIGHTING_PIPELINE_READY','protocol_locked':bool(e.get('policy_protocol_locked')),'paired_evaluation':bool(e.get('paired_evaluation')),'no_post_test_reward_retune':not bool(e.get('reward_retuned_after_test')),'graph_gain_ge_0_05':e['graph_task_success_gain']>=.05,'improved_seeds_ge_2':e['improved_policy_seed_count']>=2,'fixed_order_drop_le_0_05':e['fixed_order_drop']<=.05,'structure_specific_gain':bool(e['recovery_success_gain']>=.08 or e['worst_order_success_gain']>=.08 or e['long_horizon_completion_gain']>=.05)}
 if all(checks.values()):decision='GO_STAGE7'
 elif e['graph_task_success_gain']<=0 or e['improved_policy_seed_count']==0:decision='NARROW_TO_REWARD_ONLY'
 else:decision='REFINE_STAGE6'
 result={'decision':decision,'checks':checks,'evidence':e,'rule':'predeclared_G3'};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
