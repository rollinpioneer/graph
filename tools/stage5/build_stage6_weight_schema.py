#!/usr/bin/env python3
import argparse,json,yaml
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--reward-config',required=True);p.add_argument('--val-rewards',required=True);p.add_argument('--output-schema',required=True);p.add_argument('--normalization',required=True);p.add_argument('--example',required=True);a=p.parse_args(); fields=['episode_id','content_group_id','task_id','t_start','t_end','reward_mu','reward_std','reward_lcb','weight_positive','cost_component','phi_component','loop_penalty','recovery_cap_delta','edge_type_pred','edge_id_pred','node_confidence','edge_confidence'];Path(a.output_schema).write_text(json.dumps({'schema_version':'stage6-weight-v1','statistics_unit':'content_group_id','fields':fields},indent=2));Path(a.normalization).write_text(json.dumps({'method':'identity_positive_lcb','source':'validation_reward_traces','clip_nonnegative':True},indent=2));Path(a.example).write_text(json.dumps({k:(0 if k not in ('episode_id','content_group_id','task_id') else ('example' if k!='task_id' else 'transport_recovery')) for k in fields},indent=2))
if __name__=='__main__':main()
