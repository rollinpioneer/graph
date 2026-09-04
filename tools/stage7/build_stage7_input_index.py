#!/usr/bin/env python3
import argparse,hashlib,os
from pathlib import Path
import yaml
try:
 from .common import digest
except ImportError:
 from common import digest
def main():
 p=argparse.ArgumentParser();
 for n in ['m1','stage3-input','diagnostic-suite','supervision','real-predictions','reward-v1','persistent-model','stage6-policy-evidence','stage6r1-evidence']: p.add_argument('--'+n,required=True)
 p.add_argument('--output',required=True);p.add_argument('--hash-output',required=True);a=p.parse_args()
 m1=Path(a.m1); s3=Path(a.stage3_input); diag=Path(a.diagnostic_suite); sup=Path(a.supervision); pred=Path(a.real_predictions); rew=Path(a.reward_v1); per=Path(a.persistent_model); s6=Path(a.stage6_policy_evidence); s61=Path(a.stage6r1_evidence)
 graph=next(s3.glob('runtime_graph_specs_v1.0.1/*'))
 obj={'statistics_unit':'content_group_id','stage7_mode':'reward_only','graph_specs':{'root':str(graph.parent),'version':'v1.0.1'},'ground_truth':{'root':str(m1/'gt_v1'),'splits':str(m1/'gt_v1/gt_splits.csv')},'diagnostic_suite':{'root':str(diag),'frozen_sha256':digest(diag/'DIAGNOSTIC_SUITE_SHA256SUMS.txt')},'supervision':{'root':str(sup),'forbidden_features':['outcome','success','scenario','controller_source','episode_id','content_group_id']},'real_predictions':{'val':str(pred/'tables/ensemble_val_predictions.jsonl.gz'),'test':str(pred/'tables/ensemble_test_predictions.jsonl.gz'),'stage3_diagnostic':str(pred/'tables/ensemble_stage3_diagnostic_predictions.jsonl.gz')},'reward':{'config':str(rew/'configs/reward_config_v1.yaml'),'selection_lock':str(rew/'configs/reward_selection_lock.json'),'engine':str(rew/'code/reward_engine.py')},'model_bundle':{'persistent_manifest':str(per/'configs/model_bundle_persistent.json'),'checkpoint_count':3},'policy_evidence':{'status':'secondary_mixed','g3_r1':'NARROW_TO_REWARD_ONLY'},'source_roots':{'stage6_policy_evidence':str(s6),'stage6r1_evidence':str(s61)}}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(yaml.safe_dump(obj,sort_keys=False))
 rows=[]
 for k,v in [('stage7_input_index',a.output),('claim_source_m1',m1/'M1_SHA256SUMS.txt'),('diagnostic_lock',diag/'DIAGNOSTIC_SUITE_SHA256SUMS.txt'),('supervision_lock',sup/'SUPERVISION_SHA256SUMS.txt'),('real_prediction_lock',pred/'REAL_PREDICTIONS_SHA256SUMS.txt'),('reward_lock',rew/'STAGE5_REWARD_SHA256SUMS.txt'),('persistent_lock',per/'PERSISTED_INPUTS_SHA256SUMS.txt'),('stage6r1_decision',s61/'metrics/g3_refine1_decision.json')]:
  q=Path(v); rows.append(f'{k}\t{q}\t{digest(q) if q.is_file() else "MISSING"}')
 Path(a.hash_output).parent.mkdir(parents=True,exist_ok=True); Path(a.hash_output).write_text('artifact\tpath\tsha256\n'+'\n'.join(rows)+'\n')
 print(a.output)
if __name__=='__main__':main()
