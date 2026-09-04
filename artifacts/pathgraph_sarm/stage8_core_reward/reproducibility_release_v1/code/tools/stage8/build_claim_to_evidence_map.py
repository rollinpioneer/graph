#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--claim-registry',type=Path,required=True);p.add_argument('--tables',type=Path,required=True);p.add_argument('--figures',type=Path,required=True);p.add_argument('--manuscript',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();registry=pd.read_csv(a.claim_registry);rows=[]
 mapping={'C1':('S2','results','table_1_main_reward_results / figure_2_reward_behavior','path_consistency','reproduced_reward_main_table.csv','supported','bounded benchmark; controlled trace support is limited'),'C2':('S2','results','table_1_main_reward_results / figure_2_reward_behavior','failure_sign; recovery_sign','final_bootstrap_effects.csv','supported','content-group bootstrap'),'C3':('S2','results','table_A2_history_granularity / figure_4_history_and_uncertainty','history metric','history_granularity_summary.csv','partially_supported','current benchmark only'),'C4':('limitations','limitations','table_A4_policy_secondary_mixed / figure_A2_policy_secondary','secondary evidence','policy_secondary_evidence.csv','not_supported','secondary/mixed only'),'C5':('limitations','limitations','figure_A3_auto_graph_extension','automatic graph','auto_graph_test_metrics.csv','not_supported','not a main contribution'),'C6':('S2','results','table_A3_uncertainty / figure_4_history_and_uncertainty','AUROC','uncertainty_error_detection.csv','supported','auxiliary only'),'C7':('limitations','limitations','table_A5_negative_extensions / figure_A4_coverage_and_unseen_order_negative_results','coverage','coverage_scaling_metrics.csv','not_supported','negative extension'),'C8':('limitations','limitations','table_A5_negative_extensions / figure_A4_coverage_and_unseen_order_negative_results','unseen order','ood_reward_metrics.csv','not_supported','negative extension')}
 for _,r in registry.iterrows():
  sent,sec,art,metric,source,status,qual=mapping[r.claim_id];rows.append({'claim_id':r.claim_id,'sentence_id':sent,'section':sec,'table_or_figure':art,'metric':metric,'source_csv':source,'support_status':status,'qualifier':qual})
 a.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(a.output,index=False)
if __name__=='__main__':main()
