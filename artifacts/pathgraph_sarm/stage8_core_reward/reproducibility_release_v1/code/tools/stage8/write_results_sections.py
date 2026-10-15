#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
def line(row): return f"{row.estimand_id}: {row.point_estimate:.3f} (95% CI [{row.ci95_low:.3f}, {row.ci95_high:.3f}]; {row.support_note})."
def main():
 p=argparse.ArgumentParser()
 for x in ('main_table','model_table','ablation_table','statistics','bootstrap','history','uncertainty','policy','claim_scope'):p.add_argument('--'+x.replace('_','-'),type=Path,required=True)
 p.add_argument('--results-output',type=Path,required=True);p.add_argument('--ablation-output',type=Path,required=True);a=p.parse_args();main=pd.read_csv(a.main_table);model=pd.read_csv(a.model_table);boot=pd.read_csv(a.bootstrap);scope=json.loads(a.claim_scope.read_text());locked=main[main.method=='pathgraph_reward_v1_locked'].iloc[0]
 primary=boot[boot.estimand_kind=='primary'];struct=boot[boot.estimand_kind=='structural_contrast']
 results=f'''# Results\n\nS1. The independently reproduced locked reward-model ensemble achieved node macro-F1 {locked.node_macro_f1:.3f}, edge-type macro-F1 {locked.edge_type_macro_f1_non_none:.3f}, phi MAE {locked.phi_mae:.3f}, and remaining-cost MAE {locked.cost_mae:.3f}.\n\nS2. Group-level primary estimates were:\n\n'''+ '\n'.join('- '+line(row) for _,row in primary.iterrows())+'''\n\nThese are bounded benchmark findings. Controlled symbolic rows retain their limited-support qualifier, and no stable-policy, coverage-scaling, or unseen-order extension is claimed.\n'''
 ablation='# Structural Ablations\n\nS3. Predeclared, no-retraining structural contrasts were:\n\n'+'\n'.join('- '+line(row) for _,row in struct.iterrows())+'\n\nAlternative-order collapse has limited controlled support; the recovery and debt-cap contrasts are reported with their provenance. The no-phi contrast is not promoted beyond its measured result.\n'
 a.results_output.parent.mkdir(parents=True,exist_ok=True);a.results_output.write_text(results,encoding='utf-8');a.ablation_output.write_text(ablation,encoding='utf-8')
if __name__=='__main__':main()
