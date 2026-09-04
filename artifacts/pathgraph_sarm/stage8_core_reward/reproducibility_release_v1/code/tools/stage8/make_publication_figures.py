#!/usr/bin/env python3
"""Create all publication figures from supplied final CSVs, never literals."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def save(fig, base: Path, formats: str, dpi: int):
    for ext in formats.split(','): fig.savefig(base.with_suffix('.'+ext),dpi=dpi,bbox_inches='tight')
    plt.close(fig)
def numeric(s): return pd.to_numeric(s,errors='coerce')
def main():
    p=argparse.ArgumentParser()
    for name in ('main_results','model_results','ablations','bootstrap','history','uncertainty','policy','coverage','ood','auto_graph') : p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--formats',default='pdf,svg,png'); p.add_argument('--dpi',type=int,default=300); p.add_argument('--source-map',type=Path,required=True); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    main=pd.read_csv(a.main_results); model=pd.read_csv(a.model_results); ab=pd.read_csv(a.ablations); boot=pd.read_csv(a.bootstrap); hist=pd.read_csv(a.history); unc=pd.read_csv(a.uncertainty); policy=pd.read_csv(a.policy); cov=pd.read_csv(a.coverage); ood=pd.read_csv(a.ood); auto=pd.read_csv(a.auto_graph)
    sources=[]
    b=boot.set_index('estimand_id')
    ids=['path_consistency','failure_sign','recovery_sign','loop_safety','success_separation']; sub=b.loc[ids]; fig,ax=plt.subplots(figsize=(9,4)); ax.errorbar(range(len(sub)),sub.point_estimate,yerr=[sub.point_estimate-sub.ci95_low,sub.ci95_high-sub.point_estimate],fmt='o',capsize=4,color='#1f4e79'); ax.set_xticks(range(len(sub)),[x.replace('_','\n') for x in ids]); ax.axhline(0,color='black',lw=.7); ax.set_ylabel('Estimate (95% percentile CI)'); ax.set_title('Reward behavior: checkpoint reproduction and grouped statistics'); save(fig,a.output_dir/'figure_2_reward_behavior',a.formats,a.dpi); sources.append(('figure_2_reward_behavior',[a.bootstrap]))
    ids=['alternative_A_collapse','alternative_B_collapse','remove_recovery','remove_debt_cap','remove_phi']; sub=b.loc[ids]; fig,ax=plt.subplots(figsize=(9,4)); ax.bar(range(len(sub)),sub.point_estimate,color='#4f81bd'); ax.errorbar(range(len(sub)),sub.point_estimate,yerr=[sub.point_estimate-sub.ci95_low,sub.ci95_high-sub.point_estimate],fmt='none',ecolor='black',capsize=4); ax.set_xticks(range(len(sub)),[x.replace('_','\n') for x in ids]); ax.axhline(0,color='black',lw=.7); ax.set_ylabel('Full − ablated variant (metric-specific)'); ax.set_title('Structural ablations (controlled and real-test provenance retained)'); save(fig,a.output_dir/'figure_3_structural_ablations',a.formats,a.dpi); sources.append(('figure_3_structural_ablations',[a.bootstrap,a.ablations]))
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4)); ax1.plot(hist.variant_id,hist.node_macro_f1_mapped_to_default,marker='o'); ax1.tick_params(axis='x',rotation=35); ax1.set_ylabel('Mapped node macro-F1'); ax1.set_title('History / granularity (auxiliary)'); ax2.bar(unc.signal,numeric(unc.AUROC)); ax2.set_ylim(0,1); ax2.set_ylabel('AUROC'); ax2.set_title('Uncertainty signal (auxiliary)'); save(fig,a.output_dir/'figure_4_history_and_uncertainty',a.formats,a.dpi); sources.append(('figure_4_history_and_uncertainty',[a.history,a.uncertainty]))
    seed=model[(model.suite=='test') & (model.model_seed!='ensemble')][['model_seed','node_macro_f1','edge_type_macro_f1_non_none']]; fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(len(seed)); ax.bar(x-.18,numeric(seed.node_macro_f1),.36,label='Node macro-F1'); ax.bar(x+.18,numeric(seed.edge_type_macro_f1_non_none),.36,label='Edge-type macro-F1'); ax.set_xticks(x,seed.model_seed.astype(str)); ax.legend(); ax.set_title('Fixed checkpoint seed metrics'); save(fig,a.output_dir/'figure_A1_per_seed_metrics',a.formats,a.dpi); sources.append(('figure_A1_per_seed_metrics',[a.model_results]))
    fig,ax=plt.subplots(figsize=(6,4)); ax.bar(policy.evidence,numeric(policy.graph_task_success_gain),label='Graph-task success gain'); ax.bar(policy.evidence,numeric(policy.recovery_success_gain),bottom=numeric(policy.graph_task_success_gain),label='Recovery gain'); ax.legend(); ax.set_title('Secondary / mixed policy evidence'); save(fig,a.output_dir/'figure_A2_policy_secondary',a.formats,a.dpi); sources.append(('figure_A2_policy_secondary',[a.policy]))
    fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(numeric(auto.normalized_graph_edit_distance),numeric(auto.node_mapping_macro_f1)); ax.set_xlabel('Normalized graph edit distance'); ax.set_ylabel('Node mapping macro-F1'); ax.set_title('Automatic graph exploration (appendix only)'); save(fig,a.output_dir/'figure_A3_auto_graph_extension',a.formats,a.dpi); sources.append(('figure_A3_auto_graph_extension',[a.auto_graph]))
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4)); ax1.plot(numeric(cov.train_fraction),numeric(cov.node_macro_f1),marker='o'); ax1.set_xlabel('Train fraction'); ax1.set_ylabel('Node macro-F1'); ax1.set_title('Coverage: negative extension'); x=np.arange(len(ood)); y=numeric(ood.unseen_order_alternative_edge_f1); ax2.bar(x,y); ax2.set_xticks(x,ood.suite,rotation=30,ha='right'); ax2.set_ylabel('Unseen-order alternative-edge F1'); ax2.set_title('Unseen order: not supported');
    for pos,value in enumerate(y):
        if not np.isfinite(value): ax2.text(pos,0.02,'Not estimable',rotation=90,ha='center',va='bottom',fontsize=8)
    save(fig,a.output_dir/'figure_A4_coverage_and_unseen_order_negative_results',a.formats,a.dpi); sources.append(('figure_A4_coverage_and_unseen_order_negative_results',[a.coverage,a.ood]))
    rows=[]
    for artifact,paths in sources:
        for path in paths: rows.append({'artifact_id':artifact,'artifact_type':'figure','source_path':str(path.resolve()),'source_kind':'final_csv','statistics_unit_or_scope':'content_group_id_or_explicit_auxiliary_provenance','unsupported_claim_in_main':False})
    pd.DataFrame(rows).to_csv(a.source_map,index=False)
if __name__=='__main__': main()
