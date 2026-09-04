#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import pandas as pd
from tools.stage8.common import dump_json

TABLES=['table_1_main_reward_results','table_2_structural_ablations','table_3_model_components','table_4_final_claim_scope','table_A1_model_seed_results','table_A2_history_granularity','table_A3_uncertainty','table_A4_policy_secondary_mixed','table_A5_negative_extensions']
FIGURES=['figure_1_pathgraph_method','figure_2_reward_behavior','figure_3_structural_ablations','figure_4_history_and_uncertainty','figure_A1_per_seed_metrics','figure_A2_policy_secondary','figure_A3_auto_graph_extension','figure_A4_coverage_and_unseen_order_negative_results']
def main():
 p=argparse.ArgumentParser();p.add_argument('--tables',type=Path,required=True);p.add_argument('--figures',type=Path,required=True);p.add_argument('--captions',type=Path,required=True);p.add_argument('--table-source-map',type=Path,required=True);p.add_argument('--figure-source-map',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 missing=[str(a.tables/f'{x}.{e}') for x in TABLES for e in ('csv','md','tex') if not (a.tables/f'{x}.{e}').is_file()]+[str(a.figures/f'{x}.{e}') for x in FIGURES for e in ('pdf','svg','png') if not (a.figures/f'{x}.{e}').is_file()]
 tablemap=pd.read_csv(a.table_source_map);figmap=pd.read_csv(a.figure_source_map);mapped=set(tablemap.artifact_id)|set(figmap.artifact_id);unmapped=[x for x in TABLES+FIGURES if x not in mapped]
 captions=a.captions.read_text(encoding='utf-8');uncaptioned=[x for x in TABLES+FIGURES if x not in captions]
 display_bad=[]
 for file in a.tables.glob('*.csv'):
  table=pd.read_csv(file,dtype=str,keep_default_na=False)
  cells={str(value).strip().lower() for value in table.to_numpy().ravel()}
  if {'nan','+nan','-nan','inf','+inf','-inf'} & cells: display_bad.append(str(file))
 bad_titles=[]
 for file in ('figure_2_reward_behavior.svg','figure_3_structural_ablations.svg','figure_4_history_and_uncertainty.svg'):
  if (a.figures/file).exists() and any(term in (a.figures/file).read_text(errors='ignore').lower() for term in ('coverage scaling','unseen order','automatic graph','policy improvement')):bad_titles.append(file)
 decision='PUBLICATION_ARTIFACTS_READY' if not (missing or unmapped or uncaptioned or display_bad or bad_titles) else ('MISSING_VALUE_SOURCE' if unmapped else 'REPAIR_FIGURE_RENDERING')
 dump_json(a.output,{'decision':decision,'main_tables':4,'appendix_tables':5,'main_figures':4,'appendix_figures':4,'missing_files':missing,'unmapped_artifacts':unmapped,'uncaptioned_artifacts':uncaptioned,'nan_or_inf_display_files':display_bad,'unsupported_claim_in_main_title':bad_titles})
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(f'# Publication Artifact Summary\n\nDecision: `{decision}`. All required artifacts are source-mapped and captioned when this gate passes.\n',encoding='utf-8')
 if decision!='PUBLICATION_ARTIFACTS_READY':raise SystemExit(2)
if __name__=='__main__':main()
