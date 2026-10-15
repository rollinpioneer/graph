#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import yaml

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-plan',type=Path,required=True);p.add_argument('--table-source-map',type=Path,required=True);p.add_argument('--figure-source-map',type=Path,required=True);p.add_argument('--claim-scope',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plan=yaml.safe_load(a.artifact_plan.read_text());
 lines=['# Caption Bank','']
 for name in plan['main_figures']:
  if name=='figure_1_pathgraph_method': text='Locked manual PathGraph method schematic. Alternative legal routes, failure/recovery transitions, within-node progress, and debt-capped recovery are shown. RA-BC is dashed because it is secondary downstream use; this is not an automatic-graph result.'
  elif name=='figure_2_reward_behavior': text='Checkpoint-reproduced reward behavior. Points are content-group estimates and bars are percentile 95% bootstrap intervals; controlled symbolic path checks retain their explicitly limited support.'
  elif name=='figure_3_structural_ablations': text='Frozen structural variants recomputed from real test predictions and controlled symbolic stress where required. Error bars are content-group percentile 95% bootstrap intervals; mixed-provenance reference aggregates were not substituted.'
  else: text='Auxiliary history and uncertainty evidence. Provenance is retained from the frozen R1 evidence tables and it is not a new main-policy or generalization claim.'
  lines += [f'## {name}','',text,'']
 for name in plan['appendix_figures']:
  lines += [f'## {name}','',f'Appendix-only evidence for `{name}`. It is labelled secondary or negative where applicable and does not expand the locked core-reward claim scope.','']
 for name in plan['main_tables']+plan['appendix_tables']:
  lines += [f'## {name}','',f'Table values are taken directly from the source-map-listed final CSVs; `NA`, `Not estimable`, and unsupported status are preserved rather than replaced with zero.','']
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':main()
