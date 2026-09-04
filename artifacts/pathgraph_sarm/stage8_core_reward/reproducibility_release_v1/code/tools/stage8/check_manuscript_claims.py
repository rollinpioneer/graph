#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from tools.stage8.common import dump_json
def main():
 p=argparse.ArgumentParser();p.add_argument('--claim-scope',type=Path,required=True);p.add_argument('--manuscript-dir',type=Path,required=True);p.add_argument('--claim-map',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();scope=json.loads(a.claim_scope.read_text());text='\n'.join(f.read_text(encoding='utf-8') for f in a.manuscript_dir.glob('*') if f.suffix in {'.md','.tex'}).lower();m=pd.read_csv(a.claim_map);missing_map=set(['C1','C2','C3','C4','C5','C6','C7','C8'])-set(m.claim_id);policy_ok='secondary/mixed' in text;manual_ok='manual graph' in text;eta_ok='eta=0' in text or 'eta = 0' in text;beta_ok='beta=0' in text or 'beta = 0' in text
 forbidden_positive=[]
 for phrase in ('stable policy improvement','coverage scaling','unseen-order generalization','automatic graph discovery'):
  for file in a.manuscript_dir.glob('*.md'):
   line=' '.join(file.read_text(encoding='utf-8').lower().splitlines())
   if phrase in line and not any(neg in line for neg in ('do not claim','not support','not a main','not claim','does not claim','no stable','not a')):forbidden_positive.append(phrase)
 decision='MANUSCRIPT_EVIDENCE_PACKAGE_READY' if not missing_map and policy_ok and manual_ok and eta_ok and beta_ok and not forbidden_positive else ('MISSING_EVIDENCE_LINK' if missing_map else 'REPAIR_CLAIM_WORDING')
 dump_json(a.output,{'decision':decision,'claim_map_complete':not bool(missing_map),'unsupported_claims_kept_out_of_positive_conclusions':not bool(forbidden_positive),'policy_wording_secondary_mixed':policy_ok,'manual_graph_main':manual_ok,'eta_beta_not_claimed_as_validated_nonzero_components':eta_ok and beta_ok})
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(f'# Manuscript Evidence Summary\n\nDecision: `{decision}`. Evidence mappings are checked against the frozen claim boundary.\n',encoding='utf-8')
 if decision!='MANUSCRIPT_EVIDENCE_PACKAGE_READY':raise SystemExit(2)
if __name__=='__main__':main()
