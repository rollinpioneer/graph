#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--claim-scope',type=Path,required=True);p.add_argument('--claim-matrix',type=Path,required=True);p.add_argument('--final-results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();scope=json.loads(a.claim_scope.read_text());res=pd.read_csv(a.final_results); supported=res[res.category=='supported_primary_result'].result_id.tolist()
 text='# Final Contribution Statement\n\nThis work is limited to a locked manual, graph-structured reward representation. Its contribution is to encode multiple legal paths, explicit failure/recovery transitions, remaining-cost reduction plus within-node progress, and recovery-debt accounting. The final statistical evidence supports: '+', '.join(supported)+'.\n\nIt does not claim stable policy improvement, coverage scaling, unseen-order generalization, or automatic graph discovery as a main contribution.\n'
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text,encoding='utf-8')
if __name__=='__main__':main()
