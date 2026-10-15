#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--claim-scope',type=Path,required=True);p.add_argument('--final-summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=pd.read_csv(a.final_summary);ids=d[d.category=='supported_primary_result'].result_id.tolist()
 text='# Abstract Result Sentences\n\nWe evaluate a locked manual graph-structured reward representation for multiple legal paths and explicit failure/recovery semantics. Content-group bootstrap evidence supports '+', '.join(ids)+'. We position policy evidence as secondary/mixed and do not claim scaling, unseen-order generalization, or automatic graph discovery.\n';a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text,encoding='utf-8')
if __name__=='__main__':main()
