#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from cp_disr.analysis import s1_evidence_review as review
def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="command",required=True)
    for name in ("inspect-runner","inspect-gates","inspect-probe","summarize","verify"):
        q=s.add_parser(name); q.add_argument("--root",required=True); q.add_argument("--source",required=True); q.add_argument("--output",required=True)
    a=p.parse_args(); root=Path(a.root).resolve(); source=Path(a.source); source=source if source.is_absolute() else root/source; out=Path(a.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    if a.command=="inspect-runner": result=review.inspect_runner(root,source,out)
    elif a.command=="inspect-gates": result=review.inspect_gates(root,source,out)
    elif a.command=="inspect-probe": result=review.inspect_probe(root,source,out)
    elif a.command=="summarize": result=review.summarize(root,source,out)
    else: result=review.verify(root,source,out)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
if __name__=="__main__": main()
