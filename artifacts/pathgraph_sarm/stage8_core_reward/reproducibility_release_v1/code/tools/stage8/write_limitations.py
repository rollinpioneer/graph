#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser()
 for x in ('claim_matrix','g4','coverage','ood','policy','auto_graph'):p.add_argument('--'+x.replace('_','-'),type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args();g=json.loads(a.g4.read_text())
 text='''# Limitations\n\nThe manual graph remains the main method. The real R1 coverage reruns do not support a coverage-scaling claim, and the real order-holdout reruns do not support unseen-order generalization. Policy evidence is secondary/mixed rather than a stable cross-seed improvement. Automatic graph discovery is not a main contribution. The task and data coverage are limited. Controlled symbolic stress checks establish only frozen graph semantics, not real-robot generalization.\n''';a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text,encoding='utf-8')
if __name__=='__main__':main()
