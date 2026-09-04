#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--queue',required=True); ap.add_argument('--output-dir',required=True); a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    for r in csv.DictReader(open(a.queue)):
        x={'episode_id':r['episode_id'],'task_id':r['task_id'],'graph_id':r['task_id']+'_graph','graph_version':'1.0.0','source_path':r['source_path'],'path_signature':[],'outcome':'success','node_intervals':[],'edge_intervals':[],'progress_anchors':[],'failure_events':[],'recovery_events':[],'review':{'status':'proposed','reviewer':'stage2','notes':''}}
        (out/f"{r['episode_id']}.json").write_text(json.dumps(x,indent=2)+'\n')
if __name__=='__main__': main()
