#!/usr/bin/env python3
import argparse
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--annotation-dir',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--graph-dir'); a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); rows=list(Path(a.annotation_dir).glob('*.json')); (out/'episode_annotations.jsonl').write_text(''.join(p.read_text() for p in rows));
    for n,h in [('node_intervals.csv','episode_id,task_id,node_id,start_step,end_step\n'),('edge_intervals.csv','episode_id,task_id,edge_id,edge_type,start_step,end_step,attempt_index\n'),('progress_anchors.csv','episode_id,task_id,node_id,step,value\n'),('failure_recovery_events.csv','episode_id,task_id,failure_onset_step,recovery_complete_step\n'),('gt_episode_manifest.jsonl',''),('gt_splits.csv','episode_id,task_id,split,group_id\n'),('label_stats.csv','task_id,gt_episode_count\n'),('coverage_by_node_edge.csv','task_id,edge_id,edge_type,count\n'),('annotation_provenance.csv','episode_id,task_id,provenance,stage1_placeholder_used\n')]: (out/n).write_text(h)
if __name__=='__main__': main()
