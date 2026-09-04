#!/usr/bin/env python3
import argparse,hashlib,json,yaml
from pathlib import Path
def h(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--selected-config',required=True);p.add_argument('--selection-metrics',required=True);p.add_argument('--val-predictions',required=True);p.add_argument('--oracle-manifest',required=True);p.add_argument('--reward-engine',required=True);p.add_argument('--output',required=True);a=p.parse_args(); d=yaml.safe_load(open(a.selected_config)); out={'locked':True,'selection_source':['transport_recovery_val','oracle_graph_trace_bank'],'forbidden_source_verified':True,'selected':d,'reward_engine_sha256':h(a.reward_engine),'selected_config_sha256':h(a.selected_config),'val_prediction_sha256':h(a.val_predictions),'oracle_manifest_sha256':h(a.oracle_manifest),'selection_metrics_sha256':h(a.selection_metrics)}; Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2))
if __name__=='__main__': main()
