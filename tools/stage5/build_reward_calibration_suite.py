#!/usr/bin/env python3
import argparse,csv,hashlib,shutil
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--val-predictions',required=True);p.add_argument('--oracle-trace-dir',required=True);p.add_argument('--episode-manifest',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--manifest',required=True);a=p.parse_args(); o=Path(a.output_dir);o.mkdir(parents=True,exist_ok=True); dst=o/'ensemble_val_predictions.jsonl.gz'; shutil.copy2(a.val_predictions,dst); od=o/'oracle_traces'; shutil.copytree(a.oracle_trace_dir,od,dirs_exist_ok=True); rows=[(str(dst.resolve()),'transport_recovery_val'),(str(od.resolve()),'oracle_graph_trace_bank')]
 with open(a.manifest,'w',newline='') as f:
  w=csv.writer(f);w.writerow(['path','sha256','selection_role']);
  for path,role in rows:
   q=Path(path); h=hashlib.sha256(q.read_bytes()).hexdigest() if q.is_file() else 'directory_manifest';w.writerow([path,h,role])
if __name__=='__main__': main()
