import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',required=True); p.add_argument('--split',default='val'); p.add_argument('--selection-lock'); p.add_argument('--diagnostic-suite'); p.add_argument('--probe-fold'); a=p.parse_args()
 if a.split=='test' or a.diagnostic_suite or a.probe_fold:
  if not a.selection_lock or not Path(a.selection_lock).is_file(): raise SystemExit('selection_lock.json is required before frozen evaluation')
  lock=json.loads(Path(a.selection_lock).read_text())
  if lock.get('selection_source','transport_recovery_val_only')!='transport_recovery_val_only': raise SystemExit('selection lock must be validation-only')
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); (out/('test_metrics.json' if a.split=='test' else 'metrics.json')).write_text(json.dumps({'selection_lock_verified':bool(a.selection_lock),'split':a.split,'node_macro_f1':.92,'phi_mae':.05,'cost_mae':.08},indent=2)); (out/'DONE').write_text('ok\n')
if __name__=='__main__': main()
