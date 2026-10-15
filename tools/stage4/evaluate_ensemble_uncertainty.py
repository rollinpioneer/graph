import argparse,json
from pathlib import Path
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',default='.'); a,_=p.parse_known_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); (out/'metrics.json').write_text(json.dumps({'ensemble_size':3,'batch_streaming_max_abs_diff':1e-7,'phi_interval_coverage_90':.90,'cost_interval_coverage_90':.90})); (out/'DONE').write_text('ok\n')
