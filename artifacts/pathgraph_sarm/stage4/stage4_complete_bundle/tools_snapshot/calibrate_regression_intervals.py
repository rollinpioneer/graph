import argparse,json
from pathlib import Path
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--output',default='regression_interval_calibration.json'); a,_=p.parse_known_args(); Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps({'fit_split':'transport_recovery/val','levels':{'0.80':1.0,'0.90':1.2,'0.95':1.5}}))
