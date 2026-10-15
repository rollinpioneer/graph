#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task',required=True); ap.add_argument('--scenario',required=True); ap.add_argument('--num-episodes',type=int,required=True); ap.add_argument('--seed-start',type=int,required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--controller',default='scripted_oracle'); ap.add_argument('--checkpoint'); ap.add_argument('--save-full-history',action='store_true'); ap.add_argument('--save-intervention-log',action='store_true'); a=ap.parse_args(); Path(a.output_dir).mkdir(parents=True,exist_ok=True); Path(a.output_dir,'collection_job.json').write_text(json.dumps(vars(a),indent=2)+'\n'); print('transparent scripted-oracle collection manifest written')
if __name__=='__main__': main()
