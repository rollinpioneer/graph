#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.stage2.stage2_pipeline import mine_actual
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--field-map'); ap.add_argument('--output-dir',required=True); ap.add_argument('--workers',type=int,default=1); ap.add_argument('--task'); ap.add_argument('--episode-list'); a=ap.parse_args(); mine_actual(Path(a.output_dir)); print('semantic event mining complete')
if __name__=='__main__': main()
