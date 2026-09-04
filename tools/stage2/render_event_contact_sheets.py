#!/usr/bin/env python3
import argparse
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest'); ap.add_argument('--events'); ap.add_argument('--selection'); ap.add_argument('--output-dir',required=True); ap.add_argument('--max-recovery',type=int); ap.add_argument('--max-failure',type=int); ap.add_argument('--random-success-per-task',type=int); a=ap.parse_args(); Path(a.output_dir).mkdir(parents=True,exist_ok=True); (Path(a.output_dir)/'README.md').write_text('# Contact sheets\n\nNo image renderer was available; state evidence remains in CSV/JSON.\n')
if __name__=='__main__': main()
