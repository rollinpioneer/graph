#!/usr/bin/env python3
import argparse,csv
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--queue',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--annotation-dir'); ap.add_argument('--workers',type=int,default=1); a=ap.parse_args()
    for r in csv.DictReader(open(a.queue)):
        d=Path(a.output_dir)/r['episode_id']; d.mkdir(parents=True,exist_ok=True); (d/'review_notes.md').write_text('# Review notes\n\nState trace and proposal are retained in the parent artifact.\n')
if __name__=='__main__': main()
