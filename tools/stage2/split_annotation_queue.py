#!/usr/bin/env python3
import argparse,csv
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--queue',required=True); ap.add_argument('--num-shards',type=int,required=True); ap.add_argument('--output-dir',required=True); a=ap.parse_args(); rows=list(csv.DictReader(open(a.queue))); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    for i in range(a.num_shards):
        with open(out/f'shard_{i:02d}.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows[i::a.num_shards])
if __name__=='__main__': main()
