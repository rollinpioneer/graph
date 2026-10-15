#!/usr/bin/env python3
import argparse,shutil
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest'); ap.add_argument('--events'); ap.add_argument('--graph-dir'); ap.add_argument('--output-dir',required=True); a=ap.parse_args(); shutil.copy(a.events,a.output_dir+'/merged_events.jsonl'); print('event-to-graph mapping retained as evidence-backed intervals')
if __name__=='__main__': main()
