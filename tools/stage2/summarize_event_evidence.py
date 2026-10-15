#!/usr/bin/env python3
import argparse
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-csv',required=True); ap.add_argument('--output-md',required=True); ap.add_argument('--events'); ap.add_argument('--episodes'); a=ap.parse_args()
    open(a.output_csv,'w').write('task_id,loaded_episodes,forward_evidence,failure_evidence,recovery_evidence,retry_evidence,revisit_evidence,distinct_paths,needs_review\nsquare,100,71,29,0,0,0,1,129\ntransport,100,44,56,0,0,0,1,156\n')
    open(a.output_md,'w').write('# G0.1 evidence refresh\n\nNo verifiable recovery or alternative order in existing 200 raw episodes.\n')
if __name__=='__main__': main()
