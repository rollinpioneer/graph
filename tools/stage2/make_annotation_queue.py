#!/usr/bin/env python3
import argparse,csv
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); a=ap.parse_args()
    rows=[]
    for i in range(108):
        rows.append({'queue_id':f'q_{i:04d}','task_id':'transport_recovery' if i<52 else 'transport_dual_order','episode_id':f'episode_{i:04d}','split':'train','source_path':'scripted_oracle','category':'recovery' if i%5==0 else 'forward_success','path_signature':'','priority':1,'proposal_path':'','review_bundle_path':'','status':'proposed'})
    with open(a.output,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if __name__=='__main__': main()
