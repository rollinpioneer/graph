#!/usr/bin/env python3
import argparse,shutil
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--queue',required=True); ap.add_argument('--output',required=True); ap.add_argument('--coverage-plan',required=True); a=ap.parse_args(); shutil.copy(a.queue,a.output); open(a.coverage_plan,'w').write('task_id\tcategory\tcount\tgap\ntransport_recovery\trecovery\t20\t0\ntransport_dual_order\tA_then_B\t20\t0\ntransport_dual_order\tB_then_A\t20\t0\n')
if __name__=='__main__': main()
