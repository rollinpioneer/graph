#!/usr/bin/env python3
import argparse,csv,yaml
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--check-only',action='store_true'); ap.add_argument('--selection',required=True); ap.add_argument('--evidence',required=True); a=ap.parse_args()
    sel=yaml.safe_load(open(a.selection)); rows=list(csv.DictReader(open(a.evidence)))
    assert len(sel.get('graph_tasks',[]))>=2
    assert all(str(r.get('graph_valid')).lower()=='true' for r in rows)
    assert any(int(r.get('alternative_path_1',0))>=10 and int(r.get('alternative_path_2',0))>=10 for r in rows)
    assert any(int(r.get('recovery_episodes',0))>=10 for r in rows)
    print('stage2 task selection check: PASS')
if __name__=='__main__': main()
