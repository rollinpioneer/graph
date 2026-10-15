#!/usr/bin/env python3
import argparse,json,glob
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--annotation-dir',required=True); ap.add_argument('--graph-dir',required=True); ap.add_argument('--schema'); ap.add_argument('--report'); a=ap.parse_args(); n=0
    for p in glob.glob(a.annotation_dir+'/*.json'):
        x=json.load(open(p)); assert x['review']['status'] in {'accepted','edited','ambiguous'}
        for q in x['progress_anchors']: assert 0<=float(q['value'])<=1
        n+=1
    out={'valid':True,'annotations':n}; print(json.dumps(out))
    if a.report: open(a.report,'w').write(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
