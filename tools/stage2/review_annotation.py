#!/usr/bin/env python3
import argparse,json,datetime
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--annotation',required=True); ap.add_argument('--status',choices=['accepted','edited','ambiguous']); ap.add_argument('--note',default=''); a=ap.parse_args(); x=json.load(open(a.annotation)); old=x.get('review',{}).copy(); x['review']={'status':a.status or old.get('status','edited'),'reviewer':'cli','notes':a.note or old.get('notes','')}; open(a.annotation,'w').write(json.dumps(x,indent=2)+'\n'); open(a.annotation+'.history.jsonl','a').write(json.dumps({'timestamp':datetime.datetime.now().isoformat(),'review':x['review']})+'\n')
if __name__=='__main__': main()
