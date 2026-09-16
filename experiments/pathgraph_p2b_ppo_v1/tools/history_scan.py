"""Scan committed metadata at BASE, not new package IDs against themselves."""
from __future__ import annotations
import argparse,json,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p2b.io_utils import protocol,dump

def scan_numbers(obj,key=''):
    out=set()
    if isinstance(obj,dict):
        for k,v in obj.items():out|=scan_numbers(v,k)
    elif isinstance(obj,list):
        for v in obj:out|=scan_numbers(v,key)
    elif any(t in key.lower() for t in ('family','families')):
        if isinstance(obj,int):out.add(obj)
        elif isinstance(obj,str):out.update(int(x) for x in re.findall(r'\d+',obj))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();c=protocol();base=c['base_commit']
    names=subprocess.check_output(['git','-C',a.repo,'ls-tree','-r','--name-only',base],text=True).splitlines()
    names=[n for n in names if n.endswith('.json') and any(x in n.lower() for x in ('registr','split','protocol','manifest'))]
    observed=set();read=[];unreadable=[]
    for n in names:
        try:
            b=subprocess.check_output(['git','-C',a.repo,'show',f'{base}:{n}'])
            if len(b)>8_000_000:unreadable.append({'path':n,'reason':'oversize'});continue
            obj=json.loads(b);observed|=scan_numbers(obj);read.append(n)
        except (ValueError,subprocess.CalledProcessError) as e:unreadable.append({'path':n,'reason':type(e).__name__})
    proposed={x for v in c['families'].values() for x in v};overlap=sorted(observed&proposed)
    dump(a.out,{'scope':'COMMITTED_METADATA_AT_PINNED_BASE','base':base,'paths_read':read,
        'paths_unreadable':unreadable,'historical_family_values':len(observed),
        'collisions':overlap,'external_raw_not_scanned':True,
        'status':'FAIL_COLLISION' if overlap else 'PASS_WITH_DECLARED_SCAN_SCOPE'})
    if overlap:raise SystemExit(2)
if __name__=='__main__':main()
