#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.root/'PACKAGE_MANIFEST.json').read_text());errors=[]
    for r in m['files']:
        f=a.root/r['path']
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=r['sha256']:errors.append(r['path'])
    print(json.dumps(dict(status='PASS' if not errors else 'FAIL',checked=len(m['files']),errors=errors),indent=2))
    return int(bool(errors))
if __name__=='__main__':raise SystemExit(main())
