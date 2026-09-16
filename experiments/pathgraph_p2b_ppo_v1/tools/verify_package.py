from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    m=json.loads((root/'PACKAGE_MANIFEST.json').read_text());bad=[];expected=set()
    for r in m['files']:
        name=r['path'];q=(root/name).resolve();expected.add(name)
        if not q.is_relative_to(root) or not q.is_file():bad.append([name,'missing/unsafe']);continue
        if q.stat().st_size!=r['size_bytes'] or hashlib.sha256(q.read_bytes()).hexdigest()!=r['sha256']:bad.append([name,'bytes changed'])
    actual={x.relative_to(root).as_posix() for x in root.rglob('*') if x.is_file() and x.name!='PACKAGE_MANIFEST.json' and '__pycache__' not in x.parts}
    if actual!=expected:bad.append(['file_set',sorted(actual-expected),sorted(expected-actual)])
    print(json.dumps({'status':'PASS' if not bad else 'FAIL','checked':len(expected),'errors':bad},indent=2))
    return bool(bad)
if __name__=='__main__':raise SystemExit(main())
