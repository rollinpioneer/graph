import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
name='eg_n0_nvblox_20261024'
source=ROOT/'outputs'/name;dest=ROOT/'work/graph_archive'/name
dest.mkdir(exist_ok=True)
(dest/'.gitattributes').write_text('* -text\n',encoding='utf-8')
included=[];excluded=[]
for p in sorted(source.rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(source)
 if p.suffix not in {'.py','.sh','.json','.csv','.md','.png'} or p.name in {'connection_cleanup.json','process_audit.json'} or p.stat().st_size>5_000_000:
  excluded.append(dict(path=rel.as_posix(),bytes=p.stat().st_size));continue
 target=dest/rel;target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,target)
 included.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
shutil.copy2(__file__,dest/'archive_n0.py')
(dest/'git_archive_manifest.json').write_text(json.dumps(dict(scope='N0 two-training-frame preflight, HOLD under unchanged 99% continuous-reference gate; no calibration/C3/P2/joint I1 experiments',included=included,excluded=excluded,external_large_assets=['official nvblox checkout','N0 virtual environment','EG checkpoint and two inference caches']),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(included),bytes=sum(x['bytes'] for x in included),excluded=len(excluded))))
