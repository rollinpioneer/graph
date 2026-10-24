import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];name='eg_z0_prior_locality_20261024'
source=ROOT/'outputs'/name;dest=ROOT/'work/graph_archive'/name;dest.mkdir(exist_ok=True)
(dest/'.gitattributes').write_text('* -text\n',encoding='utf-8')
included=[];excluded=[]
for p in sorted(source.rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(source)
 if p.suffix not in {'.py','.json','.csv','.md','.png','.npz'} or p.stat().st_size>5_000_000:
  excluded.append(dict(path=rel.as_posix(),bytes=p.stat().st_size));continue
 target=dest/rel;target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,target)
 included.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
shutil.copy2(__file__,dest/'archive_z0.py')
(dest/'git_archive_manifest.json').write_text(json.dumps(dict(scope='Z0 CPU-only prior/locality comparison; no T stage; PRIOR unsupported, within LOCALITY tolerance',included=included,excluded=excluded,external_inputs='Historical angle and EG_cd NPZ files referenced by input_manifest.json; unchanged'),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(included),bytes=sum(x['bytes'] for x in included),excluded=len(excluded))))
