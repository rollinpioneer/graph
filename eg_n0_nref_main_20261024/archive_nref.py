import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];name='eg_n0_nref_main_20261024'
source=ROOT/'outputs'/name;dest=ROOT/'work/graph_archive'/name;dest.mkdir(exist_ok=True)
(dest/'.gitattributes').write_text('* -text\n',encoding='utf-8')
included=[];excluded=[]
for p in sorted(source.rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(source)
 if p.suffix not in {'.py','.sh','.json','.csv','.md','.png','.npz','.sha256'} or p.name in {'connection_cleanup.json','process_audit.json','git_delivery.json'} or p.stat().st_size>5_000_000:
  excluded.append(dict(path=rel.as_posix(),bytes=p.stat().st_size));continue
 target=dest/rel;target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,target)
 included.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
shutil.copy2(__file__,dest/'archive_nref.py')
(dest/'git_archive_manifest.json').write_text(json.dumps(dict(scope='Completed N0: continuous N_ref primary; 48 training calibration,48 C3,48 P2, dense audit and joint I1; STOP',included=included,excluded=excluded,external_assets=['original EG checkpoint and GraspNet data','new raw prediction caches','full raw-pool detector CPU intermediates','nvblox source and environment'],historical_results_unchanged=True),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(included),bytes=sum(x['bytes'] for x in included),excluded=len(excluded))))
