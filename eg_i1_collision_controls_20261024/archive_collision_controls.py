"""Lightweight archive of the completed fixed O-F comparison."""
import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
name='eg_i1_collision_controls_20261024'
source=ROOT/'outputs'/name;dest=ROOT/'work/graph_archive'/name
dest.mkdir(exist_ok=True)
(dest/'.gitattributes').write_text('# Preserve bytes used in scientific SHA-256 manifests.\n* -text\n',encoding='utf-8')
allow={'.py','.sh','.json','.csv','.md','.png','.npz'}
excluded_names={'bridge_pid.txt','connection_cleanup.json','process_audit.json','execution_commands.json','analysis_console.txt'}
included=[];excluded=[]
for p in sorted(source.rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(source);reason=None
 if p.name in excluded_names or p.suffix=='.log':reason='transient host/session or console log'
 elif p.suffix not in allow or '__pycache__' in p.parts:reason='not a lightweight scientific deliverable'
 elif p.stat().st_size>5_000_000:reason='over 5 MB per file'
 if reason:excluded.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,reason=reason));continue
 target=dest/rel;target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,target)
 included.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
shutil.copy2(Path(__file__),dest/'archive_collision_controls.py')
(dest/'git_archive_manifest.json').write_text(json.dumps(dict(scope='Completed O-F zero-training controls and limited geometric attribution; HOLD; no new force-closure evaluation',included=included,excluded=excluded),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(included),bytes=sum(r['bytes'] for r in included),excluded=len(excluded)),indent=2))
