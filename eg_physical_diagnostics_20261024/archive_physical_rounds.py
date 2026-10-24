"""Archive the four completed EG physical diagnostics, excluding heavy assets."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'work/graph_archive'
DEST=REPO/'eg_physical_diagnostics_20261024'
ROUNDS=['eg_bad_cases_20261023','eg_pcr_r0_20261024','eg_pcr_decode_zero_20261024','eg_i1_angle_physics_20261024']
allowed={'.py','.sh','.json','.csv','.md','.png','.txt','.yaml','.yml','.npz','.npy'}
excluded_names={'bridge_pid.txt','connection_cleanup.json','process_audit.json','remote_process_audit.json','execution_commands.json'}
included=[];excluded=[]
for name in ROUNDS:
 for p in sorted((ROOT/'outputs'/name).rglob('*')):
  if not p.is_file():continue
  rel=Path(name)/p.relative_to(ROOT/'outputs'/name);reason=None
  if any(x in p.parts for x in ['.git','__pycache__']):reason='runtime metadata'
  elif p.name in excluded_names or p.suffix=='.log':reason='ephemeral execution log or host session metadata'
  elif p.suffix not in allowed:reason='model weight, download archive, or other non-deliverable asset'
  elif p.stat().st_size>5_000_000:reason='individual file exceeds lightweight 5 MB limit'
  if reason:
   excluded.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,reason=reason));continue
  target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  included.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
DEST.mkdir(exist_ok=True)
(DEST/'git_archive_manifest.json').write_text(json.dumps(dict(scope='Four completed diagnostic rounds. This manifest defines the uploaded subset; each original delivery_manifest refers to its complete local round.',included=included,excluded=excluded),ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(ROOT/'AGENTS.md',REPO/'AGENTS.md')
shutil.copy2(Path(__file__),DEST/'archive_physical_rounds.py')
print(json.dumps(dict(included=len(included),bytes=sum(x['bytes'] for x in included),excluded=len(excluded)),indent=2))
