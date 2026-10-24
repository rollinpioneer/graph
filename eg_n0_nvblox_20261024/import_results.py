import base64,io,json,tarfile,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
q=R.parents[1]/'work/ssh_queue/397-n0_collect.result.json'
items=json.loads(q.read_text());assert len(items)==1 and items[0]['exit_code']==0,items[0].get('stderr')
data=base64.b64decode(items[0]['stdout'].strip())
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as t:
 for m in t.getmembers():
  p=(R/m.name).resolve();assert p.is_relative_to(R.resolve()) and m.isfile(),m.name
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
manifest=json.loads((R/'export_manifest.json').read_text())
for n,h in manifest.items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
print('Verified',len(manifest),'exported evidence files')
