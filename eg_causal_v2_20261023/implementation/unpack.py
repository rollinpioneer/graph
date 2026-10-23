import sys,json,base64,tarfile,hashlib
from pathlib import Path
W=Path(__file__).parent; O=W.parents[1]/'outputs/eg_causal_v2_20261023'
n=sys.argv[1]; q=W.parent/'ssh_queue'
p=next(q.glob(n+'-*.result.json'));r=json.loads(p.read_text(encoding='utf-8-sig'))[0]
assert r['exit_code']==0 and not r.get('stderr'),r.get('stderr')
b=base64.b64decode(r['stdout']);dest=O/f'download_{n}.tgz';dest.write_bytes(b)
with tarfile.open(dest) as t:
 for m in t.getmembers():
  target=(O/'data'/m.name).resolve();assert target.is_relative_to((O/'data').resolve())
 t.extractall(O/'data',filter='data')
print(json.dumps({'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'output':str(O/'data')}))
