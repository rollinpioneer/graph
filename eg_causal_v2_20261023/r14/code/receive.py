"""Receive a completed archive over the already authenticated SSH command queue."""
import sys,json,base64,hashlib,tarfile
from pathlib import Path
W=Path(__file__).parent;Q=W.parents[1]/'ssh_queue';O=W.parents[2]/'outputs/eg_causal_v2_20261023/r14'
number=sys.argv[1];p=next(Q.glob(number+'-*.result.json'));r=json.loads(p.read_text(encoding='utf-8-sig'))[0]
assert r['exit_code']==0 and not r.get('stderr'),r.get('stderr')
b=base64.b64decode(r['stdout']);archive=O/f'download_{number}.tgz';archive.write_bytes(b)
target=O/'data';target.mkdir(exist_ok=True)
with tarfile.open(archive) as t:
 for m in t.getmembers():assert (target/m.name).resolve().is_relative_to(target.resolve())
 t.extractall(target,filter='data')
print(json.dumps({'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'destination':str(target)}))
