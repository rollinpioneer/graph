"""Post hoc descriptive persistence from saved branches, with zero new simulation."""
from pathlib import Path
import json,numpy as np,hashlib
R=Path(__file__).resolve().parents[1]
rows=sum([json.loads((R/'reports'/f'branches_s{s}.json').read_text()) for s in [0,1]],[]);extra=[]
for x in rows:
 with np.load(x['path']) as z:o=z['observations'];cc=((o[:,:2]+6)/4).astype(int)[:,::-1];outside=~np.all(cc==[6,5],axis=1)
 changes=np.diff(np.r_[False,outside,False].astype(int));starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)
 extra.append(dict(state=x['state'],mode=x['mode'],candidate=x['candidate'],repeat=x['repeat'],outside_final=bool(outside[-1]),outside_frames=int(outside.sum()),longest_outside=int(max(ends-starts,default=0)),upright_east_exit=x['upright_east_exit'],first_exit=x['first_exit']))
(R/'reports/exploratory_exit_persistence.json').write_text(json.dumps(dict(note='Post hoc description only; does not replace frozen ever-exit outcome',rows=extra),indent=2))
index=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted((R/'states').glob('*.pkl'))]
(R/'receipts/snapshot_index.json').write_text(json.dumps(index,indent=2))
