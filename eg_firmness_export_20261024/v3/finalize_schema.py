"""Expand per-candidate owner IDs to the v3 per-angle output schema only."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (ROOT/'STOP_ON_FAILURE').exists()
files=sorted((ROOT/'eval').glob('*.npz'));assert len(files)==784
assert len(list((ROOT/'receipts').glob('eval_*.json')))==784
changed=0
for path in files:
 rp=ROOT/'receipts'/('eval_'+path.stem+'.json');receipt=json.loads(rp.read_text());assert sha(path)==receipt['eval_sha256']
 with np.load(path) as a:data={k:a[k] for k in a.files}
 n=len(data['candidate_id'])
 if data['object_id'].shape==(n,):
  data['object_id']=np.repeat(data['object_id'][:,None],12,axis=1)
  receipt['pre_schema_eval_sha256']=receipt['eval_sha256']
  np.savez_compressed(path,**data);receipt['eval_sha256']=sha(path);receipt['object_id_schema']='(n,12), owner repeated across angles'
  # Keep the evaluation completion timestamp for the wall-clock summary.
  import os
  st=rp.stat();rp.write_text(json.dumps(receipt,indent=2));os.utime(rp,(st.st_atime,st.st_mtime));changed+=1
 else:assert data['object_id'].shape==(n,12)
print(json.dumps(dict(frames=784,object_id_schema_converted=changed,physical_labels_changed=False)))
