from __future__ import annotations
import csv, json, math, pickle, hashlib, random, datetime as dt
from pathlib import Path
from collections import defaultdict
import numpy as np

class _Dummy:
    def __new__(cls,*a,**kw):
        o=object.__new__(cls); o.newargs=a; return o
    def __init__(self,*a,**kw): self.initargs=a
    def __setstate__(self,state): self.state=state

def _unpickle_block(*args): return ("BLOCK", args)

class _U(pickle.Unpickler):
    def find_class(self,module,name):
        if module.startswith("pandas"):
            return _unpickle_block if name == "_unpickle_block" else _Dummy
        return super().find_class(module,name)

def _state(x):
    if hasattr(x,"state") and isinstance(x.state,dict): return x.state
    if hasattr(x,"newargs") and len(x.newargs)>1 and isinstance(x.newargs[1],dict): return x.newargs[1]
    return {}

def load_pickle_table(path: str|Path) -> dict[str,list]:
    with open(path,"rb") as f: obj=_U(f).load()
    if not hasattr(obj,"state") or "_mgr" not in obj.state:
        return {"_object": [obj]}
    mgr=obj.state["_mgr"]; args=mgr.newargs
    blocks, axes=args[0], args[1]
    cols=np.asarray(_state(axes[0]).get("data",[]),dtype=object).tolist()
    idx=_state(axes[1]); start=int(idx.get("start",0)); stop=int(idx.get("stop",0)); step=int(idx.get("step",1) or 1)
    n=max(0,(stop-start + (step-1))//step)
    out={str(c):[None]*n for c in cols}
    for block in blocks:
        arr,sl,_ = block[1]
        for j,cidx in enumerate(range(int(sl.start),int(sl.stop))):
            vals=arr[j]
            out[str(cols[cidx])]=list(vals)
    return out

def flatten_obs(table):
    obs=table.get("obs",[]); out=[]
    for x in obs:
        a=np.asarray(x)
        if a.ndim==1: out.append(a.astype(float))
        elif a.ndim>=2: out.append(a[-1].astype(float))
    return np.asarray(out,dtype=float) if out else np.empty((0,0))

def flatten_action(table):
    a=table.get("action",[]); out=[]
    for x in a:
        z=np.asarray(x)
        if z.ndim==1: out.append(z.astype(float))
        elif z.ndim>=2: out.append(z[0].astype(float))
    return np.asarray(out,dtype=float) if out else np.empty((0,0))

def robust_num(x):
    try: return float(x)
    except Exception: return None

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def read_jsonl(path):
    with open(path,encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]

def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()

def write_manifest(path, round_id, purpose, command, extra=None):
    lines=["# Run Manifest","",f"- round_id: {round_id}",f"- purpose: {purpose}",f"- started_at: {now()}","- finished_at: "+now(),"- repo_root: /home/xushijie/CUPID","- git_commit: unavailable (CUPID root is not a git worktree)","- python: python 3.x","- gpu_ids: none (CPU-only evidence extraction and deterministic scripted collection)",f"- command: {command}"]
    for k,v in (extra or {}).items(): lines.append(f"- {k}: {v}")
    Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8")

def csv_write(path, fields, rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def split_rows(rows, seed=20260831):
    groups=defaultdict(list)
    for r in rows: groups[str(r.get("group_id") or r.get("episode_id"))].append(r)
    keys=sorted(groups); random.Random(seed).shuffle(keys)
    n=len(keys); nt=round(n*.6); nv=round(n*.2)
    out=[]
    for i,k in enumerate(keys):
        sp="train" if i<nt else "val" if i<nt+nv else "test"
        for r in groups[k]:
            q=dict(r); q["split"]=sp; out.append(q)
    return out
