from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]

def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def protocol(): return load_json(ROOT/'contracts/protocol.json')
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def value_hash(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def new_dir(path):
    p=Path(path);p.mkdir(parents=True,exist_ok=False);return p

def seed_for(*parts):
    # Stable across processes and methods, unlike Python hash().
    return int.from_bytes(hashlib.sha256(canonical(list(parts)).encode()).digest()[:8],'big')
def finite(x):
    y=float(x)
    if not math.isfinite(y):raise ValueError('nonfinite numeric input')
    return y
