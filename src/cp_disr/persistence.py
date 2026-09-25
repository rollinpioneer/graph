from __future__ import annotations
import csv, hashlib, json, os, shutil, tempfile, uuid
from pathlib import Path
from typing import Any, Mapping

class PersistenceError(RuntimeError):
    pass

def _fsync_dir(path: Path) -> None:
    fd=os.open(str(path),os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True)+"\n").encode()

def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def _write_bytes(path: Path,data: bytes) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
    except Exception:
        try: path.unlink()
        except FileNotFoundError: pass
        raise

def _materialize(value: Any) -> bytes:
    if isinstance(value,bytes): return value
    if isinstance(value,str): return value.encode()
    if isinstance(value,Path): return value.read_bytes()
    return _json_bytes(value)

class GenerationStore:
    """Crash-safe immutable checkpoint generation publisher."""
    def __init__(self,root: str|Path):
        self.root=Path(root); self.generations=self.root/"generations"
        self.events=self.root/"raw_events.jsonl"; self.metrics_csv=self.root/"metrics.csv"
        self.generations.mkdir(parents=True,exist_ok=True); _fsync_dir(self.generations)

    def append_event(self,event: Mapping[str,Any]) -> None:
        self.root.mkdir(parents=True,exist_ok=True)
        fd=os.open(str(self.events),os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o644)
        with os.fdopen(fd,"ab") as f:
            f.write(_json_bytes(dict(event))); f.flush(); os.fsync(f.fileno())
        _fsync_dir(self.root)

    def derive_csv(self) -> Path:
        rows=[]
        if self.events.exists():
            with self.events.open() as f: rows=[json.loads(x) for x in f if x.strip()]
        keys=sorted({k for row in rows for k in row}) or ["event"]
        tmp=self.metrics_csv.with_name(self.metrics_csv.name+".tmp-"+uuid.uuid4().hex)
        with tmp.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
            for row in rows: w.writerow({k:row.get(k) for k in keys})
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,self.metrics_csv); _fsync_dir(self.root); return self.metrics_csv

    def publish(self,files: Mapping[str,Any],manifest: Mapping[str,Any],*,generation: str|None=None,fail_at: str|None=None) -> str:
        name=generation or ("update_%s_%s"%(manifest.get("update","unknown"),uuid.uuid4().hex[:12]))
        if "/" in name or name in {"",".",".."}: raise ValueError("invalid generation name")
        tmp=Path(tempfile.mkdtemp(prefix=".generation-",dir=str(self.generations))); final=self.generations/name
        try:
            if fail_at=="before_files": raise OSError("injected before_files")
            hashes={}
            for rel,value in files.items():
                data=_materialize(value); _write_bytes(tmp/rel,data); hashes[rel]=hashlib.sha256(data).hexdigest()
                if fail_at=="after_file:"+rel: raise OSError("injected after_file:"+rel)
            record={"generation":name,"manifest":dict(manifest),"files":hashes}
            _write_bytes(tmp/"HASHES.json",_json_bytes(record))
            if fail_at=="before_complete": raise OSError("injected before_complete")
            _write_bytes(tmp/"COMPLETE",_json_bytes({"generation":name,"hashes_sha256":_sha256(tmp/"HASHES.json")})); _fsync_dir(tmp)
            if fail_at=="before_rename": raise OSError("injected before_rename")
            os.rename(tmp,final); _fsync_dir(self.generations)
            latest_tmp=self.root/(".LATEST.tmp-"+uuid.uuid4().hex); _write_bytes(latest_tmp,(name+"\n").encode())
            if fail_at=="before_latest": raise OSError("injected before_latest")
            os.replace(latest_tmp,self.root/"LATEST"); _fsync_dir(self.root); return name
        except Exception:
            shutil.rmtree(tmp,ignore_errors=True)
            if final.exists() and (not (self.root/"LATEST").exists() or (self.root/"LATEST").read_text().strip()!=name):
                shutil.rmtree(final,ignore_errors=True)
            raise

    def verify(self,name: str) -> dict[str,Any]:
        path=self.generations/name
        if not path.is_dir() or not (path/"COMPLETE").is_file(): raise PersistenceError("generation is not COMPLETE: "+name)
        record=json.loads((path/"HASHES.json").read_text())
        if record.get("generation")!=name: raise PersistenceError("generation name mismatch")
        for rel,expected in record.get("files",{}).items():
            target=path/rel
            if not target.is_file() or _sha256(target)!=expected: raise PersistenceError("hash mismatch: "+rel)
        return record

    def latest(self) -> str:
        name=(self.root/"LATEST").read_text().strip(); self.verify(name); return name

    def load(self,name: str|None=None) -> dict[str,Any]:
        selected=name or self.latest(); record=self.verify(selected)
        return {"generation":selected,"manifest":record["manifest"],"path":str(self.generations/selected),"files":record["files"]}
