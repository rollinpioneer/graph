"""Copy light campaign summaries into the repo artifact tree."""
from __future__ import annotations
from pathlib import Path
from .io_utils import load_json, sha256_file, write_new

def package_runner_artifacts(src, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in Path(src).glob("*"):
        if p.is_file() and p.suffix in {".json", ".csv", ".md", ".tsv"}:
            target = dest / p.name
            if not target.exists():
                if p.suffix == ".json":
                    write_new(target, load_json(p))
                else:
                    target.write_bytes(p.read_bytes())
            copied.append({"path": p.name, "sha256": sha256_file(target)})
    write_new(dest / "result_manifest.json", {
        "schema": "P2CRL_RUNNER_RESULT_MANIFEST_V1",
        "files": copied,
    })
    return dest
