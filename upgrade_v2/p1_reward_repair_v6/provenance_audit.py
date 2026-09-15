"""Audit that confirmation/engineering raw did not teleport objects or restore checkpoints."""
from __future__ import annotations
import json
from pathlib import Path
from .util import write_json


def audit_episode(dest: Path) -> dict:
    ident = json.loads((dest/"identity.json").read_text(encoding="utf-8"))
    manifest = json.loads((dest/"manifest.json").read_text(encoding="utf-8")) if (dest/"manifest.json").is_file() else {}
    mutations = []
    if (dest/"state_mutation_ledger.jsonl").is_file():
        mutations = [json.loads(l) for l in (dest/"state_mutation_ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    after = [m for m in mutations if m.get("when") != "reset_episode"]
    ckpt_restores = [m for m in mutations if "restore" in str(m.get("kind", "")).lower()]
    return {
        "episode_id": ident.get("episode_id"),
        "direct_pose_overwrites_after_start": manifest.get("direct_pose_overwrites_after_start"),
        "checkpoint_resets_inside_episode": manifest.get("checkpoint_resets_inside_episode"),
        "velocity_zeroing_after_start": manifest.get("velocity_zeroing_after_start"),
        "post_start_mutations": len(after),
        "checkpoint_restore_mutations": len(ckpt_restores),
        "passed": (manifest.get("direct_pose_overwrites_after_start")==0 and
                   manifest.get("checkpoint_resets_inside_episode")==0 and
                   manifest.get("velocity_zeroing_after_start")==0 and
                   len(after)==0 and len(ckpt_restores)==0),
        "raw": str(dest),
        "manifest_sha256_files": list((manifest.get("files") or {}).keys()),
    }


def audit_tree(raw_root: Path, out: Path) -> dict:
    dests = sorted({p.parent for p in Path(raw_root).rglob("identity.json")})
    rows = []
    for d in dests:
        if not (d/"manifest.json").is_file():
            continue
        rows.append(audit_episode(d))
    summary = {
        "episodes": len(rows),
        "passed": all(r["passed"] for r in rows) if rows else False,
        "failures": [r for r in rows if not r["passed"]],
        "rows": rows,
    }
    write_json(Path(out)/"evidence_provenance.json", summary)
    return summary
