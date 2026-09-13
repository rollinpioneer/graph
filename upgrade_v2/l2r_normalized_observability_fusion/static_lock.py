from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


COMPONENTS = (
    ("upgrade_v2/l2r_canonical_time_confirmation/guard.py", "clp3_guard"),
    ("upgrade_v2/l2r_canonical_time_confirmation/collector.py", "canonical_time_adapter"),
    ("upgrade_v2/visual_refine_l2/vision.py", "rgb_detector"),
    ("upgrade_v2/visual_refine_l2/renderer.py", "renderer"),
    ("upgrade_v2/l2r_rgb_temporal_confirmation/rgb_capture.py", "contact_proxy"),
    ("upgrade_v2/l2r_forced_drop/physical_reference.py", "physical_reference"),
    ("upgrade_v2/l2r_l3_closed_loop/runner.py", "closed_loop_controller"),
    ("upgrade_v2/l2r_recovery_supervisor/supervisor.py", "recovery_supervisor"),
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    components = []
    for relative, role in COMPONENTS:
        path = repo / relative
        blob = subprocess.check_output(["git", "-C", str(repo), "rev-parse", f"HEAD:{relative}"], text=True).strip()
        components.append({"path": relative, "role": role, "sha256": sha(path), "git_blob": blob, "mutable_in_R26": False})
    lock = {"schema": "l2rar2_r26_parent_component_lock_v1", "base_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(), "components": components}
    (args.output / "parent_component_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "runner_file_hashes.json").write_text(json.dumps({"schema": "l2rar2_r26_runner_file_hashes_v1", "runner_commit": lock["base_commit"], "files": [{"path": c["path"], "sha256": c["sha256"]} for c in components]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "component_paths.json").write_text(json.dumps({"schema": "l2rar2_r26_component_paths_v1", "components": components}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
