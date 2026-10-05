#!/usr/bin/env python
"""CP-DISR-C1-MECH-CONFIRM-V1 evaluation entry: one method on one case list, decision-level evidence, resume-only (never a second formal evaluation of a case).

    python scripts/c1_eval.py --root . --gpu N --method B2 --checkpoint <final .pt> --split-file configs/splits/c1_fresh_confirm_v1_test.json --rows-key test \
        --out runs/final_master/c1_route_b/mech_confirm_v1/fresh_confirm/B2_seed0.jsonl --label fresh_confirm --model-seed 0 --formal --prep-commit <sha>
    python scripts/c1_eval.py ... --method B_PLAN       (no checkpoint)

`--formal` is required for the Fresh Confirm test file: it checks HEAD == prep commit with a clean tracked tree and that the file hash equals the frozen one.
Smoke / qualification runs use other row files and never touch the final file.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PREP = Path("runs/final_master/c1_route_b/mech_confirm_v1/prep")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--rows-key", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model-seed", type=int)
    ap.add_argument("--formal", action="store_true")
    ap.add_argument("--prep-commit")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    split_path = (root / a.split_file).resolve()
    from cp_disr import c1_fresh_confirm as F
    from cp_disr import final_tb as ftb
    F.install_goal_sets()
    final_hit = split_path == (root / F.TEST_REL).resolve()
    if final_hit and not a.formal:
        raise SystemExit("the Fresh Confirm final file requires --formal")
    if a.formal:
        frozen = json.loads((root / PREP / "split_manifest.json").read_text())["file_sha256"]
        rel = str(split_path.relative_to(root))
        if frozen.get(rel) != sha(split_path):
            raise SystemExit("split file hash differs from the frozen hash")
        if not a.prep_commit:
            raise SystemExit("--formal needs --prep-commit")
        ftb.check_source_identity(root, a.prep_commit)
    if a.method != "B_PLAN" and not a.checkpoint:
        raise SystemExit("neural methods need --checkpoint")
    ftb.bind_worker_gpu(a.gpu)
    import torch
    from cp_disr import c1_confirm_runtime as R
    from cp_disr import final_tb_c1 as C1
    from cp_disr import struct_gen as G
    os.chdir(str(root))
    doc = json.loads(split_path.read_text())
    rows = list(doc[a.rows_key])
    if a.limit:
        rows = rows[: a.limit]
    out_path = (root / a.out).resolve() if not Path(a.out).is_absolute() else Path(a.out)
    work = out_path.parent / ("_work_" + out_path.stem)
    work.mkdir(parents=True, exist_ok=True)
    tmp_split = work / "eval_split.json"
    G.write_json(tmp_split, {"task_id": "T_B", "train": [], "dev": rows, "test": [], "note": "evaluation rows only (derived from %s)" % split_path.name})
    manifest = ftb.derive_runtime_manifest(root, work / "runtime_manifest_tb_resolved.yaml", str(tmp_split))
    uuid = ftb.query_gpu_uuid(a.gpu)
    head = ftb.git_out(root, "rev-parse", "HEAD")
    ctx = ftb.RunContext(plan_id="C1EVAL", attempt_id="C1EVAL", method="B2", training_seed=0, source_commit=head, output_directory=str(work / "unused_run_dir"), runtime_manifest=manifest["path"],
                         train_split=str(tmp_split), physical_gpu_index=a.gpu, render_gpu_device_id=a.gpu, gpu_uuid=uuid)
    v11 = C1.configure_v11(root, ctx)
    v11.ENABLED_SPLITS[ftb.TASK] = tmp_split
    v11.bind_H(root)
    device = torch.device("cuda", 0)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ck = Path(a.checkpoint).resolve() if a.checkpoint else None
    meta = {"method": a.method, "model_seed": a.model_seed, "label": a.label, "split_file": str(split_path.relative_to(root)), "split_sha256": sha(split_path), "checkpoint": str(ck) if ck else None,
            "checkpoint_sha256": sha(ck) if ck else None, "head": head, "gpu": a.gpu, "started": started, "formal": bool(a.formal), "eval_action": "deterministic_argmax", "optimizer_steps": 0, "n_rows": len(rows)}
    (out_path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    results = R.run_cases(root, v11, a.method, rows, device, out_path, checkpoint=ck, label=a.label, model_seed=a.model_seed)
    done = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    summary = {"method": a.method, "n": len(done), "success_n": sum(1 for d in done if d["success"]), "by_cell": {}}
    for d in done:
        c = summary["by_cell"].setdefault(d["cell"], [0, 0])
        c[1] += 1
        c[0] += int(d["success"])
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
