"""Isolated validation subprocess. Parent trainer state must be unchanged."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
from .io_utils import write_new
from .release import ReleaseRejected

def run_isolated_validation(*, checkpoint, data_root, out_dir, method, backend="fake"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-B", "-m", "p2crl.validation_worker",
        "--checkpoint", str(checkpoint),
        "--data-root", str(data_root or ""),
        "--out", str(out_dir),
        "--method", str(method or ""),
        "--backend", str(backend),
    ]
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    subprocess.check_call(cmd, env=env)
    return out_dir

def _worker(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.backend != "real":
        write_new(out / "summary.json", {
            "schema": "P2CRL_VAL_SUMMARY_V1",
            "backend": args.backend,
            "evaluated": 0,
            "note": "fake/isolated stub; no trainer mutation",
        })
        write_new(out / "episodes.csv", "family_id,success,steps\n")
        return
    from .data_access import contracts_for_split
    from .evaluate import evaluate_episode
    contracts = contracts_for_split(args.data_root, "validation")
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(args.checkpoint, device="cpu")
    rows = []
    for c in contracts:
        rec = evaluate_episode(model, c, deterministic=True)
        rows.append(rec)
    write_new(out / "summary.json", {
        "schema": "P2CRL_VAL_SUMMARY_V1",
        "backend": "real",
        "evaluated": len(rows),
        "success_mean": (sum(r["success"] for r in rows) / len(rows)) if rows else 0.0,
        "invalid_actions": sum(r["invalid_actions"] for r in rows),
        "nonfinite": sum(r["nonfinite"] for r in rows),
    })
    lines = ["family_id,success,steps,motif"]
    for r in rows:
        lines.append(f"{r['family_id']},{r['success']},{r['steps']},{r['motif']}")
    (out / "episodes.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--backend", default="fake")
    _worker(ap.parse_args(argv))

if __name__ == "__main__":
    main()
