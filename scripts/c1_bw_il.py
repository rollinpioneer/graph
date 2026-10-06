#!/usr/bin/env python
"""CLI for CP-DISR-C1-BW-IMITATION-A02: prep | register | train | status. There is no default 'all' and no supervisor; every training is started explicitly.

    python scripts/c1_bw_il.py prep     --root .                      # D0 + frozen config (configs/c1_bw_imitation_a02.json); --verify recomputes and compares
    python scripts/c1_bw_il.py register --root . --prep-commit <sha> --authorization-text-file <file>
    python scripts/c1_bw_il.py train    --root . --run-root <dir> --run-id R-C1-BW-IL-B2-0 --gpu 0
    python scripts/c1_bw_il.py status   --run-root <dir>
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REUSED = ("src/cp_disr/blocksworld/state.py", "src/cp_disr/blocksworld/contracts.py", "src/cp_disr/blocksworld/planner.py", "src/cp_disr/blocksworld/canonical.py", "src/cp_disr/blocksworld/generator.py",
          "src/cp_disr/blocksworld/metrics.py", "src/cp_disr/blocksworld/environment.py", "src/cp_disr/blocksworld/splits.py", "src/cp_disr/blocksworld/train.py",
          "src/cp_disr/blocksworld/classification.py", "src/cp_disr/blocksworld/launch.py", "src/cp_disr/c1_blocksworld_policies.py", "src/cp_disr/c1_qmark_policy.py", "src/cp_disr/neural.py",
          "src/cp_disr/torch_rl.py", "src/cp_disr/rl.py", "scripts/c1_bw_build.py", "scripts/c1_bw_eval.py",
          "configs/splits/c1_bw_train_dev_v1.json", "configs/splits/c1_bw_a0_iso_v1.json", "configs/splits/c1_bw_a1_color_reverse_v1.json", "configs/splits/c1_bw_a2_noniso_v1.json",
          "configs/splits/c1_bw_b_scale_v1.json")
FREEZE_COMMIT = "490a3b11d49f6155baf2a6b6c430edac6609d2af"
BASE_COMMIT = "72c0fd2cdc4394dcdee3b5bd6bba210060535d57"
BRANCH = "codex/cp-disr-c1-blocksworld-main-v1"


def sha_bytes(b):
    import hashlib
    return hashlib.sha256(b).hexdigest()


def at_commit(root, commit, rel):
    return subprocess.check_output(["git", "-C", str(root), "show", "%s:%s" % (commit, rel)])


def build(root):
    from cp_disr.blocksworld import imitation as I
    from cp_disr.blocksworld import launch as L
    root = Path(root).resolve()
    cases, half_life = I.load_train_cases(root / L.TRAIN_SPLIT_REL)
    d0 = I.build_d0(cases)
    I.verify_labels(d0, cases)
    ident = I.dataset_identity(d0)
    reused = {}
    for rel in REUSED:
        now = sha_bytes((root / rel).read_bytes())
        then = sha_bytes(at_commit(root, FREEZE_COMMIT, rel))
        reused[rel] = {"sha256": now, "sha256_at_freeze_commit": then, "unchanged_since_freeze": now == then}
    doc = {"card": L.CARD, "amendment": "CP-DISR-C1-BW-IMITATION-A02", "base_commit": BASE_COMMIT, "branch": BRANCH, "freeze_commit": FREEZE_COMMIT, "ppo_state_preserved": "PPO_ID_GATE_FAIL",
           "runs": {"R-C1-BW-IL-B2-0": "B2-CACHED", "R-C1-BW-IL-QMARK-0": "B1-K+QMARK-BW", "R-C1-BW-IL-ASNET-0": "ASNET-READOUT"}, "seed": 0, "init": "FROM_SCRATCH",
           "loss": I.LOSS_NAME, "optimizer": I.OPT, "stages": [dict(s) for s in I.STAGES], "aggregation_rounds": 2, "trainable_modules": list(I.TRAINABLE_MODULES), "frozen_modules": list(I.FROZEN_MODULES),
           "decision_chunk": I.DECISION_CHUNK, "half_life": half_life, "d0_sha256": ident["sha256"], "d0_identity": ident, "reused_files": reused,
           "id_gate": {"success_min": 33, "decision_perfect_min": 33, "nan": 0, "hard_fail": 0, "n_dev": 36}, "authorized_training_runs": 3,
           "formal_eval_condition": "ALL_THREE_PASS_IMITATION_ID_GATE", "ppo_checkpoints_loaded": False}
    return doc


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--root", default=".")
    p.add_argument("--verify", action="store_true")
    r = sub.add_parser("register")
    r.add_argument("--root", default=".")
    r.add_argument("--prep-commit", required=True)
    r.add_argument("--authorization-text-file", required=True)
    r.add_argument("--stamp")
    t = sub.add_parser("train")
    t.add_argument("--root", default=".")
    t.add_argument("--run-root", required=True)
    t.add_argument("--run-id", required=True)
    t.add_argument("--gpu", type=int, required=True)
    s = sub.add_parser("status")
    s.add_argument("--run-root", required=True)
    a = ap.parse_args()
    if a.cmd == "train":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)            # must precede any torch import
    from cp_disr.blocksworld import imitation_launch as IL
    from cp_disr.blocksworld import launch as L
    if a.cmd == "prep":
        doc = build(a.root)
        path = Path(a.root).resolve() / IL.CONFIG_REL
        text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
        if a.verify:
            same = path.read_text(encoding="utf-8") == text
            print(json.dumps({"verdict": "PASS" if same else "FAIL"}))
            sys.exit(0 if same else 1)
        path.write_text(text, encoding="utf-8")
        print(json.dumps({"written": str(path), "d0_sha256": doc["d0_sha256"], "unchanged_reused": all(v["unchanged_since_freeze"] for v in doc["reused_files"].values())}))
    elif a.cmd == "register":
        run_root = IL.register(a.root, a.prep_commit, Path(a.authorization_text_file).read_text(encoding="utf-8"), a.stamp)
        print(json.dumps({"registered": True, "run_root": str(run_root)}))
    elif a.cmd == "train":
        try:
            print(json.dumps(IL.train(a.root, a.run_root, a.run_id, a.gpu), default=str))
        except L.LaunchError as exc:
            print(json.dumps({"status": "BLOCKED", "error": str(exc)}))
            sys.exit(2)
    else:
        print(json.dumps(L.status(a.run_root)))


if __name__ == "__main__":
    main()
