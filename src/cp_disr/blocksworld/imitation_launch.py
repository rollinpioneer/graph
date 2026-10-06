"""CP-DISR-C1-BW-IMITATION-A02 registration, ledger and training launch. Import-light: torch is imported only when a run starts.

Layout of the (new, separate) result root, never the PPO result root:
    runs/final_master/c1_route_b/blocksworld_main_v1/imitation_a02/<UTC>_<prep_sha8>/{registration,prep,datasets,runs,eval,results,receipts}
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import yaml

from . import launch as L

CARD = L.CARD
AMENDMENT = "CP-DISR-C1-BW-IMITATION-A02"
IL_OUT_REL = L.OUT_ROOT_REL / "imitation_a02"
CONFIG_REL = "configs/c1_bw_imitation_a02.json"
DOC_REL = "docs/c1_blocksworld/CP_DISR_C1_Blocksworld_Uniform_Planner_Imitation_Amendment_A02.md"
RUN_IDS = ("R-C1-BW-IL-B2-0", "R-C1-BW-IL-QMARK-0", "R-C1-BW-IL-ASNET-0")
METHOD_OF = {"R-C1-BW-IL-B2-0": "B2-CACHED", "R-C1-BW-IL-QMARK-0": "B1-K+QMARK-BW", "R-C1-BW-IL-ASNET-0": "ASNET-READOUT"}
PPO_ROOT_GLOB = "blocksworld_main_v1/2026"
RECORDS = ("AMENDMENT_A02_ACCEPTED", "PPO_RESULT_STATE=PPO_ID_GATE_FAIL", "NEW_TRAINING_PROTOCOL=UNIFORM_PLANNER_IMITATION", "AUTHORIZED_TRAINING_RUNS=3",
           "FORMAL_EVAL_CONDITION=ALL_THREE_PASS_IMITATION_ID_GATE")


def check_source_identity_il(root, prep_commit, exact):
    """Training: HEAD must equal the registered prep commit and the tracked tree must be clean. After training (evaluation): the prep commit must be an ancestor of HEAD and the
    commits since then may only add result files under ``runs/``; the tracked tree must be clean."""
    head = L.git(root, "rev-parse", "HEAD")
    if L.git(root, "status", "--porcelain", "--untracked-files=no"):
        raise L.LaunchError("source drift: tracked tree is dirty")
    if head == prep_commit:
        return head
    if exact:
        raise L.LaunchError("source drift: HEAD %s != registered prep commit %s" % (head, prep_commit))
    try:
        L.git(root, "merge-base", "--is-ancestor", prep_commit, head)
    except Exception:
        raise L.LaunchError("source drift: the prep commit is not an ancestor of HEAD")
    changed = [p for p in L.git(root, "diff", "--name-only", prep_commit, head).splitlines() if p]
    bad = [p for p in changed if not p.startswith(str(L.OUT_ROOT_REL) + "/") and not p.startswith("docs/")]
    if bad:
        raise L.LaunchError("source drift: non-result files changed since the prep commit: %s" % bad[:5])
    return head


def frozen_config(root):
    return json.loads((Path(root) / CONFIG_REL).read_text(encoding="utf-8"))


def register(root, prep_commit, authorization_text, stamp=None):
    from .imitation import (STAGES, OPT, LOSS_NAME, build_d0, dataset_identity, load_train_cases, validate_identical_supervision, verify_labels, write_json_atomic, sha256_file)
    root = Path(root).resolve()
    check_source_identity_il(root, prep_commit, exact=True)
    fz = frozen_config(root)
    split = root / L.TRAIN_SPLIT_REL
    cases, half_life = load_train_cases(split)
    stamp = stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_root = root / IL_OUT_REL / ("%s_%s" % (stamp, prep_commit[:8]))
    if run_root.exists():
        raise L.LaunchError("already registered: %s" % run_root)
    for sub in ("registration", "prep", "datasets", "runs", "eval", "results", "receipts"):
        (run_root / sub).mkdir(parents=True)
    d0 = build_d0(cases)
    ident = dataset_identity(d0)
    if ident["sha256"] != fz["d0_sha256"]:
        raise L.LaunchError("D0 rebuilt at registration differs from the frozen D0 hash")
    verify_labels(d0, cases)
    write_json_atomic(run_root / "datasets" / "D0.json", d0)
    shutil.copy2(root / CONFIG_REL, run_root / "prep" / Path(CONFIG_REL).name)
    shutil.copy2(root / DOC_REL, run_root / "prep" / Path(DOC_REL).name)
    (run_root / "prep" / "dataset_D0_identity.json").write_text(json.dumps(ident, indent=1, sort_keys=True) + "\n")
    auth = {"card": CARD, "amendment": AMENDMENT, "prep_commit": prep_commit, "records": list(RECORDS), "authorization_text": authorization_text,
            "authorization_text_sha256": hashlib.sha256(authorization_text.encode()).hexdigest(), "authorized_training_runs": 3,
            "not_authorized": ["extra seeds", "extra epochs or rounds", "mid-checkpoint selection", "PPO checkpoint loading", "single-method change", "retry of a started run", "new algorithm"],
            "registered": L.utc_now()}
    L.write_json_atomic(run_root / "registration" / "authorization.json", auth)
    configs, ledger, cfg_docs = {}, {}, []
    for rid in RUN_IDS:
        cfg = {"run_id": rid, "method": METHOD_OF[rid], "seed": 0, "train_dev_split": str(split), "train_dev_split_sha256": L.sha256_file(split), "out_dir": str(run_root / "runs" / rid),
               "dataset_dir": str(run_root / "datasets"), "d0_path": str(run_root / "datasets" / "D0.json"), "d0_sha256": ident["sha256"], "loss": LOSS_NAME, "optimizer": dict(OPT),
               "stages": [dict(s) for s in STAGES], "aggregation_rounds": 2, "prep_commit": prep_commit, "half_life": half_life, "init": "FROM_SCRATCH"}
        cfg_docs.append(cfg)
        path = run_root / "registration" / (rid + ".yaml")
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        os.chmod(path, 0o444)
        configs[rid] = {"path": str(path), "sha256": L.sha256_file(path)}
        ledger[rid] = {"method": METHOD_OF[rid], "seed": 0, "status": "NOT_STARTED", "config": configs[rid]}
    validate_identical_supervision(cfg_docs)
    L.write_json_atomic(run_root / "registration" / "launch_state.json", {"card": CARD, "amendment": AMENDMENT, "attempts_used": 0, "attempts_cap": 3, "runs": ledger, "created": L.utc_now()})
    L.write_json_atomic(run_root / "registration" / "launch_plan.json", {"card": CARD, "amendment": AMENDMENT, "prep_commit": prep_commit, "configs": configs, "run_root": str(run_root)})
    return run_root


def train(root, run_root, run_id, gpu):
    """Run one registered imitation training. CUDA_VISIBLE_DEVICES must already pin the GPU (set by the CLI before torch is imported)."""
    from .imitation import validate_identical_supervision, run_imitation
    root, run_root = Path(root).resolve(), Path(run_root).resolve()
    plan = json.loads((run_root / "registration" / "launch_plan.json").read_text())
    check_source_identity_il(root, plan["prep_commit"], exact=True)
    docs = []
    for rid, entry in plan["configs"].items():
        if L.sha256_file(entry["path"]) != entry["sha256"]:
            raise L.LaunchError("run config changed after registration")
        docs.append(yaml.safe_load(Path(entry["path"]).read_text()))
    validate_identical_supervision(docs)
    cfg = yaml.safe_load(Path(plan["configs"][run_id]["path"]).read_text())
    if L.sha256_file(cfg["train_dev_split"]) != cfg["train_dev_split_sha256"]:
        raise L.LaunchError("train split changed after registration")
    if L.sha256_file(cfg["d0_path"]) != L.sha256_file(run_root / "datasets" / "D0.json"):
        raise L.LaunchError("D0 changed")
    st = os.statvfs(str(run_root))
    if st.f_bavail * st.f_frsize < L.MIN_FREE_BYTES:
        raise L.LaunchError("storage start gate")
    L.reserve(run_root, run_id, os.getpid(), gpu)
    cfg = dict(cfg, device="cuda:0")
    try:
        acct = run_imitation(cfg)
        L.finish(run_root, run_id, "COMPLETE", accounting=acct)
        return acct
    except BaseException as exc:                                    # recorded, never swallowed; no automatic rerun
        L.finish(run_root, run_id, "STOPPED", error="%s: %s" % (type(exc).__name__, exc), traceback=traceback.format_exc()[-3000:])
        raise
